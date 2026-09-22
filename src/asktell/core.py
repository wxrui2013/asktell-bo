"""Ask–tell 接口：贝叶斯优化只推荐参数，外部程序负责执行并回传结果。

Python:

    from bo_asktell import AskTellBO

    bo = AskTellBO(bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)}, q=5, seed=0)
    for _ in range(20):
        batch = bo.suggest()                          # [{"id", "values": {…}}]
        values = [run_external(item["values"]) for item in batch]
        bo.observe(batch, values)

命令行（在 asktell/ 目录下；状态默认 artifacts/asktell_state.json）:

    python bo_asktell.py init --param x -5 5 --param y -5 5 --q 5
    python bo_asktell.py suggest -o artifacts/next.json
    python eval_external.py artifacts/next.json artifacts/results.json
    python bo_asktell.py observe artifacts/results.json
    python bo_asktell.py status

一次跑完若干轮:

    python bo_asktell.py run --rounds 10 --q 5 \\
        --param x -5 5 --param y -5 5 \\
        --eval "python eval_external.py"

或直接: python run_demo.py
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

HERE = Path.cwd()
DEFAULT_STATE = Path.cwd() / "artifacts" / "asktell_state.json"
torch.set_num_threads(max(1, min(4, torch.get_num_threads() or 4)))


def _latin(n_pts: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """单位立方体上的拉丁超立方。"""
    if n_pts <= 0:
        return np.zeros((0, d), dtype=np.float64)
    u = (np.arange(n_pts)[:, None] + rng.random((n_pts, d))) / n_pts
    for j in range(d):
        rng.shuffle(u[:, j])
    return u


def _fit_gp(X: np.ndarray, y: np.ndarray, cache: dict) -> SingleTaskGP:
    """最小化 y：GP 拟合的是 -y。X 为原始参数尺度。"""
    d = int(X.shape[1])
    train_X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.double)
    train_Y = torch.as_tensor(-np.ascontiguousarray(y), dtype=torch.double).unsqueeze(-1)
    cov = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=d,
            lengthscale_constraint=Interval(0.02, 1.5),
            lengthscale_prior=LogNormalPrior(math.log(0.2), 0.8),
        )
    )
    lik = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-2))
    model = SingleTaskGP(
        train_X,
        train_Y,
        covar_module=cov,
        likelihood=lik,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    round_i = int(cache.get("round", 0))
    refit = cache.get("ls") is None or round_i < 3 or round_i % 4 == 0
    if cache.get("ls") is not None and int(cache["ls"].shape[-1]) == d:
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale.copy_(cache["ls"])
            model.covar_module.outputscale.copy_(cache["os"])
            model.likelihood.noise.copy_(cache["noise"])
    if refit:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 28}})
        except RuntimeError:
            pass
        cache["ls"] = model.covar_module.base_kernel.lengthscale.detach().clone()
        cache["os"] = model.covar_module.outputscale.detach().clone()
        cache["noise"] = model.likelihood.noise.detach().clone()
    cache["round"] = round_i + 1
    model.eval()
    return model


def _subsample(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_n: int = 200):
    if len(y) <= max_n:
        return X, y
    order = np.argsort(y)
    n_best = min(50, max_n // 4)
    best = order[:n_best]
    rest = order[n_best:]
    extra = rng.choice(rest, size=max_n - n_best, replace=False)
    idx = np.concatenate([best, extra])
    return X[idx], y[idx]


class AskTellBO:
    """批量 ask–tell 贝叶斯优化（联合后验 Thompson）。默认最小化 value。"""

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        q: int = 5,
        n_init: int | None = None,
        seed: int = 0,
    ) -> None:
        if not bounds:
            raise ValueError("至少需要一个参数及其上下界")
        self.names = list(bounds)
        self.lo = np.array([float(bounds[n][0]) for n in self.names], dtype=np.float64)
        self.hi = np.array([float(bounds[n][1]) for n in self.names], dtype=np.float64)
        if np.any(self.hi <= self.lo):
            raise ValueError("每个参数的上界必须大于下界")
        self.q = max(1, int(q))
        self.n_init = int(n_init) if n_init is not None else 4 * self.q
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.round = 0
        self._next_id = 0
        self.X = np.zeros((0, len(self.names)), dtype=np.float64)
        self.y = np.zeros((0,), dtype=np.float64)
        self.pending: list[dict] = []
        self.history: list[dict] = []
        self._gp_cache: dict = {}
        unit = _latin(self.n_init, len(self.names), self.rng)
        self._init_queue = self.lo + unit * (self.hi - self.lo)

    @property
    def d(self) -> int:
        return len(self.names)

    def _row_to_item(self, row: np.ndarray) -> dict:
        values = {name: float(row[i]) for i, name in enumerate(self.names)}
        item = {"id": str(self._next_id), "values": values}
        self._next_id += 1
        return item

    def suggest(self) -> list[dict]:
        """给出本轮 q 组推荐参数。若上一轮还没 observe，原样返回 pending。"""
        if self.pending:
            return list(self.pending)
        q_now = self.q
        rows: list[np.ndarray] = []
        while len(rows) < q_now and len(self._init_queue) > 0:
            rows.append(self._init_queue[0])
            self._init_queue = self._init_queue[1:]
        n_gp = q_now - len(rows)
        if n_gp > 0:
            if len(self.y) == 0:
                extra = self.lo + self.rng.random((n_gp, self.d)) * (self.hi - self.lo)
                rows.extend(list(extra))
            else:
                rows.extend(list(self._thompson(n_gp)))
        self.pending = [self._row_to_item(np.asarray(r, dtype=np.float64)) for r in rows[:q_now]]
        return list(self.pending)

    def _thompson(self, n_want: int) -> np.ndarray:
        Xs, ys = _subsample(self.X, self.y, self.rng)
        model = _fit_gp(Xs, ys, self._gp_cache)
        n_cand = min(4096, max(512, n_want * 256))
        unit = self.rng.random((n_cand, self.d))
        cand = self.lo + unit * (self.hi - self.lo)
        xstar = torch.as_tensor(cand, dtype=torch.double)
        n_draw = int(min(len(cand), max(n_want * 6, 12)))
        with torch.no_grad():
            selected = MaxPosteriorSampling(model, replacement=False)(xstar, num_samples=n_draw)
        picked = selected.detach().cpu().numpy()
        out = []
        for row in picked:
            if any(np.allclose(row, p, atol=1e-8) for p in out):
                continue
            out.append(row)
            if len(out) >= n_want:
                break
        while len(out) < n_want:
            out.append(self.lo + self.rng.random(self.d) * (self.hi - self.lo))
        return np.asarray(out[:n_want], dtype=np.float64)

    def observe(self, batch: list[dict] | None = None, values: list[float] | None = None) -> None:
        """登记外部程序返回的 value（默认越小越好）。"""
        if not self.pending:
            raise RuntimeError("没有待观察的推荐。请先调用 suggest。")
        if values is not None:
            if len(values) != len(self.pending):
                raise ValueError(f"期望 {len(self.pending)} 个 value，收到 {len(values)}")
            pairs = list(zip(self.pending, [float(v) for v in values], strict=True))
        elif batch is not None:
            by_id = {}
            for item in batch:
                if "id" not in item or "value" not in item:
                    raise ValueError("results 里每一项需要 id 和 value")
                by_id[str(item["id"])] = float(item["value"])
            try:
                pairs = [(p, by_id[str(p["id"])]) for p in self.pending]
            except KeyError as exc:
                raise ValueError(f"结果里缺少 id={exc}") from exc
        else:
            raise ValueError("请传入 batch（含 id/value）或 values 列表")

        xs = []
        ys = []
        for item, value in pairs:
            row = np.array([item["values"][n] for n in self.names], dtype=np.float64)
            xs.append(row)
            ys.append(value)
            rec = {
                "round": self.round,
                "id": item["id"],
                "values": item["values"],
                "value": value,
            }
            self.history.append(rec)
        self.X = np.vstack([self.X, np.asarray(xs)]) if len(self.X) else np.asarray(xs)
        self.y = np.concatenate([self.y, np.asarray(ys, dtype=np.float64)])
        self.pending = []
        self.round += 1

    def best(self) -> dict | None:
        if len(self.y) == 0:
            return None
        i = int(np.argmin(self.y))
        return {
            "values": {n: float(self.X[i, j]) for j, n in enumerate(self.names)},
            "value": float(self.y[i]),
            "index": i,
        }

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
            "q": self.q,
            "n_init": self.n_init,
            "seed": self.seed,
            "round": self.round,
            "next_id": self._next_id,
            "X": self.X.tolist(),
            "y": self.y.tolist(),
            "pending": self.pending,
            "history": self.history,
            "init_queue": self._init_queue.tolist(),
            "rng_state": _rng_state_to_json(self.rng),
        }

    @classmethod
    def from_dict(cls, blob: dict) -> AskTellBO:
        bounds = {
            n: (float(lo), float(hi))
            for n, lo, hi in zip(blob["names"], blob["lo"], blob["hi"], strict=True)
        }
        bo = cls(bounds=bounds, q=int(blob["q"]), n_init=int(blob["n_init"]), seed=int(blob["seed"]))
        bo.round = int(blob["round"])
        bo._next_id = int(blob.get("next_id", 0))
        bo.X = np.asarray(blob["X"], dtype=np.float64).reshape(-1, bo.d)
        bo.y = np.asarray(blob["y"], dtype=np.float64).reshape(-1)
        bo.pending = list(blob.get("pending") or [])
        bo.history = list(blob.get("history") or [])
        bo._init_queue = np.asarray(blob.get("init_queue") or [], dtype=np.float64).reshape(-1, bo.d)
        if "rng_state" in blob:
            bo.rng = _rng_from_json(blob["rng_state"])
        return bo

    def save(self, path: str | Path = DEFAULT_STATE) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_STATE) -> AskTellBO:
        path = Path(path)
        blob = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(blob)


def _rng_state_to_json(rng: np.random.Generator) -> dict:
    st = rng.bit_generator.state

    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    return conv(st)


def _rng_from_json(state: dict) -> np.random.Generator:
    bg_name = state.get("bit_generator", "PCG64")
    bg = getattr(np.random, bg_name)()
    restored = dict(state)
    inner = dict(restored.get("state", {}))
    if "state" in inner:
        inner["state"] = int(inner["state"])
    if "inc" in inner:
        inner["inc"] = int(inner["inc"])
    restored["state"] = inner
    bg.state = restored
    return np.random.Generator(bg)


def suggest_payload(bo: AskTellBO, batch: list[dict]) -> dict:
    return {
        "round": bo.round,
        "q": len(batch),
        "minimize": True,
        "param_names": bo.names,
        "bounds": {n: [float(bo.lo[i]), float(bo.hi[i])] for i, n in enumerate(bo.names)},
        "params": batch,
    }


def parse_results(blob: dict | list, pending: list[dict]) -> list[dict]:
    if isinstance(blob, list):
        if blob and isinstance(blob[0], (int, float)):
            if len(blob) != len(pending):
                raise ValueError(f"期望 {len(pending)} 个数值，收到 {len(blob)}")
            return [{"id": p["id"], "value": float(v)} for p, v in zip(pending, blob, strict=True)]
        return blob
    if "results" in blob:
        return parse_results(blob["results"], pending)
    if "value" in blob and len(pending) == 1:
        return [{"id": pending[0]["id"], "value": float(blob["value"])}]
    raise ValueError("结果 JSON 需要 results 列表，或与推荐等长的数值数组")


def _load_or_die(state_path: Path) -> AskTellBO:
    if not state_path.exists():
        raise SystemExit(f"还没有状态文件 {state_path}，请先运行 init 或 run")
    return AskTellBO.load(state_path)


def _print_json(obj: dict, out: str) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if out in ("", "-", None):
        sys.stdout.write(text + "\n")
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"已写入 {path}", file=sys.stderr)


def _eval_command(cmd: str, payload: dict, timeout: float | None = None) -> dict:
    proc = subprocess.run(
        shlex.split(cmd),
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"外部程序退出码 {proc.returncode}: {cmd}")
    raw = proc.stdout.strip()
    if not raw:
        sys.stderr.write(proc.stderr)
        raise SystemExit("外部程序没有在 stdout 写出 JSON 结果")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"外部程序 stdout 不是 JSON: {raw[:300]}") from exc


def _cmd_init(args: argparse.Namespace) -> None:
    bounds = _bounds_from_args(args)
    bo = AskTellBO(bounds=bounds, q=args.q, n_init=args.n_init, seed=args.seed)
    path = bo.save(args.state)
    print(f"已初始化 {path}  参数 {bo.names}  q={bo.q}  n_init={bo.n_init}")


def _cmd_suggest(args: argparse.Namespace) -> None:
    bo = _load_or_die(Path(args.state))
    batch = bo.suggest()
    bo.save(args.state)
    _print_json(suggest_payload(bo, batch), args.out)


def _cmd_observe(args: argparse.Namespace) -> None:
    bo = _load_or_die(Path(args.state))
    if args.results in ("", "-"):
        blob = json.loads(sys.stdin.read())
    else:
        blob = json.loads(Path(args.results).read_text(encoding="utf-8"))
    results = parse_results(blob, bo.pending)
    bo.observe(results)
    bo.save(args.state)
    best = bo.best()
    last = bo.history[-len(results) :]
    for rec in last:
        print(f"  id={rec['id']}  value={rec['value']:.6g}  {rec['values']}")
    if best:
        print(f"目前最好 value={best['value']:.6g}  {best['values']}  已观察 {len(bo.y)} 点 / {bo.round} 轮")


def _cmd_status(args: argparse.Namespace) -> None:
    bo = _load_or_die(Path(args.state))
    best = bo.best()
    print(f"参数 {bo.names}")
    print(f"轮次 {bo.round}  已观察 {len(bo.y)}  待观察 {len(bo.pending)}  q={bo.q}")
    if best:
        print(f"最好 value={best['value']:.6g}  {best['values']}")
    else:
        print("还没有观察结果")
    if bo.pending:
        print("pending:")
        for item in bo.pending:
            print(f"  id={item['id']}  {item['values']}")


def _cmd_run(args: argparse.Namespace) -> None:
    state = Path(args.state)
    if state.exists() and not args.reset:
        bo = AskTellBO.load(state)
        print(f"继续 {state}，已有 {len(bo.y)} 点")
    else:
        bounds = _bounds_from_args(args)
        bo = AskTellBO(bounds=bounds, q=args.q, n_init=args.n_init, seed=args.seed)
        bo.save(state)
        print(f"新建 {state}  参数 {bo.names}  q={bo.q}")
    t0 = time.perf_counter()
    for _ in range(args.rounds):
        batch = bo.suggest()
        payload = suggest_payload(bo, batch)
        blob = _eval_command(args.eval, payload)
        results = parse_results(blob, batch)
        bo.observe(results)
        bo.save(state)
        best = bo.best()
        last_v = ", ".join(f"{r['value']:.4g}" for r in bo.history[-len(batch) :])
        print(
            f"第 {bo.round} 轮  本轮 {last_v}  最好 {best['value']:.4g}  {best['values']}",
            flush=True,
        )
    dt = time.perf_counter() - t0
    best = bo.best()
    print(f"结束：{bo.round} 轮 / {len(bo.y)} 点  {dt:.1f}s  最好 value={best['value']:.6g}  {best['values']}")


def _bounds_from_args(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    params = getattr(args, "param", None) or []
    if not params:
        raise SystemExit("请用 --param NAME LO HI 声明至少一个参数，例如 --param x 0 1 --param y 0 1")
    bounds: dict[str, tuple[float, float]] = {}
    for name, lo, hi in params:
        bounds[name] = (float(lo), float(hi))
    return bounds


def _add_state(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--state", default=str(DEFAULT_STATE), help="状态文件，默认 artifacts/asktell_state.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ask–tell 贝叶斯优化：推荐参数 → 外部程序 → 回传结果")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_param(sp: argparse.ArgumentParser) -> None:
        _add_state(sp)
        sp.add_argument(
            "--param",
            nargs=3,
            action="append",
            metavar=("NAME", "LO", "HI"),
            help="参数名与上下界，可重复。例：--param x 0 1 --param y 0 1",
        )
        sp.add_argument("--q", type=int, default=5, help="每轮并行推荐点数")
        sp.add_argument("--n-init", type=int, default=None, help="拉丁初值点数，默认 4*q")
        sp.add_argument("--seed", type=int, default=0)

    init = sub.add_parser("init", help="新建一次优化（清空状态）")
    add_param(init)

    sug = sub.add_parser("suggest", help="写出本轮推荐参数 JSON")
    _add_state(sug)
    sug.add_argument("-o", "--out", default="-", help="输出路径，默认 stdout")

    obs = sub.add_parser("observe", help="读入外部程序的结果 JSON")
    _add_state(obs)
    obs.add_argument("results", help="结果文件，或 - 表示 stdin")

    st = sub.add_parser("status", help="打印当前最好点")
    _add_state(st)

    run = sub.add_parser("run", help="循环：suggest → 外部程序 → observe")
    add_param(run)
    run.add_argument("--rounds", type=int, required=True, help="迭代轮数")
    run.add_argument(
        "--eval",
        required=True,
        help="外部程序命令。stdin 为推荐 JSON，stdout 须为结果 JSON",
    )
    run.add_argument("--reset", action="store_true", help="忽略已有状态，重新开始")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "init":
        _cmd_init(args)
    elif args.cmd == "suggest":
        _cmd_suggest(args)
    elif args.cmd == "observe":
        _cmd_observe(args)
    elif args.cmd == "status":
        _cmd_status(args)
    elif args.cmd == "run":
        _cmd_run(args)
    else:
        parser.error(args.cmd)


if __name__ == "__main__":
    main()

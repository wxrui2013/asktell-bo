"""示例外部程序：读推荐 JSON，算出标量 value，写回结果 JSON。

本文件夹可独立运行，不依赖仓库根目录的格点场。默认 Himmelblau。

    python eval_external.py artifacts/next.json artifacts/results.json
    python eval_external.py --fn wells artifacts/next.json artifacts/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_payload(src: str | None) -> dict:
    if src in (None, "", "-"):
        return json.loads(sys.stdin.read())
    return json.loads(Path(src).read_text(encoding="utf-8"))


def _write_results(blob: dict, dest: str | None) -> None:
    text = json.dumps(blob, ensure_ascii=False, indent=2)
    if dest in (None, "", "-"):
        sys.stdout.write(text + "\n")
        return
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _xy(values: dict) -> tuple[float, float]:
    if "x" in values and "y" in values:
        return float(values["x"]), float(values["y"])
    keys = list(values)
    if len(keys) < 2:
        raise ValueError("至少需要两个参数 x, y")
    return float(values[keys[0]]), float(values[keys[1]])


def eval_himmelblau(values: dict) -> float:
    """定义域约 [-5, 5]，四个全局最小约为 0。"""
    x, y = _xy(values)
    return float((x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2)


def eval_wells(values: dict) -> float:
    """[0, 1]² 上的多井函数：若干浅坑 + 一个更深的全局最小。"""
    x, y = _xy(values)
    z = 8.5
    wells = (
        (0.42, 0.27, 7.4, 0.055),
        (0.18, 0.72, 4.2, 0.07),
        (0.78, 0.22, 3.8, 0.06),
        (0.62, 0.68, 4.6, 0.065),
        (0.30, 0.48, 3.5, 0.05),
    )
    for cx, cy, depth, sig in wells:
        z -= depth * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sig * sig))
    return float(z)


def evaluate_batch(payload: dict, fn: str) -> dict:
    params = payload.get("params")
    if not params:
        raise ValueError("推荐 JSON 缺少 params")
    fn_map = {"himmelblau": eval_himmelblau, "wells": eval_wells}
    if fn not in fn_map:
        raise ValueError(f"未知 --fn {fn}，可选 {sorted(fn_map)}")
    f = fn_map[fn]
    results = [{"id": str(item["id"]), "value": f(item["values"])} for item in params]
    return {"results": results}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="示例外部评测：推荐参数 → 标量 value")
    parser.add_argument("suggest", nargs="?", default="-", help="推荐 JSON，默认 stdin")
    parser.add_argument("results", nargs="?", default="-", help="结果 JSON，默认 stdout")
    parser.add_argument(
        "--fn",
        default="himmelblau",
        choices=("himmelblau", "wells"),
        help="himmelblau: [-5,5]；wells: [0,1] 多井",
    )
    args = parser.parse_args(argv)
    payload = _load_payload(args.suggest)
    blob = evaluate_batch(payload, args.fn)
    _write_results(blob, args.results)


if __name__ == "__main__":
    main()

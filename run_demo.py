"""跑一遍 ask–tell 全过程，并把中间文件留在 artifacts/ 里方便查看。

    cd asktell
    python run_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
STATE = ART / "asktell_state.json"
NEXT = ART / "next.json"
RESULTS = ART / "results.json"
HISTORY = ART / "history.json"
PLOT = ART / "history.png"
PY = sys.executable


def _run(args: list[str]) -> None:
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=str(HERE), check=True)


def _setup_cjk_font() -> None:
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    for path in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _plot(history: list[dict], best_value: float, best_xy: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_cjk_font()

    xs = [h["values"]["x"] for h in history]
    ys = [h["values"]["y"] for h in history]
    vs = [h["value"] for h in history]
    rounds = [h["round"] for h in history]
    best_curve = np.minimum.accumulate(vs)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
    ax = axes[0]
    sc = ax.scatter(xs, ys, c=rounds, cmap="autumn_r", s=36, edgecolors="k", linewidths=0.3)
    ax.scatter([best_xy["x"]], [best_xy["y"]], marker="*", s=180, c="#ff3d8a", edgecolors="white", zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("探测位置（颜色=轮次）")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="轮次")

    ax = axes[1]
    ax.plot(np.arange(1, len(vs) + 1), vs, "o", ms=4, color="0.55", label="每次观测")
    ax.plot(np.arange(1, len(best_curve) + 1), best_curve, "-", color="#1f77b4", lw=2, label="迄今最好")
    ax.axhline(best_value, color="#ff3d8a", ls="--", lw=0.8)
    ax.set_xlabel("探测序号")
    ax.set_ylabel("value（越小越好）")
    ax.set_title("收敛")
    ax.legend(fontsize=8)
    fig.savefig(PLOT, dpi=140)
    plt.close(fig)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    print("工作目录", HERE)

    print("\n=== 1. 文件协议：init → suggest → 外部程序 → observe ===")
    _run(
        [
            PY,
            "bo_asktell.py",
            "init",
            "--state",
            str(STATE),
            "--param",
            "x",
            "-5",
            "5",
            "--param",
            "y",
            "-5",
            "5",
            "--q",
            "2",
            "--n-init",
            "4",
            "--seed",
            "0",
        ]
    )
    _run([PY, "bo_asktell.py", "suggest", "--state", str(STATE), "-o", str(NEXT)])
    _run([PY, "eval_external.py", "--fn", "himmelblau", str(NEXT), str(RESULTS)])
    _run([PY, "bo_asktell.py", "observe", "--state", str(STATE), str(RESULTS)])
    _run([PY, "bo_asktell.py", "status", "--state", str(STATE)])

    print("\n=== 2. 再自动迭代若干轮（stdin/stdout 喂给外部程序）===")
    _run(
        [
            PY,
            "bo_asktell.py",
            "run",
            "--state",
            str(STATE),
            "--rounds",
            "7",
            "--eval",
            f"{PY} eval_external.py --fn himmelblau",
        ]
    )

    blob = json.loads(STATE.read_text(encoding="utf-8"))
    HISTORY.write_text(json.dumps(blob["history"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_i = int(np.argmin(blob["y"]))
    best_xy = {n: blob["X"][best_i][j] for j, n in enumerate(blob["names"])}
    best_v = float(blob["y"][best_i])
    _plot(blob["history"], best_v, best_xy)

    print("\n=== 生成的文件（打开这些就能看全过程）===")
    for p in (NEXT, RESULTS, STATE, HISTORY, PLOT):
        print(f"  {p.relative_to(HERE)}  ({p.stat().st_size} bytes)")
    print(f"最好 value={best_v:.4g}  {best_xy}")


if __name__ == "__main__":
    main()

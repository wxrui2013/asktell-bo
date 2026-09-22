# asktell-bo

Ask–tell 贝叶斯优化：**只推荐参数**，外部程序负责执行并返回标量 `value`（默认越小越好），再观察、再推荐。

私有库。安装：

```bash
pip install -e .[demo]
```

## 命令行

```bash
python -m asktell init --param x -5 5 --param y -5 5 --q 2
python -m asktell suggest -o artifacts/next.json
python -m asktell.eval_external artifacts/next.json artifacts/results.json
python -m asktell observe artifacts/results.json
python -m asktell status
```

等价入口：`asktell` / `asktell-eval`。

一次多轮（stdin=推荐 JSON，stdout=结果 JSON）：

```bash
python -m asktell run --rounds 8 --q 2 --param x -5 5 --param y -5 5 \
    --eval "asktell-eval"
```

演示：

```bash
python run_demo.py
```

## Python

```python
from asktell import AskTellBO

def run_external(params: dict) -> float:
    ...

bo = AskTellBO(bounds={"x": (-5, 5), "y": (-5, 5)}, q=2, seed=0)
for _ in range(10):
    batch = bo.suggest()
    values = [run_external(item["values"]) for item in batch]
    bo.observe(batch, values)
print(bo.best())
```

## JSON 协议

推荐：

```json
{"params": [{"id": "0", "values": {"x": 1.2, "y": -0.4}}]}
```

外部程序必须回：

```json
{"results": [{"id": "0", "value": 12.3}]}
```

`id` 必须对得上。

## 依赖

CPU 版 PyTorch + BoTorch + GPyTorch + NumPy。演示图需要 matplotlib（`pip install -e .[demo]`）。

# 行优先与列优先窗口布局 demo

同一张图有 4×8 个窗口、4 个 rank。颜色表示窗口归属，数字始终是行优先编号的全局窗口 ID；改变的是 owner 和本地扫描顺序。

![行优先与列优先窗口布局](../../docs/figures/owner_layouts.png)

窗口边长为 4，向上、向左各 shift 2 个 token。深色部分要跨 rank 发送，浅色部分留在本卡；包含周期回绕，不含入口和出口。

![一次内部 shift 的通信区域](../../docs/figures/shift_traffic.png)

本例行优先发送 256 个 token，列优先发送 128 个；两侧均为 512 个 token，每个 rank 均持有 8 个窗口。

## 直接调用

进入本目录后运行，不需要 PyTorch：

```bash
python -B window_layout_demo.py
```

```python
from window_layout_demo import build_layout, shift_plan

for order in ("row", "column"):
    layout = build_layout(4, 8, 4, order, "paper_floor")
    plan = shift_plan(layout, window_size=4, shift_y=2, shift_x=2)
    print(order, layout.window_ids_by_rank[0], plan.remote_tokens)
```

输出：

```text
row    (0, 1, 2, 3, 4, 5, 6, 7)     256
column (0, 8, 16, 24, 1, 9, 17, 25) 128
```

## 重新画图

仍在本目录执行，需要 numpy、matplotlib 和中文字体：

```bash
python -B render_layouts.py --output-dir ../../docs/figures
```

可通过 `TERRA_PLOT_FONT` 指定中文字体文件。画图和布局计划是两个独立入口；上面的计划命令默认输出 JSON，绘图命令输出 PNG／SVG。

## 实际 P2P 调用

下面启动 4 个真实 CPU/Gloo 进程，调用通用运行时的 `TokenLayout`／`TokenRoute` 并检查前向值与加权反向梯度，需要可用的 PyTorch/Gloo：

```bash
python -B demo_token_route.py --runtime-root ../../src --output p2p_demo_output
```

输出目录需要尚不存在。这是路由正确性示例，不是训练或 GPU/NPU 性能测试。

[完整实验说明](../../README.md) · [通信量证明](../../docs/window_layouts/README.md)

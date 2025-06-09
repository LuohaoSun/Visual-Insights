# Visual Insights

通用可视化和可解释性工具, 目标是在所有项目中应用, 避免重复造轮子.

## 构建与安装

```bash
hatch clean
hatch build
```

```bash
uv pip install xxx.whl
```

## 如何使用:

```python
import visual_insights as vi

actual = ...
predicted = ...
vi.plot_actual_vs_predicted(actual, predicted)
```

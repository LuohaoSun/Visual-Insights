# Visual Insights

通用可视化和可解释性工具, 目标是在所有项目中应用, 避免重复造轮子.

## 安装

```bash
uv pip install https://github.com/LuohaoSun/Visual-Insights.git
```

## 如何使用:

```python
import visual_insights as vi

actual = ...
predicted = ...
vi.plot_actual_vs_predicted(actual, predicted)
```

## 开发

大量使用 AI 生成, 无开发文档.

## 构建

```bash
hatch clean
hatch build
```

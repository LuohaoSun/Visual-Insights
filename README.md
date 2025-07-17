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

大量使用 AI 生成, 目前正在重构中.

### API 约定

- 所有绘图函数统一以`plot_`开头, 并返回`Figure`对象
- 数据相关数据参数命名方式:
  - 如果需要同时传入横轴和纵轴数据, 则使用`input`和`target`作为参数名
  - 如果只需要传入一个数据序列, 则使用`data`作为参数名
- 图表相关参数命名: 统一使用`title`、`input_name`、`target_name`、`figsize`和`show`作为参数名
- 图表相关参数置于函数参数列表的最后

## 构建

```bash
hatch clean
hatch build
```

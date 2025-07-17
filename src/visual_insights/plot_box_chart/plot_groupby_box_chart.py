import pandas as pd
from matplotlib.figure import Figure

from ._core import _convert_long_to_wide_for_boxplot, _plot_box_from_dataframe


def plot_groupby_box_chart(
    data: pd.DataFrame,
    group_by: str | list[str],
    value_column: str,
    title: str | None = None,
    input_name: str | None = None,
    target_name: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    绘制分组箱线图. DataFrame友好接口,
    直接传入包含分组列和数值列的DataFrame, 自动完成分组和绘图

    参数:
        data: 包含分组和数值的 DataFrame
        group_by: 分组列名
        value_column: 数值列名
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表

    返回:
        matplotlib Figure 对象
    """
    data_for_plotting = _convert_long_to_wide_for_boxplot(
        data=data,
        group_by=group_by,
        value_column=value_column,
    )

    final_input_name = input_name or (
        group_by if isinstance(group_by, str) else "Group"
    )
    final_target_name = target_name or value_column

    return _plot_box_from_dataframe(
        data=data_for_plotting,
        title=title,
        input_name=final_input_name,
        target_name=final_target_name,
        figsize=figsize,
        show=show,
    )

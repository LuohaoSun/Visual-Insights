from typing import Literal

import pandas as pd
from matplotlib.figure import Figure

from visual_insights.common.to_numpy import parse_1darray
from visual_insights.common.typing import ArrayLike

from ._core import _convert_long_to_wide_for_boxplot, _plot_box_from_dataframe


def _prepare_dataframe_from_binned_data(
    input_data: ArrayLike,
    target_data: ArrayLike,
    n_bins: int,
    bin_method: Literal["cut", "qcut"],
) -> pd.DataFrame:
    """为 binned_box_chart 准备数据。"""
    input_array = parse_1darray(input_data, parse_data_name=False)
    target_array = parse_1darray(target_data, parse_data_name=False)

    temp_df = pd.DataFrame({"_input_": input_array, "_target_": target_array})

    if bin_method == "cut":
        temp_df["_bin_"] = pd.cut(temp_df["_input_"], bins=n_bins, duplicates="drop")
    elif bin_method == "qcut":
        temp_df["_bin_"] = pd.qcut(temp_df["_input_"], q=n_bins, duplicates="drop")

    return _convert_long_to_wide_for_boxplot(temp_df, "_bin_", "_target_")


def plot_binned_box_chart(
    input: ArrayLike,
    target: ArrayLike,
    n_bins: int = 10,
    bin_method: Literal["cut", "qcut"] = "cut",
    title: str | None = None,
    input_name: str | None = None,
    target_name: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """先分箱再绘制箱线图. 典型用法:
    传入x序列和y序列, 将x序列分为多个bins, 每个bin内的y序列绘制为一个box

    Args:
        input: 横轴序列
        target: 纵轴序列
        n_bins: 分箱数量
        bin_method: 分箱方法
            - "cut": 等宽分箱, 默认
            - "qcut": 等频分箱
        title: 图表标题
        input_name: 横轴名称
        target_name: 纵轴名称
        figsize: 图表大小
        show: 是否立即显示图表
    """
    data_for_plotting = _prepare_dataframe_from_binned_data(
        input_data=input,
        target_data=target,
        n_bins=n_bins,
        bin_method=bin_method,
    )

    final_input_name = input_name or "Binned Input"
    final_target_name = target_name or "Target Value"

    return _plot_box_from_dataframe(
        data=data_for_plotting,
        title=title,
        input_name=final_input_name,
        target_name=final_target_name,
        figsize=figsize,
        show=show,
    )

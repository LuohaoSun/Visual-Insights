from typing import overload

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from visual_insights.common.typing import ArrayLike

from ._core import _plot_box_from_dataframe, _prepare_dataframe_from_input_target


@overload
def plot_box_chart(
    *,
    data: pd.DataFrame,
    title: str | None = None,
    input_name: str | None = None,
    target_name: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """绘制箱线图

    Args:
        data: 数据 shape: (n_samples, n_features). 将使用列名作为横轴刻度.
        title: 图表标题
        input_name: 横轴名称
        target_name: 纵轴名称
        figsize: 图表大小
        show: 是否立即显示图表
    """
    ...


@overload
def plot_box_chart(
    *,
    input: ArrayLike,
    target: list[ArrayLike] | np.ndarray,
    title: str | None = None,
    input_name: str | None = None,
    target_name: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """绘制箱线图

    Args:
        input: 类别名称序列, 将作为横轴刻度
        target: 每个类别对应的数值序列.
            如果是list[ArrayLike], 则每个元素对应一个类别的数值(支持每个类别不等长)
            如果是ArrayLike, shape: (n_samples, n_features)
        input_name: 横轴名称
        target_name: 纵轴名称
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表
    """
    ...


def plot_box_chart(
    title: str | None = None,
    input_name: str | None = None,
    target_name: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
    **kwargs,
) -> Figure:
    """
    绘制箱线图。

    支持两种调用方式:
    1.  plot_box_chart(data: pd.DataFrame, ...)
    2.  plot_box_chart(input: ArrayLike, target: list[ArrayLike] | np.ndarray, ...)
    """
    if "input" in kwargs and "target" in kwargs:
        # Mode 1: Prepare data first, then plot
        data_for_plotting = _prepare_dataframe_from_input_target(
            input_data=kwargs["input"],
            target_data=kwargs["target"],
        )
    elif "data" in kwargs:
        # Mode 2: Data is already prepared
        data_for_plotting = kwargs["data"]
        if not isinstance(data_for_plotting, pd.DataFrame):
            raise TypeError("'data' argument must be a pandas DataFrame.")
    else:
        raise TypeError(
            "plot_box_chart requires either 'data' (DataFrame) or both 'input' and 'target' arguments."
        )

    return _plot_box_from_dataframe(
        data=data_for_plotting,
        title=title,
        input_name=input_name,
        target_name=target_name,
        figsize=figsize,
        show=show,
    )

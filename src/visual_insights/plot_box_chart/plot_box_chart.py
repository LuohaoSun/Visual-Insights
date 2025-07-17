from typing import overload

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from visual_insights.common.to_numpy import parse_1darray
from visual_insights.common.typing import ArrayLike

from ._core import _plot_box_from_dataframe


def _prepare_dataframe_from_input_target(
    input_data: ArrayLike,
    target_data: list[ArrayLike] | np.ndarray,
) -> pd.DataFrame:
    """Converts input/target data into a single DataFrame for plotting."""
    input_array = parse_1darray(input_data, parse_data_name=False)

    if isinstance(target_data, list):
        if len(target_data) != len(input_array):
            raise ValueError(
                "The length of 'input' must match the length of the 'target' list."
            )
        target_arrays = [parse_1darray(t, parse_data_name=False) for t in target_data]
        return pd.concat(
            [
                pd.DataFrame(arr, columns=[name])
                for arr, name in zip(target_arrays, input_array)
            ],
            axis=1,
        )
    else:
        target_array = np.asarray(target_data)
        if target_array.ndim != 2:
            raise ValueError("If 'target' is not a list, it must be a 2D array.")
        if input_array.shape[0] != target_array.shape[1]:
            raise ValueError(
                "The length of 'input' must match the number of columns in 'target'."
            )
        return pd.DataFrame(target_array, columns=input_array)


@overload
def plot_box_chart(
    *,
    data: pd.DataFrame,
    show_anova: bool = False,
    show_dist: bool = False,
    title: str | None = None,
    input_name: str | None = None,
    target_name: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """绘制箱线图

    Args:
        data: 数据 shape: (n_samples, n_features). 将使用列名作为横轴刻度.
        show_anova: 是否显示ANOVA分析结果
        show_dist: 是否显示分布图
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
    show_anova: bool = False,
    show_dist: bool = False,
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
        show_anova: 是否显示ANOVA分析结果
        show_dist: 是否显示分布图
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表
    """
    ...


def plot_box_chart(
    *,
    show_dist: bool = False,
    show_anova: bool = False,
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
            "plot_box_chart requires either 'data' (DataFrame) or both 'input' and 'target' arguments."  # noqa: E501
        )

    return _plot_box_from_dataframe(
        data=data_for_plotting,
        show_anova=show_anova,
        show_dist=show_dist,
        title=title,
        input_name=input_name,
        target_name=target_name,
        figsize=figsize,
        show=show,
    )

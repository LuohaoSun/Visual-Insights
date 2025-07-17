import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from visual_insights.common.to_numpy import parse_1darray
from visual_insights.common.typing import ArrayLike


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


def _convert_long_to_wide_for_boxplot(
    data: pd.DataFrame, group_by: str | list[str], value_column: str
) -> pd.DataFrame:
    """
    通用工具：将长格式DataFrame转换为适合箱线图的宽格式DataFrame。

    Args:
        data: 长格式 DataFrame.
        group_col: 分组列的名称.
        value_col: 数值列的名称.

    Returns:
        宽格式 DataFrame，列名为分组，值为对应的数值序列.
    """
    grouped = data.groupby(group_by)[value_column]

    return pd.concat(
        [pd.Series(group.values, name=name) for name, group in grouped], axis=1
    )


def _plot_box_from_dataframe(
    data: pd.DataFrame,
    title: str | None,
    input_name: str | None,
    target_name: str | None,
    figsize: tuple[int, int],
    show: bool,
) -> Figure:
    """Plots a box chart from a pre-formatted DataFrame."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=data, ax=ax)

    if len(data.columns) > 10:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    if title:
        ax.set_title(title)
    if input_name:
        ax.set_xlabel(input_name)
    if target_name:
        ax.set_ylabel(target_name)

    plt.tight_layout()

    if show:
        plt.show()

    return fig

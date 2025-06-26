"""
通用可视化工具

提供创建图表、添加统计信息等通用可视化功能。
"""

from collections.abc import Mapping
from typing import Literal, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from seaborn.matrix import ClusterGrid

# 类型别名
LocationType = Literal["upper left", "upper right", "lower left", "lower right"]
StatValue = Union[float, str, int]


def create_figure(
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> tuple[Figure, Axes]:
    """
    创建具有通用设置的 matplotlib 图表

    参数:
        figsize: 图表大小，以 (宽度, 高度) 英寸为单位
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签

    返回:
        (figure, axes) 元组
    """
    fig, ax = plt.subplots(figsize=figsize or (8, 6))

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    return fig, ax


def add_stats_textbox(
    ax: Axes,
    stats: Mapping[str, StatValue],
    loc: LocationType = "upper left",
    fontsize: int = 10,
) -> None:
    """
    向 matplotlib 轴添加统计信息文本框

    参数:
        ax: Matplotlib 轴
        stats: 要显示的统计信息字典
        loc: 文本框位置 ('upper left', 'upper right', 'lower left', 'lower right')
        fontsize: 文本字体大小
    """
    # 将统计信息转换为文本
    stats_text = "\n".join(
        [
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in stats.items()
        ]
    )

    # 确定位置坐标
    if loc == "upper left":
        x, y = 0.05, 0.95
        va, ha = "top", "left"
    elif loc == "upper right":
        x, y = 0.95, 0.95
        va, ha = "top", "right"
    elif loc == "lower left":
        x, y = 0.05, 0.05
        va, ha = "bottom", "left"
    elif loc == "lower right":
        x, y = 0.95, 0.05
        va, ha = "bottom", "right"
    else:
        raise ValueError(f"Invalid location: {loc}")

    # 添加文本框
    ax.text(
        x,
        y,
        stats_text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    cmap: str = "seismic",
    annot: bool = False,
    cluster: bool = False,
    show: bool = True,
) -> Figure:
    """
    绘制相关矩阵热图

    参数:
        corr_matrix: 作为 pandas DataFrame 的相关矩阵
        figsize: 图表大小
        title: 图表标题
        cmap: 颜色映射名称
        annot: 是否用值标注单元格
        cluster: 是否聚类行和列
        show: 如果为 True，立即使用 plt.show() 显示图表

    返回:
        Matplotlib 图表
    """
    if cluster:
        # 将nan值替换为0
        corr_matrix = corr_matrix.fillna(0)
        g: ClusterGrid = sns.clustermap(
            corr_matrix,
            figsize=figsize or (10, 8),
            cmap=cmap,
            center=0,
            annot=annot,
            fmt=".2f",
            method="ward",
            vmin=-1,
            vmax=1,
        )
        if title:
            plt.suptitle(title, y=1.02)

        # 获取图表
        fig = g.figure

        # 如果需要，显示图表
        if show:
            plt.show()

        return fig
    else:
        plt.figure(figsize=figsize or (10, 8))
        fig = plt.gcf()
        sns.heatmap(
            corr_matrix,
            cmap=cmap,
            center=0,
            annot=annot,
            fmt=".2f",
            square=True,
            vmin=-1,
            vmax=1,
        )
        if title:
            plt.title(title)
        plt.tight_layout()

        # 如果需要，显示图表
        if show:
            plt.show()

        return fig


def set_equal_axes(ax: Axes) -> None:
    """
    设置轴的比例相等，并使 x 和 y 轴具有相同的范围

    参数:
        ax: Matplotlib 轴
    """
    # 获取当前轴的限制
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # 找到最小和最大值
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)

    # 设置相同的限制
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # 设置纵横比为 1
    ax.set_aspect("equal")

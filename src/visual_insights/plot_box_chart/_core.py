import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:
    from scipy.stats import f_oneway
except ImportError:
    f_oneway = None


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


def _annotate_ax_with_anova(ax: Axes, data: pd.DataFrame) -> None:
    """
    在给定的 Axes 上计算并标注 ANOVA 结果。
    此函数会直接修改传入的 ax 对象。
    """
    if f_oneway is None:
        warnings.warn(
            "To show ANOVA results, 'scipy' must be installed. Skipping ANOVA."
        )
        return

    if len(data.columns) < 2:
        warnings.warn("ANOVA requires at least two groups. Skipping analysis.")
        return

    # 准备 ANOVA 输入数据：移除每个组中的 NaN 值
    groups = [data[col].dropna() for col in data.columns]
    valid_groups = [g for g in groups if len(g) >= 2]

    if len(valid_groups) < 2:
        warnings.warn(
            "At least two groups need 2+ data points for ANOVA. Skipping analysis."
        )
        return

    try:
        f_stat, p_value = f_oneway(*valid_groups)

        p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        annotation = f"ANOVA: F={f_stat:.2f}, {p_text}"

        ax.text(
            0.98,
            0.98,
            annotation,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5),
        )
    except Exception as e:
        warnings.warn(f"An error occurred during ANOVA calculation: {e}")


def _plot_box_from_dataframe(
    data: pd.DataFrame,
    title: str | None,
    input_name: str | None,
    target_name: str | None,
    figsize: tuple[int, int],
    show_dist: bool,
    show_anova: bool,
    show: bool,
) -> Figure:
    """
    从预格式化的 DataFrame 编排箱线图的绘制过程。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if show_dist:
        sns.violinplot(
            data=data,
            ax=ax,
            color="lightblue",
            inner=None,
            linewidth=0,
        )
        sns.boxplot(
            data=data,
            ax=ax,
            width=0.3,
            boxprops={"facecolor": "white", "zorder": 10},
            whiskerprops={"zorder": 10},
            capprops={"zorder": 10},
            medianprops={"zorder": 10},
        )
    else:
        sns.boxplot(data=data, ax=ax)

    if show_anova:
        _annotate_ax_with_anova(ax, data)

    if len(data.columns) > 10:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    if title:
        ax.set_title(title)
    if input_name:
        ax.set_xlabel(input_name)
    if target_name:
        ax.set_ylabel(target_name)

    # Finalize - 逻辑不变
    plt.tight_layout()

    if show:
        plt.show()

    return fig

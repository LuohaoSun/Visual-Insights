import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .typing import ArrayLike
from .utils.data_handlers import ensure_numpy_array
from .utils.visualization import create_figure


def plot_bar_chart(
    input: list[str],
    target: ArrayLike,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    figsize: tuple[int, int] | None = None,
    palette: str | list[str] | None = None,
    show: bool = True,
    **kwargs,
) -> Figure:
    """
    根据给定的标签和数值绘制条形图。

    此函数遵循本库的通用绘图接口约定，使用 input 和 target 作为输入。

    参数:
    ----------
    input : List[str]
        条形图的分类标签 (x 轴)。
    target : ArrayLike
        每个分类对应的数值 (y 轴)。其长度必须与 input 相同。
        可以是 list, numpy array, 或 pandas Series。
    title : str, optional
        图表的标题。
    x_label : str, optional
        x 轴的标签。如果未提供，将使用 "Category"。
    y_label : str, optional
        y 轴的标签。如果未提供，将使用 "Value"。
    figsize : tuple, optional
        图表大小 (宽度, 高度)。
    palette : str or list of str, optional
        用于设置图表颜色的调色板。
    show : bool, optional
        如果为 True (默认)，则显示图表。
    **kwargs : dict
        其他要传递给 seaborn.barplot 的关键字参数。

    返回:
    -------
    matplotlib.figure.Figure
        包含绘图的 Figure 对象。
    """
    target_array = ensure_numpy_array(target).flatten()

    if len(input) != len(target_array):
        raise ValueError("参数 'input' (标签) 和 'target' (数值) 的长度必须相同。")

    plot_df = pd.DataFrame({"input": input, "target": target_array})

    fig, ax = create_figure(figsize=figsize)

    sns.barplot(data=plot_df, x="input", y="target", palette=palette, ax=ax, **kwargs)

    ax.set_title(title if title else "Bar Chart")
    ax.set_xlabel(x_label if x_label else "Category")
    ax.set_ylabel(y_label if y_label else "Value")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

from typing import List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats
import seaborn as sns

from .utils.data_handlers import ensure_numpy_array, ArrayLike
from .utils.visualization import create_figure

# 类型别名
TestType = Literal[
    "pearson", "spearman", "kendall", "chi2", "t-test", "anova", "f-test"
]


def plot_binned_box_plot(
    input: ArrayLike,
    target: ArrayLike,
    n_bins: int = 10,
    bin_method: Literal["cut", "qcut"] = "cut",
    feature_name: Optional[str] = None,
    target_name: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    show: bool = True,
) -> Figure:
    """
    创建一个按input值分箱的target值箱线图，并添加ANOVA检验结果。

    参数：
        input: 输入数据（将被分箱）。
        target: 目标数据。
        n_bins: input的分箱数量。
        bin_method: 分箱方法（'cut'或'qcut', 'cut' for equal width, 'qcut' for equal frequency）。
        feature_name: 输入特征名称，用于标注。
        target_name: 目标变量名称，用于标注。
        figsize: 图表大小。
        show: 如果为True，显示图表。

    返回：
        Matplotlib图表对象。
    """
    # 导入ANOVA检验需要的库
    from scipy import stats

    # 设置默认的ANOVA文本属性
    anova_text_props = {
        "size": 10,
        "ha": "left",
        "va": "top",
        "bbox": {"facecolor": "white", "alpha": 0.5},
    }
    anova_text_position = (0.02, 0.98)
    showfliers = False
    include_anova = True

    # Ensure input and target are pandas Series and names are strings
    _feature_name = (
        feature_name
        if feature_name
        else (
            input.name
            if isinstance(input, pd.Series) and input.name is not None
            else "Feature"
        )
    )
    _target_name = (
        target_name
        if target_name
        else (
            target.name
            if isinstance(target, pd.Series) and target.name is not None
            else "Target"
        )
    )

    input_name = str(_feature_name)
    target_name = str(_target_name)

    if not isinstance(input, pd.Series):
        input = pd.Series(np.array(input).flatten(), name=input_name)
    if not isinstance(target, pd.Series):
        target = pd.Series(np.array(target).flatten(), name=target_name)

    # Remove NaNs based on both input and target
    mask = ~(input.isna() | target.isna())
    input_clean = input[mask]
    target_clean = target[mask]

    if len(input_clean) == 0:
        print(
            "Warning: No valid data points after removing NaNs. Cannot generate plot."
        )
        # Return an empty figure or handle as appropriate
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No data to plot",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.xlabel(input_name)
        plt.ylabel(target_name)
        plt.title(f"Box Plot of {target_name} Binned by {input_name} (No Data)")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)  # Close if not showing
        return fig

    # Create bins for input
    try:
        if bin_method == "cut":
            # include_lowest=True ensures the minimum value is included
            input_binned = pd.cut(
                input_clean, bins=n_bins, include_lowest=True, right=True
            )
        elif bin_method == "qcut":
            # Use rank(method='first') to handle duplicate edges better
            # duplicates='drop' removes bins with non-unique edges
            input_binned = pd.qcut(
                input_clean.rank(method="first"), q=n_bins, duplicates="drop"
            )
        else:
            raise ValueError("bin_method must be 'cut' or 'qcut'")
    except ValueError as e:
        print(
            f"Warning: Binning failed for {input_name} with method '{bin_method}' and {n_bins} bins. Error: {e}"
        )
        # Fallback or return empty plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            f"Binning failed for {input_name}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.xlabel(input_name)
        plt.ylabel(target_name)
        plt.title(f"Box Plot of {target_name} Binned by {input_name} (Binning Failed)")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)  # Close if not showing
        return fig

    # Combine binned input and target into a DataFrame
    # Convert bins to string for plotting as categorical data but preserve order
    plot_df = pd.DataFrame(
        {
            input_name: input_binned.astype(str),
            target_name: target_clean,
            # Add bin midpoints or left edges for proper ordering
            "_bin_order": input_binned.apply(
                lambda bin_val: bin_val.left
                if hasattr(bin_val, "left")
                else bin_val.mid
            ),
        }
    )

    # Get the correct order of bins based on their numerical values
    bin_order = (
        plot_df.groupby(input_name)["_bin_order"].first().sort_values().index.tolist()
    )

    # Create the plot with explicit ordering
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        x=input_name,
        y=target_name,
        data=plot_df,
        order=bin_order,
        showfliers=showfliers,
        ax=ax,
    )

    # 执行ANOVA检验
    if include_anova and len(bin_order) >= 2:
        # 准备ANOVA的数据：每个分箱中的target值
        anova_data = [
            plot_df[plot_df[input_name] == bin_name][target_name].values
            for bin_name in bin_order
        ]
        # 过滤掉空的组
        anova_data = [group for group in anova_data if len(group) > 0]

        if len(anova_data) >= 2:  # 至少需要两个非空组才能进行ANOVA
            # 执行单因素ANOVA
            f_val, p_val = stats.f_oneway(*anova_data)

            # 准备ANOVA结果文本
            anova_text = f"ANOVA: F={f_val:.2f}, p={p_val:.4f}"
            if p_val < 0.05:
                anova_text += " *"
            if p_val < 0.01:
                anova_text += "*"
            if p_val < 0.001:
                anova_text += "*"

            # 在图上添加ANOVA结果
            ax.text(
                anova_text_position[0],
                anova_text_position[1],
                anova_text,
                transform=ax.transAxes,
                **anova_text_props,
            )

    # Improve x-axis labels by rotating them
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Set labels and title
    ax.set_xlabel(input_name)
    ax.set_ylabel(target_name)
    ax.set_title(f"Box Plot of {target_name} Binned by {input_name}")
    plt.tight_layout()  # Adjust layout to prevent labels overlapping

    if show:
        plt.show()
    else:
        plt.close(fig)  # Close the figure if not showing to save memory

    return fig


def plot_box_plot(
    data: ArrayLike,
    group_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    show: bool = True,
) -> Figure:
    """创建箱线图，支持单变量或多变量分组，并自动添加ANOVA统计信息。

    Args:
        data: 要绘制的数据。如果是2维数组，则第二维将作为分组。
              例如：np.array([[1,2,3], [4,5,6]]) 将创建两个组的箱线图。
              如果是1维数组，则创建单个箱线图。
              如果是DataFrame，则每列作为一个组。
              如果是列表，则每个元素作为一个组。
        group_names: 分组名称列表，用于标记每个组。如果不提供且data是DataFrame，
                    将使用列名作为组名。
        title: 图表标题。
        figsize: 图表尺寸元组 (宽度, 高度)，单位为英寸。
        show: 是否立即显示图表。

    Returns:
        matplotlib图形对象。
    """
    # 处理输入数据并提取分组信息
    melted_data = []  # 展平后的数据
    groups = []  # 对应的分组索引

    # 处理不同类型的输入数据
    if isinstance(data, pd.DataFrame):
        # DataFrame: 每列是一个组
        if group_names is None:
            group_names = data.columns.tolist()

        for i, col in enumerate(data.columns):
            col_data = data[col].dropna().values
            melted_data.extend(col_data)
            groups.extend([i] * len(col_data))

    elif isinstance(data, list):
        # 列表: 每个元素是一个组
        if group_names is None:
            group_names = [f"组{i + 1}" for i in range(len(data))]

        for i, group_data in enumerate(data):
            group_arr = ensure_numpy_array(group_data).flatten()
            melted_data.extend(group_arr)
            groups.extend([i] * len(group_arr))

    else:
        # numpy数组
        data_arr = ensure_numpy_array(data)

        if data_arr.ndim == 1:
            # 单组数据
            melted_data = data_arr.tolist()
            groups = [0] * len(data_arr)
            if group_names is None:
                group_names = ["数据"]

        elif data_arr.ndim == 2:
            # 多组数据: axis=1作为分组维度
            for i in range(data_arr.shape[1]):
                melted_data.extend(data_arr[:, i])
                groups.extend([i] * data_arr.shape[0])

            if group_names is None:
                group_names = [f"组{i + 1}" for i in range(data_arr.shape[1])]

        else:
            raise ValueError(f"数据维度必须是1或2，但得到了{data_arr.ndim}")

    # 创建DataFrame用于seaborn绘图
    df = pd.DataFrame(
        {"value": melted_data, "group": [group_names[int(g)] for g in groups]}
    )

    # 创建图形
    fig, ax = create_figure(figsize=figsize)

    # 绘制箱线图
    sns.boxplot(x="group", y="value", data=df, ax=ax)

    # 添加ANOVA统计信息（如果有多个组）
    if len(group_names) >= 2:
        # 准备ANOVA的数据：每个组中的数据
        anova_groups = [df[df["group"] == name]["value"].values for name in group_names]
        anova_groups = [group for group in anova_groups if len(group) > 0]

        if len(anova_groups) >= 2:  # 至少需要两个非空组才能进行ANOVA
            # 执行单因素ANOVA
            f_val, p_val = stats.f_oneway(*anova_groups)

            # 准备ANOVA结果文本
            anova_text = f"ANOVA: F={f_val:.2f}, p={p_val:.4f}"
            if p_val < 0.05:
                anova_text += " *"
            if p_val < 0.01:
                anova_text += "*"
            if p_val < 0.001:
                anova_text += "*"

            # 在图上添加ANOVA结果
            ax.text(
                0.02,
                0.98,
                anova_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="aliceblue", alpha=0.5),
            )

    # 设置标题
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("box plot", fontsize=14)

    # 旋转x轴标签以避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # 添加网格线
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    # 调整布局
    try:
        plt.tight_layout()
    except RuntimeError:
        pass

    if show:
        plt.show()
    else:
        plt.close(fig)  # 如果不显示则关闭图形以节省内存

    return fig

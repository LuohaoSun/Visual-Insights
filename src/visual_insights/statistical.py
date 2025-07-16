"""
统计分析可视化模块

提供用于可视化统计检验结果的函数。
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

from .utils.data_handlers import ArrayLike, ensure_numpy_array, handle_feature_names
from .utils.visualization import create_figure, plot_correlation_heatmap

# 类型别名
TestType = Literal[
    "pearson", "spearman", "kendall", "chi2", "t-test", "anova", "f-test"
]


def _calculate_test_statistic(
    x: np.ndarray, y: np.ndarray, test_type: TestType
) -> tuple[float, float]:
    """
    计算统计检验结果

    参数：
        x: 输入特征
        y: 目标变量
        test_type: 检验类型

    返回：
        (统计量, p值) 元组
    """
    if test_type == "pearson":
        return stats.pearsonr(x, y)
    elif test_type == "spearman":
        return stats.spearmanr(x, y)
    elif test_type == "kendall":
        return stats.kendalltau(x, y)
    elif test_type == "chi2":
        # 对于卡方检验，需要将连续变量离散化
        x_bins = pd.qcut(x, 4, labels=False, duplicates="drop")
        y_bins = (
            pd.qcut(y, 4, labels=False, duplicates="drop")
            if not np.issubdtype(y.dtype, np.integer)
            else y
        )
        contingency = pd.crosstab(x_bins, y_bins)
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        return chi2, p  # type: ignore
    elif test_type == "t-test":
        # 对于 t 检验，将 y 视为二分类变量
        if np.unique(y).size != 2:
            raise ValueError("t-test requires binary target variable")
        group1 = x[y == np.unique(y)[0]]
        group2 = x[y == np.unique(y)[1]]
        t, p = stats.ttest_ind(group1, group2, equal_var=False)
        return t, p  # type: ignore
    elif test_type in ["anova", "f-test"]:
        # 对于 ANOVA，将 y 视为分类变量
        groups = [x[y == val] for val in np.unique(y)]
        f, p = stats.f_oneway(*groups)
        return f, p
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def plot_statistical_tests(
    input: ArrayLike,
    target: ArrayLike,
    test_type: TestType = "pearson",
    feature_names: list[str] | None = None,
    target_name: str = "Target",
    n_features: int = 20,
    sort: bool = True,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    绘制统计检验结果条形图

    参数：
        input: 输入数据
        target: 目标变量
        test_type: 检验类型 ('pearson', 'spearman', 'kendall', 'chi2', 't-test', 'anova')
        feature_names: 特征名称
        target_name: 目标变量名称
        n_features: 显示的特征数量
        sort: 是否按检验统计量排序
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表

    返回：
        matplotlib Figure 对象
    """
    # 转换为 numpy 数组
    X = ensure_numpy_array(input)
    y = ensure_numpy_array(target)

    # 确保 y 是一维的
    if y.ndim > 1:
        y = y.ravel()

    # 获取特征名称
    names = handle_feature_names(input, feature_names)

    # 计算每个特征的统计量
    stats_values = []
    p_values = []

    for i in range(X.shape[1]):
        try:
            stat, p = _calculate_test_statistic(X[:, i], y, test_type)
            stats_values.append(stat)
            p_values.append(p)
        except Exception as e:
            print(f"Error calculating {test_type} for feature {names[i]}: {e}")
            stats_values.append(np.nan)
            p_values.append(np.nan)

    # 创建 DataFrame 以便排序
    stats_df = pd.DataFrame(
        {"Feature": names, "Statistic": stats_values, "p-value": p_values}
    )

    # 移除 NaN 值
    stats_df = stats_df.dropna()

    # 排序（如果需要）
    if sort:
        stats_df = stats_df.sort_values("Statistic", ascending=False)

    # 限制特征数量
    if n_features < len(stats_df):
        stats_df = stats_df.head(n_features)

    # 创建图表
    fig, ax = create_figure(
        figsize=figsize,
        title=title or f"{test_type.capitalize()} Correlation with {target_name}",
        xlabel=f"{test_type.capitalize()} Statistic",
        ylabel="Feature",
    )

    # 绘制水平条形图
    bars = ax.barh(stats_df["Feature"], stats_df["Statistic"])

    # 为显著的特征添加星号
    for i, (_, row) in enumerate(stats_df.iterrows()):
        if row["p-value"] < 0.001:
            ax.text(row["Statistic"], i, "***", ha="left", va="center")
        elif row["p-value"] < 0.01:
            ax.text(row["Statistic"], i, "**", ha="left", va="center")
        elif row["p-value"] < 0.05:
            ax.text(row["Statistic"], i, "*", ha="left", va="center")

    # 添加显著性图例
    ax.text(
        0.95,
        0.05,
        "* p<0.05, ** p<0.01, *** p<0.001",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 反转 y 轴，使最重要的特征在顶部
    ax.invert_yaxis()

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_statistical_tests_heatmap(
    input: ArrayLike,
    target: ArrayLike,
    test_type: TestType = "pearson",
    feature_names: list[str] | None = None,
    target_names: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    annot: bool = False,
    cmap: str = "seismic",
    title: str | None = None,
    show: bool = True,
) -> Figure:
    """
    绘制统计检验结果热图

    参数：
        input: 输入数据
        target: 目标变量
        test_type: 检验类型 ('pearson', 'spearman', 'kendall', 'chi2', 't-test', 'anova')
        feature_names: 特征名称
        target_names: 目标变量名称
        figsize: 图表大小
        annot: 是否在单元格中标注值
        cmap: 颜色映射名称
        title: 图表标题
        show: 是否立即显示图表

    返回：
        matplotlib Figure 对象
    """
    # 转换为 numpy 数组
    X = ensure_numpy_array(input)
    y = ensure_numpy_array(target)

    # 获取特征名称
    x_names = handle_feature_names(input, feature_names)

    # 处理目标变量
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        y_names = [target_names[0] if target_names else "Target"]
    else:
        y_names = handle_feature_names(target, target_names, prefix="Target")

    # 计算统计量矩阵
    stats_matrix = np.zeros((X.shape[1], y.shape[1]))

    for i in range(X.shape[1]):
        for j in range(y.shape[1]):
            try:
                stat, _ = _calculate_test_statistic(X[:, i], y[:, j], test_type)
                stats_matrix[i, j] = stat
            except Exception as e:
                print(
                    f"Error calculating {test_type} for {x_names[i]} vs {y_names[j]}: {e}"
                )
                stats_matrix[i, j] = np.nan

    # 创建 DataFrame
    stats_df = pd.DataFrame(stats_matrix, index=x_names, columns=y_names)

    # 设置标题
    if title is None:
        title = f"{test_type.capitalize()} Statistics Heatmap"

    # 绘制热图
    fig = plt.figure(figsize=figsize)
    sns.heatmap(stats_df, cmap=cmap, annot=annot, fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_correlation_matrix(
    input: ArrayLike,
    target: ArrayLike | None = None,
    input_feature_names: list[str] | None = None,
    target_feature_names: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    figsize: tuple[int, int] | None = None,
    cluster: bool = False,
    cmap: str = "seismic",
    annot: bool = False,
    abs_values: bool = False,
    title: str | None = None,
    show: bool = True,
) -> Figure:
    """
    绘制相关矩阵热图

    参数：
        input: 输入数据
        target: 目标数据（可选），如果提供则绘制交叉相关
        input_feature_names: 输入特征名称
        target_feature_names: 目标特征名称
        method: 相关方法 ('pearson', 'spearman', 'kendall')
        figsize: 图表大小
        cluster: 是否聚类特征
        cmap: 颜色映射名称
        annot: 是否在单元格中标注值
        abs_values: 是否取相关性的绝对值
        title: 图表标题
        show: 是否立即显示图表

    返回：
        matplotlib Figure 对象
    """
    # 转换为 pandas DataFrame
    if isinstance(input, np.ndarray):
        x_feature_names = input_feature_names or [
            f"Feature {i}" for i in range(input.shape[1])
        ]
        x_df = pd.DataFrame(input, columns=x_feature_names)
    else:
        x_df = pd.DataFrame(input)
        if input_feature_names is not None:
            x_df.columns = input_feature_names

    # 如果没有提供目标数据，绘制输入数据的自相关矩阵
    if target is None:
        corr_matrix = x_df.corr(method=method)

        if abs_values:
            corr_matrix = corr_matrix.abs()

        if title is None:
            title = "Feature Correlation Matrix"
            if cluster:
                title += " (Clustered)"

        if figsize is None:
            figsize = (max(8, x_df.shape[1] // 2), max(6, x_df.shape[1] // 2))

        fig = plot_correlation_heatmap(
            corr_matrix,
            figsize=figsize,
            title=title,
            cmap=cmap,
            annot=annot,
            cluster=cluster,
            show=show,
        )

        return fig

    # 如果提供了目标数据，绘制交叉相关矩阵
    if isinstance(target, np.ndarray):
        y_feature_names = target_feature_names or [
            f"Target {i}" for i in range(target.shape[1])
        ]
        y_df = pd.DataFrame(target, columns=y_feature_names)
    else:
        y_df = pd.DataFrame(target)
        if target_feature_names is not None:
            y_df.columns = target_feature_names

    # 合并输入和目标数据
    merged_df = pd.concat([x_df, y_df], axis=1)

    # 计算相关矩阵
    full_corr = merged_df.corr(method=method)

    # 提取交叉相关部分（X vs Y）
    cross_corr = full_corr.iloc[: x_df.shape[1], x_df.shape[1] :]

    if abs_values:
        cross_corr = cross_corr.abs()

    if title is None:
        title = "Feature-Target Cross-Correlation Matrix"
        if cluster:
            title += " (Clustered)"

    if figsize is None:
        figsize = (max(8, y_df.shape[1] // 2), max(6, x_df.shape[1] // 2))

    fig = plot_correlation_heatmap(
        cross_corr,
        figsize=figsize,
        title=title,
        cmap=cmap,
        annot=annot,
        cluster=cluster,
        show=show,
    )

    return fig


def plot_scatter(
    input: ArrayLike,
    target: ArrayLike,
    feature_name: str | None = None,
    target_name: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_mean_lines: bool = True,
    show_linear_fit: bool = True,
    show_spearman: bool = True,
    show: bool = True,
) -> Figure:
    """Generates a scatter plot of an input feature against a target variable.

    Optionally plots the mean of the input feature (x-mean) as a vertical line,
    the mean of the target variable (y-mean) as a horizontal line,
    and a linear regression fit line with statistics.

    Args:
        input: Array-like data for the input feature (x-axis).
        target: Array-like data for the target variable (y-axis).
        feature_name: Optional name of the input feature for labeling.
        target_name: Optional name of the target variable for labeling.
        title: Optional custom title for the plot. If None, a default title is generated.
        figsize: Optional tuple specifying the figure size (width, height) in inches.
        xlim: Optional tuple specifying the x-axis limits (min, max).
        ylim: Optional tuple specifying the y-axis limits (min, max).
        show_mean_lines: If True, displays mean lines for x and y axes.
        show_linear_fit: If True, displays linear regression fit line and R²/p-value.
        show_spearman: If True, displays Spearman correlation and p-value.
        show: If True, displays the plot.

    Returns:
        The matplotlib Figure object.

    Raises:
        ValueError: If input and target have different number of samples,
                    or if input or target are not 1D or squeezable to 1D.
    """
    input_arr = ensure_numpy_array(input).squeeze()
    target_arr = ensure_numpy_array(target).squeeze()

    if input_arr.ndim != 1:
        original_shape = ensure_numpy_array(input).shape
        raise ValueError(
            f"Input 'input' (original shape: {original_shape}, after squeeze: {input_arr.shape}) "
            "must be 1D or squeezable to 1D for scatter plot."
        )
    if target_arr.ndim != 1:
        original_shape = ensure_numpy_array(target).shape
        raise ValueError(
            f"Input 'target' (original shape: {original_shape}, after squeeze: {target_arr.shape}) "
            "must be 1D or squeezable to 1D for scatter plot."
        )

    if input_arr.shape[0] != target_arr.shape[0]:
        raise ValueError(
            f"Input feature (length {input_arr.shape[0]}) and target variable (length {target_arr.shape[0]}) "
            "must have the same number of samples."
        )

    fig, ax = create_figure(figsize=figsize)

    sns.scatterplot(
        x=input_arr,
        y=target_arr,
        ax=ax,
        s=50,  # marker size
        alpha=0.7,  # marker transparency
        edgecolor="k",  # marker edge color
        linewidth=0.5,  # marker edge width
        label="Data points",  # Added label for legend
    )

    # Plot mean lines if requested
    if show_mean_lines:
        x_mean = np.mean(input_arr)
        y_mean = np.mean(target_arr)
        ax.axvline(
            x_mean,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"X Mean: {x_mean:.2f}",
        )
        ax.axhline(
            y_mean,
            color="green",
            linestyle="--",
            linewidth=1,
            label=f"Y Mean: {y_mean:.2f}",
        )

    # Perform linear regression and plot the fit line if requested
    if show_linear_fit:
        slope, intercept, r_value, p_value_lin, std_err = stats.linregress(
            input_arr, target_arr
        )
        line_x = np.array([np.min(input_arr), np.max(input_arr)])
        line_y = slope * line_x + intercept
        ax.plot(
            line_x,
            line_y,
            color="purple",
            linestyle="-",
            linewidth=2,
            label=f"Linear Fit: y={slope:.2f}x+{intercept:.2f}\nR²={r_value**2:.2f}, p-lin={p_value_lin:.3f}",
        )

    # Perform Spearman's rank correlation for a more general independence test if requested
    if show_spearman:
        spearman_corr, spearman_p_value = stats.spearmanr(input_arr, target_arr)
        # Add Spearman correlation as an empty line in the plot for legend purposes
        ax.plot(
            [],
            [],
            " ",
            label=f"Spearman's ρ: {spearman_corr:.2f}, p-value: {spearman_p_value:.3f}",
        )

    _feature_name = feature_name if feature_name else "Feature"
    _target_name = target_name if target_name else "Target"

    ax.set_xlabel(_feature_name, fontsize=12)
    ax.set_ylabel(_target_name, fontsize=12)

    # 设置标题，如果提供了自定义标题则使用，否则使用默认标题
    plot_title = (
        title
        if title is not None
        else f"Scatter Plot: {_feature_name} vs {_target_name}"
    )
    ax.set_title(plot_title, fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.7)

    # Set axis limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Add legend with a nice box
    legend = ax.legend(frameon=True, loc="best", fontsize=9)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_edgecolor("lightgray")

    try:
        plt.tight_layout()
    except RuntimeError:
        # tight_layout can sometimes fail in non-GUI backends or specific figure states
        pass

    if show:
        plt.show()

    return fig


def plot_distribution(
    data: ArrayLike,
    column_name: str | None = None,
    bins: int = 30,
    figsize: tuple[int, int] | None = None,
    xlim: tuple[float, float] | None = None,
    title: str | None = None,
    show_cdf: bool = False,
    show: bool = True,
) -> Figure:
    """
    绘制单列数据的分布图

    参数：
        data: 输入数据（可以是数组、Series或DataFrame的单列）
        column_name: 列名称（如果为None，将使用Series名称或默认名称）
        bins: 直方图的箱数
        figsize: 图表大小，默认为(8, 6)
        xlim: 横轴范围，格式为(min, max)，默认为None（自动确定范围）
        title: 图表标题
        show_cdf: 是否显示累计分布函数曲线
        show: 是否立即显示图表

    返回：
        matplotlib Figure 对象
    """
    # 处理输入数据，确保是一维数据
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame必须只包含一列数据")
        series = data.iloc[:, 0]
        if column_name is None:
            column_name = data.columns[0]
    elif isinstance(data, pd.Series):
        series = data
        if column_name is None:
            column_name = data.name or "data"
    else:  # np.ndarray 或其他类型
        data_array = np.asarray(data)
        if data_array.ndim > 1:
            if data_array.shape[1] == 1:
                data_array = data_array.flatten()
            else:
                raise ValueError("输入数组必须是一维的或只有一列的二维数组")
        series = pd.Series(data_array)
        if column_name is None:
            column_name = "data"

    # 设置图表大小
    if figsize is None:
        figsize = (8, 6)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制直方图和核密度估计
    sns.histplot(x=series, bins=bins, kde=True, ax=ax)

    # 获取KDE曲线对象用于图例
    kde_line = None
    for line in ax.get_lines():
        kde_line = line  # 最后一条线应该是KDE曲线

    # 添加KDE图例标签
    if kde_line:
        kde_line.set_label("KDE")

    # 计算统计信息
    mean = series.mean()
    std = series.std()
    median = series.median()
    min_val = series.min()
    max_val = series.max()

    stats_text = f"Mean: {mean:.2f}\nStd: {std:.2f}\nMedian: {median:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}"

    # 准备图例元素
    handles = []
    labels = []

    # 添加直方图的图例
    patch = ax.patches[0] if ax.patches else None
    if patch:
        handles.append(patch)
        labels.append("Histogram")

    # 添加KDE的图例
    if kde_line:
        handles.append(kde_line)
        labels.append("KDE")

    # 如果需要，添加累计分布函数曲线
    if show_cdf:
        # 创建第二个Y轴
        ax2 = ax.twinx()

        # 计算累计分布函数
        counts, bin_edges = np.histogram(series, bins=bins)
        cdf = np.cumsum(counts) / len(series)

        # 绘制CDF曲线
        cdf_line = ax2.plot(bin_edges[1:], cdf, "r-", linewidth=2, label="CDF")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_ylim(0, 1.05)
        ax2.grid(False)

        # 添加CDF的图例
        handles.append(cdf_line[0])
        labels.append("CDF")

    # 添加统计信息到图例中
    # 使用透明矩形作为占位符
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="white", alpha=0))
    labels.append(stats_text)

    # 创建统一的图例，放在图表外部右侧
    fig.legend(
        handles,
        labels,
        loc="center left",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9,
    )

    # 设置标题和标签
    title = title or f"{column_name} distribution"
    ax.set_title(title)
    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")

    # 设置横轴范围（如果指定）
    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_mutual_information(
    input: ArrayLike,
    target: ArrayLike,
    feature_names: list[str] | None = None,
    target_name: str = "Target",
    n_features: int = 20,
    sort: bool = True,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    绘制互信息条形图

    参数：
        input: 输入数据
        target: 目标变量
        feature_names: 特征名称
        target_name: 目标变量名称
        n_features: 显示的特征数量
        sort: 是否按互信息值排序
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表

    返回：
        matplotlib Figure 对象
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    # 转换为 numpy 数组
    X = ensure_numpy_array(input)
    y = ensure_numpy_array(target)

    # 确保 y 是一维的
    if y.ndim > 1:
        y = y.ravel()

    # 获取特征名称
    names = handle_feature_names(input, feature_names)

    # 检测目标变量类型（分类或回归）
    is_classification = False
    if np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_):
        is_classification = True
        unique_values = np.unique(y)
        if len(unique_values) < 2:
            # 如果只有一个类别，退回到回归
            is_classification = False
        elif len(unique_values) > 10:
            # 如果类别太多，可能不是分类问题
            is_classification = False

    # 计算互信息
    if is_classification:
        mi_values = mutual_info_classif(X, y)
    else:
        mi_values = mutual_info_regression(X, y)

    # 创建 DataFrame 以便排序
    mi_df = pd.DataFrame({"Feature": names, "Mutual Information": mi_values})

    # 排序（如果需要）
    if sort:
        mi_df = mi_df.sort_values("Mutual Information", ascending=False)

    # 限制特征数量
    if n_features < len(mi_df):
        mi_df = mi_df.head(n_features)

    # 创建图表
    fig, ax = create_figure(
        figsize=figsize,
        title=title or f"Mutual Information with {target_name}",
        xlabel="Mutual Information",
        ylabel="Feature",
    )

    # 绘制水平条形图
    ax.barh(mi_df["Feature"], mi_df["Mutual Information"])

    # 反转 y 轴，使最重要的特征在顶部
    ax.invert_yaxis()

    plt.tight_layout()

    if show:
        plt.show()

    return fig

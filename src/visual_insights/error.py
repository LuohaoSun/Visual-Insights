"""
误差可视化模块

提供用于可视化模型误差和预测的函数。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from visual_insights.statistical import plot_distribution
from visual_insights.typing import ArrayLike
from visual_insights.utils.data_handlers import ensure_numpy_array
from visual_insights.utils.visualization import (
    add_stats_textbox,
    create_figure,
    set_equal_axes,
)


def plot_error_distribution(
    input: ArrayLike,
    target: ArrayLike,
    bins: int = 30,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show_cdf: bool = False,
    show: bool = True,
) -> Figure:
    """
    绘制误差分布图

    参数:
        input: 预测值 of shape (n_samples, n_dimensions) 或 (n_samples,)
        target: 真实值 of shape (n_samples, n_dimensions) 或 (n_samples,)
        bins: 直方图的箱数
        title: 图表标题
        figsize: 图表大小
        show_cdf: 是否显示累计分布函数曲线
        show: 是否立即显示图表

    返回:
        matplotlib Figure 对象
    """
    # 转换为 numpy 数组
    y_pred = ensure_numpy_array(input)
    y_true = ensure_numpy_array(target)

    # 计算误差
    errors = y_true - y_pred

    if errors.ndim > 1:
        errors = errors.flatten()  # 展平多维误差
    return plot_distribution(
        data=errors,
        column_name="Error",
        bins=bins,
        figsize=figsize,
        title=title or "Error Distribution",
        show_cdf=show_cdf,
        show=show,
    )


def plot_residuals(
    input: ArrayLike,
    target: ArrayLike,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    绘制残差图

    参数:
        input: 预测值
        target: 真实值
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表

    返回:
        matplotlib Figure 对象
    """
    # 转换为 numpy 数组
    y_pred = ensure_numpy_array(input)
    y_true = ensure_numpy_array(target)

    # 计算残差
    residuals = y_true - y_pred

    # 创建图表
    fig, ax = create_figure(
        figsize=figsize,
        title=title or "Residual Plot",
        xlabel="Predicted Value",
        ylabel="Residual",
    )

    # 绘制散点图
    ax.scatter(y_pred, residuals, alpha=0.5)

    # 添加零线
    ax.axhline(y=0, color="r", linestyle="-", alpha=0.3)

    # 添加统计信息
    stats = {
        "Mean": float(np.mean(residuals)),
        "Std": float(np.std(residuals)),
        "RMSE": float(np.sqrt(np.mean(residuals**2))),
    }
    add_stats_textbox(ax, stats)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_actual_vs_predicted(
    input: ArrayLike,
    target: ArrayLike,
    title: str | None = None,
    colorful: bool = False,
    figsize: tuple[int, int] = (10, 6),
    min_val: float | None = None,
    max_val: float | None = None,
    show: bool = True,
) -> Figure:
    """
    绘制真实值与预测值对比图

    参数:
        input: 预测值
        target: 真实值
        title: 图表标题
        colorful: 是否使用彩色（不同维度使用不同颜色）
        figsize: 图表大小
        show: 是否立即显示图表

    返回:
        matplotlib Figure 对象
    """
    # 转换为 numpy 数组
    y_pred = ensure_numpy_array(input)
    y_true = ensure_numpy_array(target)

    # 创建图表
    fig, ax = create_figure(
        figsize=figsize,
        title=title or "Actual vs Predicted",
        xlabel="Actual",
        ylabel="Predicted",
    )

    # 如果是多维输出且使用彩色
    if y_pred.ndim > 1 and colorful:
        n_dims = y_pred.shape[1]
        for i in range(n_dims):
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, label=f"Dimension {i}")
        ax.legend()
    else:
        # 如果是多维输出但不使用彩色，或者是一维输出
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
        ax.scatter(y_true, y_pred, alpha=0.5)

    # 添加对角线（完美预测线）
    # 确保 min_val 和 max_val 是 float 类型
    current_min_val = min_val if min_val is not None else float(np.min(y_true))
    current_max_val = max_val if max_val is not None else float(np.max(y_true))

    # 如果 min_val 或 max_val 未提供，则从数据中计算
    if min_val is None:
        current_min_val = min(current_min_val, float(np.min(y_pred)))
    if max_val is None:
        current_max_val = max(current_max_val, float(np.max(y_pred)))

    ax.plot(
        [current_min_val, current_max_val],
        [current_min_val, current_max_val],
        "r--",
        alpha=0.5,
    )

    # 设置相等的轴比例
    set_equal_axes(ax)

    # 添加统计信息
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    stats = {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "R²": float(r2)}
    add_stats_textbox(ax, stats)

    plt.tight_layout()

    if show:
        plt.show()

    return fig

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

__all__ = ["plot_offset_map"]


def plot_offset_map(
    ideal_centers: np.ndarray,
    offset_pred: np.ndarray,
    offset_true: np.ndarray | None = None,
    edge_indices: list | np.ndarray = [],
    site_box_size: int = 10,
    figsize: tuple = (20, 20),
    show: bool = True,
) -> Figure:
    """绘制偏移量 MAP 图。
    如果不提供 `offset_true`，信息框会显示每个site预测值的平均偏移量
    如果提供了 `offset_true`，信息框会显示每个site的误差, 以及每个象限的误差。

    Args:
        ideal_centers (np.ndarray): A 2D array of shape (num_centers, 2)
            which indicates the (x, y) coordinates of the ideal centers.
        offset_true (np.ndarray): A 3D array of shape (num_samples, num_centers, 2)
            which indicates the true offsets of the centers in the BMTP map.
        offset_pred (np.ndarray): A 3D array of shape (num_samples, num_centers, 2)
            which indicates the predicted offsets of the centers in the BMTP map.
        edge_indices (np.ndarray): A 2D array of shape (num_edges, 2) which indicates
            the indices of the centers that form lines in the BMTP map.
        figsize (tuple): The size of the figure to be plotted.
        show (bool): Whether to show the plot immediately.

    Returns:
        Figure: The matplotlib figure object containing the plot.
    """
    if offset_true is None:
        return _plot_offset_map_single(
            ideal_centers,
            offset_pred,
            edge_indices,
            site_box_size,
            figsize,
            show,
        )
    else:
        return _plot_offset_map_compared(
            ideal_centers,
            offset_pred,
            offset_true,
            edge_indices,
            site_box_size,
            figsize,
            show,
        )


def _plot_offset_map_single(
    ideal_centers: np.ndarray,
    offset: np.ndarray,
    edge_indices: list | np.ndarray,
    site_box_size: int,
    figsize: tuple = (20, 20),
    show: bool = True,
) -> Figure:
    """信息框会显示每个site预测值的平均偏移量"""

    ideal_centers = _ideal_centers_transform(ideal_centers, site_box_size)

    sample_points, sample_centers = _offset_transform(ideal_centers, offset)

    c_ideal = "#929292"
    c_pred = "#fc9235"

    fig, ax = plt.subplots(figsize=figsize, dpi=450)

    _plot_sample_points(ax, sample_points, c=c_pred, label="Predicted Samples")

    _plot_centers(ax, ideal_centers, c=c_ideal, label="Ideal Center")
    _plot_centers(ax, sample_centers, c=c_pred, label="Predicted Sample Center")

    _draw_lines_between_centers(ax, ideal_centers, edge_indices, c=c_ideal)
    _draw_lines_between_centers(ax, sample_centers, edge_indices, c=c_pred)

    _draw_site_info_boxes(ax, ideal_centers, sample_points, None, site_box_size)
    _draw_quadrant_info_boxes(ax, ideal_centers, sample_points, None, site_box_size)

    _set_plot_style(ax)

    if show:
        plt.show()

    return fig


def _plot_offset_map_compared(
    ideal_centers: np.ndarray,
    offset_pred: np.ndarray,
    offset_true: np.ndarray,
    edge_indices: list | np.ndarray,
    site_box_size: int,
    figsize: tuple = (20, 20),
    show: bool = True,
) -> Figure:
    """信息框会显示每个site的误差, 以及每个象限的误差"""
    ideal_centers = _ideal_centers_transform(ideal_centers, site_box_size)
    (
        sample_points_true,
        sample_centers_true,
    ) = _offset_transform(ideal_centers, offset_true)

    (
        sample_points_pred,
        sample_centers_pred,
    ) = _offset_transform(ideal_centers, offset_pred)

    c_ideal = "#929292"
    c_true = "#1083d6"
    c_pred = "#fc9235"

    fig, ax = plt.subplots(figsize=figsize, dpi=450)

    _plot_sample_points(ax, sample_points_true, c=c_true, label="True Samples")
    _plot_sample_points(ax, sample_points_pred, c=c_pred, label="Predicted Samples")

    _plot_centers(ax, ideal_centers, c=c_ideal, label="Ideal Center")
    _plot_centers(ax, sample_centers_true, c=c_true, label="True Sample Center")
    _plot_centers(ax, sample_centers_pred, c=c_pred, label="Predicted Sample Center")

    _draw_lines_between_centers(ax, ideal_centers, edge_indices, c=c_ideal)
    _draw_lines_between_centers(ax, sample_centers_true, edge_indices, c=c_true)
    _draw_lines_between_centers(ax, sample_centers_pred, edge_indices, c=c_pred)

    _draw_site_info_boxes(
        ax, ideal_centers, sample_points_pred, sample_points_true, site_box_size
    )
    _draw_quadrant_info_boxes(
        ax, ideal_centers, sample_points_pred, sample_points_true, site_box_size
    )

    _set_plot_style(ax)

    if show:
        plt.show()

    return fig


def _ideal_centers_transform(
    ideal_centers: np.ndarray, site_box_size: int
) -> np.ndarray:
    """
    对理想中心点进行线性变换以将过于靠近坐标轴的点分开。
    同时匹配offset的范围从而计算实际样本点、预测样本点以及它们中心的绝对坐标。
    """
    ideal_centers = ideal_centers / ideal_centers.max() * site_box_size * 2.5
    move_vector = np.zeros_like(ideal_centers)
    move_vector[:, 0] = np.where(
        ideal_centers[:, 0] < 0, -site_box_size / 1.5, site_box_size / 1.5
    )
    move_vector[:, 1] = np.where(
        ideal_centers[:, 1] < 0, -site_box_size / 1.5, site_box_size / 1.5
    )
    return ideal_centers + move_vector


def _offset_transform(
    ideal_centers: np.ndarray,
    offset: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将offset值转换为绝对坐标, 并给出中心.
    """
    sample_points = np.expand_dims(ideal_centers, axis=0) + offset
    sample_centers = sample_points.mean(axis=0)
    return (sample_points, sample_centers)


def _plot_sample_points(ax: Axes, sample_point_coords: np.ndarray, c: str, label: str):
    """
    绘制实际样本点和预测样本点。
    """
    ax.scatter(
        sample_point_coords.reshape(-1, 2)[:, 0],
        sample_point_coords.reshape(-1, 2)[:, 1],
        c=c,
        label=label,
        s=10,
        alpha=0.3,
    )


def _plot_centers(
    ax: Axes,
    center_coords: np.ndarray,
    c: str,
    label: str,
):
    """
    绘制理想中心点、实际样本中心点和预测样本中心点。
    """
    ax.scatter(
        center_coords[:, 0],
        center_coords[:, 1],
        c=c,
        label=label,
        s=80,
        marker="*",
        zorder=3,
    )


def _draw_site_info_boxes(
    ax: Axes,
    ideal_centers: np.ndarray,
    sample_points_pred: np.ndarray,
    sample_points_true: np.ndarray | None,
    site_box_size: int,
):
    """
    在每个理想中心点周围绘制方框并显示误差。
    """
    for i, center in enumerate(ideal_centers):
        # draw border box
        rect = Rectangle(
            (center[0] - site_box_size * 0.5, center[1] - site_box_size * 0.5),
            site_box_size,
            site_box_size,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            linestyle=":",
        )
        ax.add_patch(rect)

        # add info text
        if sample_points_true is None:
            # 计算平均offset
            mean_offset_x = sample_points_pred[:, i, 0].mean() - center[0]
            mean_offset_y = sample_points_pred[:, i, 1].mean() - center[1]
            info_text = f"Box Range: ±{site_box_size / 2}\nMean Offset:\n"
            info_text += f"(X: {mean_offset_x:.2f} Y: {mean_offset_y:.2f})"
        else:
            # 计算MAE
            mae_x = np.abs(
                sample_points_true[:, i, 0] - sample_points_pred[:, i, 0]
            ).mean()
            mae_y = np.abs(
                sample_points_true[:, i, 1] - sample_points_pred[:, i, 1]
            ).mean()
            info_text = f"MAE X: {mae_x:.2f}\nMAE Y: {mae_y:.2f}"
        ax.text(
            center[0] - site_box_size * 0.45,
            center[1] + site_box_size * 0.45,
            info_text,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(
                facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"
            ),
        )


def _draw_quadrant_info_boxes(
    ax: Axes,
    ideal_centers: np.ndarray,
    sample_points_pred: np.ndarray,
    sample_points_true: np.ndarray | None,
    site_box_size: int,
):
    """
    在每个象限绘制大方框并显示该象限的平均误差。
    """
    # 确定象限的边界，基于所有小方框的最远边缘，并增加一些边距
    padding = site_box_size * 0.2  # 增加一个小的边距
    x_min_overall = ideal_centers[:, 0].min() - site_box_size * 0.4 - padding
    x_max_overall = ideal_centers[:, 0].max() + site_box_size * 0.4 + padding
    y_min_overall = ideal_centers[:, 1].min() - site_box_size * 0.4 - padding
    y_max_overall = ideal_centers[:, 1].max() + site_box_size * 0.4 + padding

    text_offset = site_box_size * 0.05  # 文本与方框的偏移量

    quadrants = {
        "Quadrant I": {"x_range": (0, x_max_overall), "y_range": (0, y_max_overall)},
        "Quadrant II": {"x_range": (x_min_overall, 0), "y_range": (0, y_max_overall)},
        "Quadrant III": {"x_range": (x_min_overall, 0), "y_range": (y_min_overall, 0)},
        "Quadrant IV": {"x_range": (0, x_max_overall), "y_range": (y_min_overall, 0)},
    }

    for q_name, q_info in quadrants.items():
        x_start, x_end = q_info["x_range"]
        y_start, y_end = q_info["y_range"]

        # 绘制象限方框
        rect_x = min(x_start, x_end)
        rect_y = min(y_start, y_end)
        rect_width = abs(x_end - x_start)
        rect_height = abs(y_end - y_start)

        rect = Rectangle(
            (rect_x, rect_y),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor="dimgray",
            facecolor="none",
            linestyle=":",
            # zorder=0,  # 确保在最底层
        )
        ax.add_patch(rect)

        # 只有提供了真实样本点时才计算和显示误差
        if sample_points_true is not None:
            # 筛选出落在当前象限的理想中心点
            indices_in_quadrant = np.where(
                (ideal_centers[:, 0] >= min(x_start, x_end))
                & (ideal_centers[:, 0] <= max(x_start, x_end))
                & (ideal_centers[:, 1] >= min(y_start, y_end))
                & (ideal_centers[:, 1] <= max(y_start, y_end))
            )[0]

            # 计算该象限内所有点的平均MAE
            mae_x_quadrant = np.abs(
                sample_points_true[:, indices_in_quadrant, 0]
                - sample_points_pred[:, indices_in_quadrant, 0]
            ).mean()
            mae_y_quadrant = np.abs(
                sample_points_true[:, indices_in_quadrant, 1]
                - sample_points_pred[:, indices_in_quadrant, 1]
            ).mean()

            # 添加误差文本
            info_text = (
                f"{q_name}\nMAE X: {mae_x_quadrant:.2f}\nMAE Y: {mae_y_quadrant:.2f}"
            )

            # 统一放在方框右下角
            text_x = max(x_start, x_end) - text_offset  # 靠近右边缘
            text_y = min(y_start, y_end) + text_offset  # 靠近下边缘
            ha_val = "right"
            va_val = "bottom"

            ax.text(
                text_x,
                text_y,
                info_text,
                ha=ha_val,
                va=va_val,
                fontsize=10,
                color="dimgray",
                zorder=1,
            )


def _draw_lines_between_centers(
    ax: Axes,
    center_coord: np.ndarray,
    edge_indices: list | np.ndarray,
    c: str,
):
    """
    绘制理想中心点、真实样本中心点和预测样本中心点之间的连线。
    """
    for edge in edge_indices:
        i, j = edge
        ax.plot(
            [center_coord[i, 0], center_coord[j, 0]],
            [center_coord[i, 1], center_coord[j, 1]],
            c=c,
            linestyle="-",
            alpha=0.7,
        )


def _set_plot_style(ax: Axes):
    """
    设置图表标题、轴标签、刻度以及图例。
    """
    ax.set_title("BMTP Map Plot", fontsize=16, fontweight="bold")
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.axis("equal")

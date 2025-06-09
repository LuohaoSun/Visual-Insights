"""
空间可视化模块

提供用于在空间坐标上绘制图形、值和字符串的函数。
"""

import io
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# 类型别名
CoordinatePair = Tuple[float, float]
FigureDict = Dict[CoordinatePair, Figure]
ValueDict = Dict[CoordinatePair, float]
StringDict = Dict[CoordinatePair, str]


def plot_spatial_figures(
    figures: List[Figure],
    x_coords: Union[List[float], pd.Series],
    y_coords: Union[List[float], pd.Series],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12),
    show: bool = True,
) -> Figure:
    """
    Plot multiple figures at their spatial coordinates (original spacing logic).
    """
    if not figures:
        raise ValueError("Empty figures list provided")

    unique_x = np.sort(np.unique(x_coords))
    unique_y = np.sort(np.unique(y_coords))  # Smallest y first

    fig = plt.figure(figsize=figsize, dpi=300)

    if title:
        fig.suptitle(title, fontsize=16)

    n_rows = len(unique_y)
    n_cols = len(unique_x)

    # Use original logic for cell size calculation (implicitly includes margins)
    # Avoid division by zero if only one row/column
    cell_width = 1.0 / (n_cols + 1) if n_cols > 0 else 1.0
    cell_height = 1.0 / (n_rows + 1) if n_rows > 0 else 1.0

    # Subplot size fills the calculated cell
    subplot_width = cell_width
    subplot_height = cell_height

    for x, y, subfig in zip(x_coords, y_coords, figures):
        x_idx = np.where(unique_x == x)[0][0]
        y_idx = np.where(unique_y == y)[0][0]  # Use direct y_idx for vertical position

        # Calculate position using original centering logic
        # Use y_idx (smallest y -> smallest index -> lowest position)
        left = (
            (x_idx + 0.5) * cell_width - subplot_width / 2 + (cell_width / 2)
        )  # Adjusted offset
        bottom = (
            (y_idx + 0.5) * cell_height - subplot_height / 2 + (cell_height / 2)
        )  # Adjusted offset

        # Add a small margin adjustment if needed (optional, fine-tuning)
        margin_adjust_factor = 0.05  # e.g., 5% of cell size as margin
        left += cell_width * margin_adjust_factor / 2
        bottom += cell_height * margin_adjust_factor / 2
        plot_w = subplot_width * (1 - margin_adjust_factor)
        plot_h = subplot_height * (1 - margin_adjust_factor)

        # Ensure position is within [0, 1] bounds (robustness)
        left = max(0, left)
        bottom = max(0, bottom)
        plot_w = min(plot_w, 1 - left)
        plot_h = min(plot_h, 1 - bottom)

        # Create axes using tuple
        ax = fig.add_axes((left, bottom, plot_w, plot_h))

        buf = io.BytesIO()
        subfig.savefig(buf, format="png", dpi=subfig.dpi)
        buf.seek(0)
        plt.close(subfig)

        img = mpimg.imread(buf)
        buf.close()

        ax.imshow(img)
        ax.axis("off")

    # Removed fig.tight_layout()

    if show:
        plt.show()

    return fig


# --- Keep the other functions (plot_spatial_values, plot_spatial_strings) as they are ---
# (Assuming plot_spatial_values works as intended)


def plot_spatial_values(
    values: Union[List[float], pd.Series],
    x_coords: Union[List[float], pd.Series],
    y_coords: Union[List[float], pd.Series],
    cmap="coolwarm",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12),
    show: bool = True,
):
    """
    根据坐标-值字典绘制热力图

    Parameters:
        values: 数值列表或Pandas系列
        x_coords: X坐标列表或Pandas系列
        y_coords: Y坐标列表或Pandas系列
        figsize: 图形大小
        cmap: 颜色映射
        title: 图形标题

    Returns:
        matplotlib Figure对象
    """
    if len(values) != len(x_coords) or len(values) != len(y_coords):
        raise ValueError("Values, x_coords, and y_coords must have the same length.")

    # 获取唯一的X和Y坐标
    unique_x = np.sort(np.unique(x_coords))
    unique_y = np.sort(np.unique(y_coords))

    # 创建网格
    grid_shape = (len(unique_y), len(unique_x))
    grid = np.full(grid_shape, np.nan)  # 初始化为NaN

    # 填充网格
    for x, y, value in zip(x_coords, y_coords, values):
        x_idx = np.where(unique_x == x)[0][0]
        y_idx = np.where(unique_y == y)[0][0]
        # 反转y索引使原点在左下角 (for imshow where row 0 is top)
        grid[len(unique_y) - 1 - y_idx, x_idx] = value

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 设置标题
    if title:
        ax.set_title(title, fontsize=14)

    # Find min/max for text color normalization
    valid_values = grid[~np.isnan(grid)]
    if len(valid_values) > 0:
        vmin = np.nanmin(grid)
        vmax = np.nanmax(grid)
        # Threshold for text color inversion
        text_threshold = (vmax - vmin) / 2 + vmin
    else:
        vmin, vmax = 0, 1  # Default if no valid data
        text_threshold = 0.5

    # 创建热力图
    im = ax.imshow(
        grid, cmap=cmap, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax
    )

    # 添加colorbar
    cbar = fig.colorbar(im, ax=ax)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(len(unique_x)))
    ax.set_yticks(np.arange(len(unique_y)))

    # 设置坐标轴标签
    ax.set_xticklabels([f"{x:.1f}" for x in unique_x])
    # Reverse Y labels to match visual bottom-up representation
    ax.set_yticklabels([f"{y:.1f}" for y in unique_y[::-1]])

    # 旋转X轴标签以避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 添加网格线
    ax.set_xticks(np.arange(-0.5, len(unique_x), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(unique_y), 1), minor=True)
    # Adjust grid appearance
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)  # Hide minor tick marks

    # 在每个单元格中添加数值
    for i in range(len(unique_y)):
        for j in range(len(unique_x)):
            value = grid[i, j]
            if not np.isnan(value):
                # Determine text color based on background brightness
                text_color = "white" if abs(value) > text_threshold else "black"
                # Adjust text color slightly based on cmap if needed (more complex)
                # A simple threshold usually works ok for diverging cmaps like coolwarm

                ax.text(
                    j,  # x-position in grid index
                    i,  # y-position in grid index
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_spatial_texts(
    texts: Union[List[str], pd.Series],
    x_coords: Union[List[float], pd.Series],
    y_coords: Union[List[float], pd.Series],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 12),
    fontsize: int = 10,
    show: bool = True,
) -> Figure:
    """
    Plot strings at their spatial coordinates.

    Parameters:
        strings_dict: Dictionary mapping coordinate pairs (x, y) to strings
        title: Main figure title
        figsize: Size of the figure
        fontsize: Base font size for the text
        show: Whether to show the figure immediately

    Returns:
        Matplotlib figure
    """
    if len(texts) != len(x_coords) or len(texts) != len(y_coords):
        raise ValueError("Strings, x_coords, and y_coords must have the same length.")

    # Get unique X and Y coordinates
    unique_x = np.sort(np.unique(x_coords))
    unique_y = np.sort(np.unique(y_coords))

    # Create grid
    grid_shape = (len(unique_y), len(unique_x))
    grid = np.full(grid_shape, "", dtype=object)  # Initialize with empty strings

    # Fill the grid with strings
    for x, y, text in zip(x_coords, y_coords, texts):
        x_idx = np.where(unique_x == x)[0][0]
        y_idx = np.where(unique_y == y)[0][0]
        # Invert y index for display grid (row 0 at top)
        grid[len(unique_y) - 1 - y_idx, x_idx] = text

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set title
    if title:
        ax.set_title(title, fontsize=14)

    # Create a blank canvas (imshow sets up axes correctly for grid)
    ax.imshow(np.zeros(grid_shape), cmap="Greys", alpha=0)  # Use zeros for background

    # Add text to each cell
    for i in range(len(unique_y)):  # Grid row index (top is 0)
        for j in range(len(unique_x)):  # Grid col index (left is 0)
            text = grid[i, j]
            if text:
                # Process escape characters if necessary
                try:
                    # Attempt to decode potential escape sequences
                    processed_text = text.encode("latin-1", "backslashreplace").decode(
                        "unicode-escape"
                    )
                except Exception:
                    processed_text = text  # Fallback if decoding fails

                ax.text(
                    j,  # x-position in grid index
                    i,  # y-position in grid index
                    processed_text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    wrap=True,  # Allows text wrapping within cell bounds (approx)
                )

    # Set axis ticks
    ax.set_xticks(np.arange(len(unique_x)))
    ax.set_yticks(np.arange(len(unique_y)))

    # Set axis labels
    ax.set_xticklabels([f"{x:.1f}" for x in unique_x])
    # Reverse Y labels to match visual bottom-up representation
    ax.set_yticklabels([f"{y:.1f}" for y in unique_y[::-1]])

    # Rotate X axis labels to avoid overlap
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add grid lines for visual separation
    ax.set_xticks(np.arange(-0.5, len(unique_x), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(unique_y), 1), minor=True)
    # Use a visible color like lightgrey
    ax.grid(which="minor", color="lightgrey", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)  # Hide minor tick marks

    # Hide major tick marks if desired
    ax.tick_params(axis="both", which="major", length=0)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    plt.tight_layout()
    if show:
        plt.show()
    return fig

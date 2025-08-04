import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

from .utils.data_handlers import ArrayLike, ensure_numpy_array
from .utils.visualization import create_figure


def _validate_and_prepare_data(
    input: ArrayLike, target: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Validate input data and convert to numpy arrays."""
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

    return input_arr, target_arr


def _filter_nan_values(
    input_arr: np.ndarray, target_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Filter out NaN values from input and target arrays."""
    valid_mask = ~np.isnan(input_arr) & ~np.isnan(target_arr)

    if not np.any(valid_mask):
        raise ValueError(
            "All data points contain NaN values. Cannot create scatter plot."
        )

    return input_arr[valid_mask], target_arr[valid_mask]


def _calculate_mean_lines(
    input_arr: np.ndarray, target_arr: np.ndarray
) -> tuple[float, float]:
    """Calculate mean values for x and y axes."""
    return float(np.mean(input_arr)), float(np.mean(target_arr))


def _calculate_linear_regression(
    input_arr: np.ndarray, target_arr: np.ndarray
) -> dict[str, float]:
    """Calculate linear regression statistics."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        input_arr, target_arr
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "r_squared": float(r_value) ** 2,
    }


def _calculate_spearman_correlation(
    input_arr: np.ndarray, target_arr: np.ndarray
) -> dict[str, float]:
    """Calculate Spearman correlation statistics."""
    correlation, p_value = stats.spearmanr(input_arr, target_arr)
    return {"correlation": float(correlation), "p_value": float(p_value)}


def _plot_mean_lines(ax, x_mean: float, y_mean: float) -> None:
    """Plot mean lines on the axes."""
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


def _plot_linear_fit(ax, input_arr: np.ndarray, target_arr: np.ndarray) -> None:
    """Plot linear regression fit line."""
    reg_stats = _calculate_linear_regression(input_arr, target_arr)
    line_x = np.array([np.min(input_arr), np.max(input_arr)])
    line_y = reg_stats["slope"] * line_x + reg_stats["intercept"]

    ax.plot(
        line_x,
        line_y,
        color="purple",
        linestyle="-",
        linewidth=2,
        label=f"Linear Fit: y={reg_stats['slope']:.2f}x+{reg_stats['intercept']:.2f}\nR²={reg_stats['r_squared']:.2f}, p-lin={reg_stats['p_value']:.3f}",
    )


def _plot_spearman_info(ax, input_arr: np.ndarray, target_arr: np.ndarray) -> None:
    """Plot Spearman correlation information."""
    spearman_stats = _calculate_spearman_correlation(input_arr, target_arr)
    ax.plot(
        [],
        [],
        " ",
        label=f"Spearman's ρ: {spearman_stats['correlation']:.2f}, p-value: {spearman_stats['p_value']:.3f}",
    )


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
    """
    # Validate and prepare data
    input_arr, target_arr = _validate_and_prepare_data(input, target)

    # Filter out NaN values
    input_arr, target_arr = _filter_nan_values(input_arr, target_arr)

    fig, ax = create_figure(figsize=figsize)

    # Plot scatter points
    sns.scatterplot(
        x=input_arr,
        y=target_arr,
        ax=ax,
        s=50,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
        label="Data points",
    )

    # Plot mean lines if requested
    if show_mean_lines:
        x_mean, y_mean = _calculate_mean_lines(input_arr, target_arr)
        _plot_mean_lines(ax, x_mean, y_mean)

    # Plot linear regression if requested
    if show_linear_fit:
        _plot_linear_fit(ax, input_arr, target_arr)

    # Plot Spearman correlation if requested
    if show_spearman:
        _plot_spearman_info(ax, input_arr, target_arr)

    _feature_name = feature_name if feature_name else "Feature"
    _target_name = target_name if target_name else "Target"

    ax.set_xlabel(_feature_name, fontsize=12)
    ax.set_ylabel(_target_name, fontsize=12)

    plot_title = (
        title
        if title is not None
        else f"Scatter Plot: {_target_name} vs {_feature_name}"
    )
    ax.set_title(plot_title, fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    legend = ax.legend(frameon=True, loc="best", fontsize=9)
    legend.get_frame().set_facecolor("aliceblue")
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_edgecolor("lightgray")

    try:
        plt.tight_layout()
    except RuntimeError:
        pass

    if show:
        plt.show()

    return fig

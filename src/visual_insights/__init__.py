"""
VisualInsight: 一个数据和模型可视化模块

提供了一系列函数用于可视化数据关系、模型性能和模型解释。
"""

from .box_plot import plot_binned_box_plot, plot_box_plot
from .error import plot_actual_vs_predicted, plot_error_distribution, plot_residuals
from .metrics import calculate_regression_metrics
from .offset_map import plot_offset_map
from .shap import (
    get_shap_values,
    plot_shap_dependence,
    plot_shap_force,
    plot_shap_summary,
    plot_shap_waterfall,
)
from .sobol import (
    get_sobol_indices,
    plot_sobol_indices,
    plot_sobol_interactions,
    plot_sobol_radar,
)
from .spatial import plot_spatial_figures, plot_spatial_texts, plot_spatial_values
from .statistical import (
    plot_correlation_matrix,
    plot_distribution,
    plot_mutual_information,
    plot_scatter,
    plot_statistical_tests,
    plot_statistical_tests_heatmap,
)
from .utils.cache import clear_cache

__all__ = [
    # metrics
    "calculate_regression_metrics",
    # error
    "plot_error_distribution",
    "plot_residuals",
    "plot_actual_vs_predicted",
    # shap
    "plot_shap_summary",
    "plot_shap_dependence",
    "plot_shap_waterfall",
    "plot_shap_force",
    "get_shap_values",
    # sobol
    "plot_sobol_indices",
    "plot_sobol_radar",
    "plot_sobol_interactions",
    "get_sobol_indices",
    # statistical
    "plot_statistical_tests",
    "plot_statistical_tests_heatmap",
    "plot_correlation_matrix",
    "plot_scatter",
    "plot_distribution",
    "plot_mutual_information",
    # box_plot
    "plot_binned_box_plot",
    "plot_box_plot",
    # spatial
    "plot_spatial_figures",
    "plot_spatial_values",
    "plot_spatial_texts",
    # map
    "plot_offset_map",
    # utils
    "clear_cache",
]

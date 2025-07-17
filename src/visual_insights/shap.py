"""
SHAP 可视化和计算模块

提供用于计算和可视化 SHAP 值的函数。
"""

import logging
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory
from matplotlib.figure import Figure

from visual_insights.common.typing import ArrayLike
from visual_insights.utils.data_handlers import ensure_numpy_array

__all__ = [
    "plot_shap_summary",
    "plot_shap_dependence",
    "plot_shap_waterfall",
    "plot_shap_force",
    "get_shap_values",
]

logger = logging.getLogger(__name__)
memory = Memory(location=".cache/shap", verbose=0)

# 尝试导入 SHAP，但如果不可用则不会失败
try:
    import shap
    from shap.explainers._explainer import Explainer

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not found. SHAP visualization will not be available.")

ModelTypeLiteral = Literal["sklearn", "xgboost", "torch"]
ExplainerTypeLiteral = Literal["tree", "deep", "linear", "kernel"]


def _detect_explainer_type(model: Any) -> ExplainerTypeLiteral:
    """
    检测适合模型的 SHAP 解释器类型

    参数:
        model: 要解释的模型

    返回:
        解释器类型 ('tree', 'deep', 'linear', 'kernel')
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    model_type = str(type(model).__name__).lower()

    if any(
        x in model_type for x in ["tree", "forest", "gbm", "xgb", "lgbm", "catboost"]
    ):
        return "tree"
    elif any(
        x in model_type for x in ["linear", "logistic", "regression", "lasso", "ridge"]
    ):
        return "linear"
    elif any(
        x in model_type
        for x in ["nn", "net", "network", "keras", "torch", "tensorflow"]
    ):
        return "deep"
    else:
        return "kernel"


def _create_explainer(
    model: Any,
    model_type: ModelTypeLiteral | None,
    explainer_type: ExplainerTypeLiteral | None,
    input_data: ArrayLike,
) -> "Explainer":
    """
    创建 SHAP 解释器

    参数:
        model: 要解释的模型
        model_type: 模型类型
        explainer_type: 解释器类型
        input_data: 输入数据

    返回:
        SHAP 解释器
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 如果未提供解释器类型，尝试检测
    if explainer_type is None:
        explainer_type = _detect_explainer_type(model)

    # 转换为 numpy 数组
    x = ensure_numpy_array(input_data)

    # 创建解释器
    if explainer_type == "tree":
        if model_type and "xgb" in model_type.lower():
            return shap.TreeExplainer(model)
        else:
            return shap.TreeExplainer(model)
    elif explainer_type == "deep":
        return shap.DeepExplainer(model, x[:100])  # 使用前100个样本作为背景
    elif explainer_type == "linear":
        return shap.LinearExplainer(model, x)
    elif explainer_type == "kernel":
        return shap.KernelExplainer(model.predict, x[:100])  # 使用前100个样本作为背景
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")


@memory.cache
def _calculate_shap_values(
    explainer: "Explainer", input_data: ArrayLike, output_dimension: int | None = None
) -> Any:
    """
    计算 SHAP 值

    参数:
        explainer: SHAP 解释器
        input_data: 输入数据
        output_dimension: 输出维度

    返回:
        SHAP 值
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 转换为 numpy 数组
    x = ensure_numpy_array(input_data)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(x)  # type: ignore

    # 处理多输出情况
    if isinstance(shap_values, list) and output_dimension is not None:
        return shap_values[output_dimension]

    elif (
        isinstance(shap_values, np.ndarray)
        and len(shap_values.shape) == 3
        and output_dimension is not None
    ):
        return shap_values[:, :, output_dimension]

    return shap_values


@memory.cache
def _get_explainer(
    model: Any,
    input_data: ArrayLike,
    model_type: Literal["sklearn", "xgboost", "torch"] | None = None,
    explainer_type: Literal["tree", "deep", "linear", "kernel"] | None = None,
) -> "Explainer":
    """
    获取或创建 SHAP 解释器

    参数:
        model: 要解释的模型
        input_data: 输入数据
        model_type: 模型类型
        explainer_type: 解释器类型
        use_cache: 是否使用缓存

    返回:
        SHAP 解释器
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    explainer = _create_explainer(model, model_type, explainer_type, input_data)

    return explainer


@memory.cache
def _get_shap_values(
    explainer: "Explainer",
    input_data: ArrayLike,
    output_dimension: int | None = None,
) -> Any:
    """
    获取或计算 SHAP 值

    参数:
        explainer: SHAP 解释器
        input_data: 输入数据
        output_dimension: 输出维度
        use_cache: 是否使用缓存

    返回:
        SHAP 值
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    return _calculate_shap_values(explainer, input_data, output_dimension)


def plot_shap_summary(
    input: ArrayLike,
    model: Any,
    model_type: ModelTypeLiteral | None = None,
    explainer_type: ExplainerTypeLiteral | None = None,
    feature_names: list[str] | None = None,
    max_display: int = 20,
    plot_type: str = "dot",
    output_dimension: int | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    绘制 SHAP 摘要图

    参数:
        input: 输入数据
        model: 要解释的模型
        model_type: 模型类型 ("sklearn", "xgboost", "torch" 等)
        explainer_type: SHAP 解释器类型 (tree, deep, linear, kernel)
        feature_names: 特征名称列表
        max_display: 最多显示的特征数量
        plot_type: 图表类型 ('bar', 'dot', 'violin')
        output_dimension: 输出维度（对于多输出模型）
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表

    返回:
        matplotlib Figure 对象
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension)
    print(f"shap_values: {shap_values.shape}")

    # 处理特征名称
    if isinstance(input, pd.DataFrame) and feature_names is None:
        feature_names = list(input.columns)

    # 创建图表
    plt.figure(figsize=figsize)

    # 根据图表类型绘制
    if plot_type == "bar":
        shap.summary_plot(
            shap_values,
            input,
            feature_names=feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False,
        )
    elif plot_type == "violin":
        shap.summary_plot(
            shap_values,
            input,
            feature_names=feature_names,
            max_display=max_display,
            plot_type="violin",
            show=False,
        )
    else:  # 默认为点图
        shap.summary_plot(
            shap_values,
            input,
            feature_names=feature_names,
            max_display=max_display,
            plot_type="dot",
            show=False,
        )

    # 设置标题
    if title:
        plt.title(title)

    # 调整布局
    plt.tight_layout()

    # 获取当前图表
    fig = plt.gcf()

    if show:
        plt.show()

    return fig


def plot_shap_dependence(
    input: ArrayLike,
    model: Any,
    feature_idx: int | str,
    model_type: ModelTypeLiteral | None = None,
    explainer_type: ExplainerTypeLiteral | None = None,
    interaction_idx: int | str | Literal["auto"] = "auto",
    feature_names: list[str] | None = None,
    output_dimension: int | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    绘制 SHAP 依赖图

    参数:
        input: 输入数据
        model: 要解释的模型
        feature_idx: 要分析的特征索引或名称
        model_type: 模型类型
        explainer_type: SHAP 解释器类型
        interaction_idx: 交互特征索引或名称
        feature_names: 特征名称列表
        output_dimension: 输出维度（对于多输出模型）
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表

    返回:
        matplotlib Figure 对象
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension)

    # 处理特征名称
    if isinstance(input, pd.DataFrame) and feature_names is None:
        feature_names = list(input.columns)

    # 如果提供了特征名称，将特征索引转换为整数
    if isinstance(feature_idx, str) and feature_names is not None:
        if feature_idx in feature_names:
            feature_idx = feature_names.index(feature_idx)
        else:
            raise ValueError(f"Feature name '{feature_idx}' not found in feature_names")

    # 如果提供了交互特征名称，将其转换为整数
    if (
        isinstance(interaction_idx, str)
        and interaction_idx != "auto"
        and feature_names is not None
    ):
        if interaction_idx in feature_names:
            interaction_idx = feature_names.index(interaction_idx)
        else:
            raise ValueError(
                f"Interaction feature name '{interaction_idx}' \
                    not found in feature_names"
            )

    # 创建图表
    plt.figure(figsize=figsize)

    # 绘制依赖图
    shap.dependence_plot(
        feature_idx,
        shap_values,
        input,
        interaction_index=interaction_idx,
        feature_names=feature_names,
        show=False,
    )

    # 设置标题
    if title:
        plt.title(title)

    # 调整布局
    plt.tight_layout()

    # 获取当前图表
    fig = plt.gcf()

    if show:
        plt.show()

    return fig


def plot_shap_waterfall(
    input: ArrayLike,
    model: Any,
    instance_idx: int = 0,
    model_type: ModelTypeLiteral | None = None,
    explainer_type: ExplainerTypeLiteral | None = None,
    feature_names: list[str] | None = None,
    output_dimension: int | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> Figure:
    """
    绘制 SHAP 瀑布图

    参数:
        input: 输入数据
        model: 要解释的模型
        instance_idx: 实例索引
        model_type: 模型类型
        explainer_type: SHAP 解释器类型
        feature_names: 特征名称列表
        output_dimension: 输出维度（对于多输出模型）
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表

    返回:
        matplotlib Figure 对象
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension)

    # 处理特征名称
    if isinstance(input, pd.DataFrame) and feature_names is None:
        feature_names = list(input.columns)

    # 创建图表
    plt.figure(figsize=figsize)

    # 获取单个实例的 SHAP 值
    if isinstance(shap_values, np.ndarray):
        instance_shap = shap_values[instance_idx]
        instance_value = ensure_numpy_array(input)[instance_idx]
    else:
        # 对于某些解释器，shap_values 可能是一个列表
        instance_shap = (
            shap_values[instance_idx] if isinstance(shap_values, list) else shap_values
        )
        instance_value = ensure_numpy_array(input)[instance_idx]

    # 绘制瀑布图
    shap.plots.waterfall(
        shap.Explanation(
            values=instance_shap,
            base_values=explainer.expected_value  # type: ignore
            if hasattr(explainer, "expected_value")
            else 0,
            data=instance_value,
            feature_names=feature_names,
        ),
        show=False,
    )

    # 设置标题
    if title:
        plt.title(title)

    # 调整布局
    plt.tight_layout()

    # 获取当前图表
    fig = plt.gcf()

    if show:
        plt.show()

    return fig


def get_shap_values(
    input: ArrayLike,
    model: Any,
    model_type: ModelTypeLiteral | None = None,
    explainer_type: ExplainerTypeLiteral | None = None,
    output_dimension: int | None = None,
) -> dict[str, Any]:
    """
    计算并返回 SHAP 值

    参数:
        input: 输入数据
        model: 要解释的模型
        model_type: 模型类型 ("sklearn", "xgboost", "torch" 等)
        explainer_type: SHAP 解释器类型 (tree, deep, linear, kernel)
        output_dimension: 输出维度（对于多输出模型）

    返回:
        包含 SHAP 值和相关信息的字典
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension)

    # 处理特征名称
    if isinstance(input, pd.DataFrame):
        feature_names = list(input.columns)
    else:
        feature_names = [
            f"Feature {i}" for i in range(ensure_numpy_array(input).shape[1])
        ]

    # 获取预期值（基准值）
    if hasattr(explainer, "expected_value"):
        expected_value = explainer.expected_value  # type: ignore
    else:
        expected_value = 0

    # 创建结果字典
    result = {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "feature_names": feature_names,
    }

    return result


def plot_shap_force(
    input: ArrayLike,
    model: Any,
    instance_idx: int = 0,
    model_type: ModelTypeLiteral | None = None,
    explainer_type: ExplainerTypeLiteral | None = None,
    feature_names: list[str] | None = None,
    output_dimension: int | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 3),
    show: bool = True,
) -> Figure:
    """
    绘制 SHAP 力图

    参数:
        input: 输入数据
        model: 要解释的模型
        instance_idx: 实例索引
        model_type: 模型类型
        explainer_type: SHAP 解释器类型
        feature_names: 特征名称列表
        output_dimension: 输出维度（对于多输出模型）
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表

    返回:
        matplotlib Figure 对象
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension)

    # 处理特征名称
    if isinstance(input, pd.DataFrame) and feature_names is None:
        feature_names = list(input.columns)

    # 创建图表
    plt.figure(figsize=figsize)

    # 获取单个实例的 SHAP 值
    if isinstance(shap_values, np.ndarray):
        instance_shap = shap_values[instance_idx]
        instance_value = ensure_numpy_array(input)[instance_idx]
    else:
        # 对于某些解释器，shap_values 可能是一个列表
        instance_shap = (
            shap_values[instance_idx] if isinstance(shap_values, list) else shap_values
        )
        instance_value = ensure_numpy_array(input)[instance_idx]

    # 绘制力图
    shap.plots.force(
        shap.Explanation(
            values=instance_shap,
            base_values=explainer.expected_value  # type: ignore
            if hasattr(explainer, "expected_value")
            else 0,
            data=instance_value,
            feature_names=feature_names,
        ),
        show=False,
    )

    # 设置标题
    if title:
        plt.title(title)

    # 调整布局
    plt.tight_layout()

    # 获取当前图表
    fig = plt.gcf()

    if show:
        plt.show()

    return fig

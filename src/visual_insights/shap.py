"""
SHAP 可视化和计算模块

提供用于计算和可视化 SHAP 值的函数。
"""

import hashlib
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .utils.cache import explainer_cache, shap_values_cache
from .utils.data_handlers import ArrayLike, ensure_numpy_array

logger = logging.getLogger(__name__)

# 尝试导入 SHAP，但如果不可用则不会失败
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not found. SHAP visualization will not be available.")


def _create_cache_key(
    model: Any, model_type: Optional[str], explainer_type: Optional[str]
) -> str:
    """
    为模型创建唯一的缓存键

    参数:
        model: 模型
        model_type: 模型类型
        explainer_type: 解释器类型

    返回:
        缓存键
    """
    # 使用模型的内存地址作为唯一标识
    model_id = str(id(model))
    key_parts = [model_id, str(model_type), str(explainer_type)]
    return "_".join(key_parts)


def _create_data_cache_key(
    explainer: Any, input_data: ArrayLike, output_dimension: Optional[int]
) -> str:
    """
    为数据创建唯一的缓存键

    参数:
        explainer: SHAP 解释器
        input_data: 输入数据
        output_dimension: 输出维度

    返回:
        缓存键
    """
    # 使用解释器的内存地址和数据的哈希值
    explainer_id = str(id(explainer))

    # 为数据创建哈希值
    data_array = ensure_numpy_array(input_data)
    data_hash = hashlib.md5(np.ascontiguousarray(data_array).tobytes()).hexdigest()

    key_parts = [explainer_id, data_hash, str(output_dimension)]
    return "_".join(key_parts)


def _detect_explainer_type(model: Any) -> str:
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
    model_type: Optional[str],
    explainer_type: Optional[str],
    input_data: ArrayLike,
) -> Any:
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
    X = ensure_numpy_array(input_data)

    # 创建解释器
    if explainer_type == "tree":
        if model_type and "xgb" in model_type.lower():
            return shap.TreeExplainer(model, feature_perturbation="interventional")
        else:
            return shap.TreeExplainer(model)
    elif explainer_type == "deep":
        return shap.DeepExplainer(model, X[:100])  # 使用前100个样本作为背景
    elif explainer_type == "linear":
        return shap.LinearExplainer(model, X)
    elif explainer_type == "kernel":
        return shap.KernelExplainer(model.predict, X[:100])  # 使用前100个样本作为背景
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")


def _calculate_shap_values(
    explainer: Any, input_data: ArrayLike, output_dimension: Optional[int] = None
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
    X = ensure_numpy_array(input_data)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(X)

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


def _get_explainer(
    model: Any,
    input_data: ArrayLike,
    model_type: Optional[str] = None,
    explainer_type: Optional[str] = None,
    use_cache: bool = True,
) -> Any:
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

    if not use_cache:
        return _create_explainer(model, model_type, explainer_type, input_data)

    # 创建缓存键
    cache_key = _create_cache_key(model, model_type, explainer_type)

    # 检查缓存
    explainer = explainer_cache.get(cache_key)
    if explainer is not None:
        return explainer

    # 创建新的解释器并缓存
    explainer = _create_explainer(model, model_type, explainer_type, input_data)
    explainer_cache.set(cache_key, explainer)
    return explainer


def _get_shap_values(
    explainer: Any,
    input_data: ArrayLike,
    output_dimension: Optional[int] = None,
    use_cache: bool = True,
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

    if not use_cache:
        return _calculate_shap_values(explainer, input_data, output_dimension)

    # 创建缓存键
    cache_key = _create_data_cache_key(explainer, input_data, output_dimension)

    # 检查缓存
    shap_values = shap_values_cache.get(cache_key)
    if shap_values is not None:
        return shap_values

    # 计算新的 SHAP 值并缓存
    shap_values = _calculate_shap_values(explainer, input_data, output_dimension)
    shap_values_cache.set(cache_key, shap_values)
    return shap_values


def plot_shap_summary(
    input: ArrayLike,
    model: Any,
    model_type: Optional[str] = None,
    explainer_type: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
    plot_type: str = "dot",
    output_dimension: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
    use_cache: bool = True,
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
        use_cache: 是否使用缓存

    返回:
        matplotlib Figure 对象
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type, use_cache)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension, use_cache)
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
    feature_idx: Union[int, str],
    model_type: Optional[str] = None,
    explainer_type: Optional[str] = None,
    interaction_idx: Union[int, str, Literal["auto"]] = "auto",
    feature_names: Optional[List[str]] = None,
    output_dimension: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
    use_cache: bool = True,
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
        use_cache: 是否使用缓存

    返回:
        matplotlib Figure 对象
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type, use_cache)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension, use_cache)

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
                f"Interaction feature name '{interaction_idx}' not found in feature_names"
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
    model_type: Optional[str] = None,
    explainer_type: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    output_dimension: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
    use_cache: bool = True,
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
        use_cache: 是否使用缓存

    返回:
        matplotlib Figure 对象
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type, use_cache)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension, use_cache)

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
            base_values=explainer.expected_value
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
    model_type: Optional[str] = None,
    explainer_type: Optional[str] = None,
    output_dimension: Optional[int] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    计算并返回 SHAP 值

    参数:
        input: 输入数据
        model: 要解释的模型
        model_type: 模型类型 ("sklearn", "xgboost", "torch" 等)
        explainer_type: SHAP 解释器类型 (tree, deep, linear, kernel)
        output_dimension: 输出维度（对于多输出模型）
        use_cache: 是否使用缓存

    返回:
        包含 SHAP 值和相关信息的字典
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type, use_cache)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension, use_cache)

    # 处理特征名称
    if isinstance(input, pd.DataFrame):
        feature_names = list(input.columns)
    else:
        feature_names = [
            f"Feature {i}" for i in range(ensure_numpy_array(input).shape[1])
        ]

    # 获取预期值（基准值）
    if hasattr(explainer, "expected_value"):
        expected_value = explainer.expected_value
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
    model_type: Optional[str] = None,
    explainer_type: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    output_dimension: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 3),
    show: bool = True,
    use_cache: bool = True,
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
        use_cache: 是否使用缓存

    返回:
        matplotlib Figure 对象
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is required. Install with 'pip install shap'.")

    # 获取 SHAP 解释器
    explainer = _get_explainer(model, input, model_type, explainer_type, use_cache)

    # 计算 SHAP 值
    shap_values = _get_shap_values(explainer, input, output_dimension, use_cache)

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
            base_values=explainer.expected_value
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

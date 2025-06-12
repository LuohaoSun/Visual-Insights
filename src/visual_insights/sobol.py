"""
Sobol 可视化和计算模块

提供用于计算和可视化 Sobol 敏感性分析结果的函数。
"""

import hashlib
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .utils.cache import sobol_indices_cache
from .utils.data_handlers import ArrayLike, ensure_numpy_array, handle_feature_names
from .utils.visualization import create_figure

logger = logging.getLogger(__name__)

# 尝试导入 SALib，但如果不可用则不会失败
try:
    from SALib.analyze import sobol
    from SALib.sample import saltelli

    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logger.warning(
        "SALib library not found. Sobol visualization will not be available."
    )


def _create_cache_key(
    model: Any,
    input_data: ArrayLike,
    model_type: Optional[str],
    n_samples: int,
    output_dimension: Optional[int],
) -> str:
    """
    为 Sobol 分析创建唯一的缓存键

    参数:
        model: 模型
        input_data: 输入数据
        model_type: 模型类型
        n_samples: 样本数量
        output_dimension: 输出维度

    返回:
        缓存键
    """
    # 使用模型的内存地址作为唯一标识
    model_id = str(id(model))

    # 为数据创建哈希值
    data_array = ensure_numpy_array(input_data)
    data_hash = hashlib.md5(
        np.ascontiguousarray(data_array[:100]).tobytes()
    ).hexdigest()  # 只使用前100个样本

    key_parts = [
        model_id,
        data_hash,
        str(model_type),
        str(n_samples),
        str(output_dimension),
    ]
    return "_".join(key_parts)


def _get_model_predict_function(
    model: Any, model_type: Optional[str], output_dimension: Optional[int] = None
):
    """
    获取模型的预测函数

    参数:
        model: 模型
        model_type: 模型类型
        output_dimension: 输出维度

    返回:
        预测函数
    """
    if model_type is None:
        # 尝试自动检测模型类型
        model_class = model.__class__.__name__.lower()
        if "sklearn" in str(model.__class__.__module__).lower() or any(
            x in model_class for x in ["forest", "tree", "boost", "linear"]
        ):
            model_type = "sklearn"
        elif "xgb" in model_class or "xgboost" in model_class:
            model_type = "xgboost"
        elif "torch" in model_class or "module" in model_class:
            model_type = "torch"
        elif "keras" in model_class or "tensorflow" in model_class:
            model_type = "tensorflow"
        else:
            model_type = "sklearn"  # 默认

    # 根据模型类型创建预测函数
    if model_type.lower() in ["sklearn", "xgboost", "xgb"]:
        if output_dimension is None:
            return lambda x: model.predict(x)
        else:
            return lambda x: model.predict(x)[:, output_dimension]
    elif model_type.lower() in ["torch", "pytorch"]:
        import torch

        def predict_torch(x):
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32)
                output = model(x_tensor).numpy()
                if output_dimension is not None:
                    return output[:, output_dimension]
                return output

        return predict_torch
    elif model_type.lower() in ["tensorflow", "keras"]:

        def predict_tf(x):
            output = model.predict(x)
            if output_dimension is not None:
                return output[:, output_dimension]
            return output

        return predict_tf
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _calculate_sobol_indices(
    input_data: ArrayLike,
    model: Any,
    model_type: Optional[str] = None,
    n_samples: int = 1000,
    output_dimension: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    calc_second_order: bool = True,
) -> Dict[str, np.ndarray]:
    """
    计算 Sobol 敏感性指数

    参数:
        input_data: 输入数据用于确定特征范围
        model: 要解释的模型
        model_type: 模型类型
        n_samples: Sobol 分析的样本数量
        output_dimension: 输出维度（对于多输出模型）
        feature_names: 特征名称列表
        calc_second_order: 是否计算二阶交互

    返回:
        包含 Sobol 指数的字典 (S1, ST, S2)
    """
    if not SALIB_AVAILABLE:
        raise ImportError(
            "SALib library is required. Install with 'pip install SALib'."
        )

    # 转换为 numpy 数组
    X = ensure_numpy_array(input_data)

    # 获取特征名称
    names = handle_feature_names(input_data, feature_names)

    # 确定特征范围
    bounds = []
    for i in range(X.shape[1]):
        feature_min = np.min(X[:, i])
        feature_max = np.max(X[:, i])
        # 确保最小值和最大值不同
        if feature_min == feature_max:
            feature_min -= 0.1
            feature_max += 0.1
        bounds.append([feature_min, feature_max])

    # 创建问题定义
    problem = {"num_vars": X.shape[1], "names": names, "bounds": bounds}

    # 生成样本
    param_values = saltelli.sample(
        problem, n_samples, calc_second_order=calc_second_order
    )

    # 获取模型预测函数
    predict_func = _get_model_predict_function(model, model_type, output_dimension)

    # 计算模型输出
    Y = predict_func(param_values)

    # 计算 Sobol 指数
    Si = sobol.analyze(
        problem, Y, calc_second_order=calc_second_order, print_to_console=False
    )

    # 返回结果
    result = {
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    }

    if calc_second_order:
        result["S2"] = Si["S2"]
        result["S2_conf"] = Si["S2_conf"]

    return result


def _get_sobol_indices(
    input_data: ArrayLike,
    model: Any,
    model_type: Optional[str] = None,
    n_samples: int = 1000,
    output_dimension: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    calc_second_order: bool = True,
    use_cache: bool = True,
) -> Dict[str, np.ndarray]:
    """
    获取或计算 Sobol 敏感性指数

    参数:
        input_data: 输入数据用于确定特征范围
        model: 要解释的模型
        model_type: 模型类型
        n_samples: Sobol 分析的样本数量
        output_dimension: 输出维度（对于多输出模型）
        feature_names: 特征名称列表
        calc_second_order: 是否计算二阶交互
        use_cache: 是否使用缓存

    返回:
        包含 Sobol 指数的字典 (S1, ST, S2)
    """
    if not SALIB_AVAILABLE:
        raise ImportError(
            "SALib library is required. Install with 'pip install SALib'."
        )

    if not use_cache:
        return _calculate_sobol_indices(
            input_data,
            model,
            model_type,
            n_samples,
            output_dimension,
            feature_names,
            calc_second_order,
        )

    # 创建缓存键
    cache_key = _create_cache_key(
        model, input_data, model_type, n_samples, output_dimension
    )

    # 检查缓存
    sobol_indices = sobol_indices_cache.get(cache_key)
    if sobol_indices is not None:
        return sobol_indices

    # 计算新的 Sobol 指数并缓存
    sobol_indices = _calculate_sobol_indices(
        input_data,
        model,
        model_type,
        n_samples,
        output_dimension,
        feature_names,
        calc_second_order,
    )
    sobol_indices_cache.set(cache_key, sobol_indices)
    return sobol_indices


def _rank_features_by_importance(
    indices: np.ndarray,
    feature_names: List[str],
    conf_intervals: Optional[np.ndarray] = None,
    normalized: bool = True,
) -> pd.DataFrame:
    """
    根据重要性对特征进行排序

    参数:
        indices: Sobol 指数
        feature_names: 特征名称
        conf_intervals: 置信区间
        normalized: 是否归一化

    返回:
        排序后的 DataFrame
    """
    # 创建 DataFrame
    df = pd.DataFrame({"Feature": feature_names, "Importance": indices})

    if conf_intervals is not None:
        df["Confidence"] = conf_intervals

    # 归一化
    if normalized:
        max_importance = df["Importance"].max()
        if max_importance > 0:
            df["Importance"] = df["Importance"] / max_importance

    # 排序
    df = df.sort_values("Importance", ascending=False)

    return df


def _get_top_interactions(
    S2: np.ndarray, feature_names: List[str], top_n: int = 10
) -> pd.DataFrame:
    """
    获取顶部交互

    参数:
        S2: 二阶 Sobol 指数
        feature_names: 特征名称
        top_n: 返回的顶部交互数量

    返回:
        排序后的 DataFrame
    """
    n_features = len(feature_names)
    interactions = []

    # 收集所有交互
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append(
                {
                    "Feature1": feature_names[i],
                    "Feature2": feature_names[j],
                    "Interaction": S2[i, j],
                }
            )

    # 创建 DataFrame 并排序
    df = pd.DataFrame(interactions)
    df = df.sort_values("Interaction", ascending=False)

    # 返回顶部交互
    return df.head(top_n)


def get_sobol_indices(
    input: ArrayLike,
    model: Any,
    model_type: Optional[str] = None,
    n_samples: int = 1000,
    output_dimension: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    calc_second_order: bool = True,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    计算并返回 Sobol 敏感性指数

    参数:
        input: 输入数据用于确定特征范围
        model: 要解释的模型
        model_type: 模型类型
        n_samples: Sobol 分析的样本数量
        output_dimension: 输出维度（对于多输出模型）
        feature_names: 特征名称列表
        calc_second_order: 是否计算二阶交互
        use_cache: 是否使用缓存

    返回:
        包含 Sobol 指数和相关信息的字典
    """
    if not SALIB_AVAILABLE:
        raise ImportError(
            "SALib library is required. Install with 'pip install SALib'."
        )

    # 获取 Sobol 指数
    sobol_indices = _get_sobol_indices(
        input,
        model,
        model_type,
        n_samples,
        output_dimension,
        feature_names,
        calc_second_order,
        use_cache,
    )

    # 获取特征名称
    names = handle_feature_names(input, feature_names)

    # 创建结果字典
    result = {
        "S1": sobol_indices["S1"],
        "S1_conf": sobol_indices["S1_conf"],
        "ST": sobol_indices["ST"],
        "ST_conf": sobol_indices["ST_conf"],
        "feature_names": names,
    }

    if calc_second_order and "S2" in sobol_indices:
        result["S2"] = sobol_indices["S2"]
        result["S2_conf"] = sobol_indices["S2_conf"]

        # 添加排序后的交互
        top_interactions = _get_top_interactions(sobol_indices["S2"], names)
        result["top_interactions"] = top_interactions

    # 添加排序后的特征重要性
    result["S1_ranked"] = _rank_features_by_importance(
        sobol_indices["S1"], names, sobol_indices["S1_conf"], normalized=True
    )

    result["ST_ranked"] = _rank_features_by_importance(
        sobol_indices["ST"], names, sobol_indices["ST_conf"], normalized=True
    )

    return result


def plot_sobol_indices(
    input: ArrayLike,
    model: Any,
    model_type: Optional[str] = None,
    n_samples: int = 1000,
    indices_type: Literal["S1", "ST"] = "ST",
    output_dimension: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    sort_features: bool = True,
    show: bool = True,
    use_cache: bool = True,
) -> Figure:
    """
    绘制 Sobol 指数条形图

    参数:
        input: 输入数据用于确定特征范围
        model: 要解释的模型
        model_type: 模型类型
        n_samples: Sobol 分析的样本数量
        indices_type: 指数类型 ('S1', 'ST')
        output_dimension: 输出维度（对于多输出模型）
        feature_names: 特征名称列表
        title: 图表标题
        figsize: 图表大小
        sort_features: 是否按重要性排序特征
        show: 是否立即显示图表
        use_cache: 是否使用缓存

    返回:
        matplotlib Figure 对象
    """
    if not SALIB_AVAILABLE:
        raise ImportError(
            "SALib library is required. Install with 'pip install SALib'."
        )

    # 获取 Sobol 指数
    sobol_indices = _get_sobol_indices(
        input,
        model,
        model_type,
        n_samples,
        output_dimension,
        feature_names,
        True,
        use_cache,
    )

    # 获取特征名称
    names = handle_feature_names(input, feature_names)

    # 选择指数类型
    if indices_type == "S1":
        indices = sobol_indices["S1"]
        conf = sobol_indices["S1_conf"]
        indices_name = "First-order"
    else:  # 默认为总效应
        indices = sobol_indices["ST"]
        conf = sobol_indices["ST_conf"]
        indices_name = "Total-effect"

    # 排序特征
    if sort_features:
        importance_df = _rank_features_by_importance(
            indices, names, conf, normalized=False
        )
        sorted_names = importance_df["Feature"].tolist()
        sorted_indices = importance_df["Importance"].to_numpy()
        sorted_conf = (
            importance_df["Confidence"].to_numpy()
            if "Confidence" in importance_df.columns
            else None
        )
    else:
        sorted_names = names
        sorted_indices = indices
        sorted_conf = conf

    # 创建图表
    fig, ax = create_figure(
        figsize=figsize,
        title=title or f"{indices_name} Sobol Sensitivity Indices",
        xlabel="Sensitivity Index",
        ylabel="Feature",
    )

    # 绘制水平条形图
    ax.barh(sorted_names, sorted_indices, xerr=sorted_conf, capsize=5)

    # 反转 y 轴，使最重要的特征在顶部
    if sort_features:
        ax.invert_yaxis()

    # 添加网格线
    ax.grid(True, linestyle="--", alpha=0.7)

    # 设置 x 轴范围
    ax.set_xlim(0, max(1.0, max(sorted_indices) * 1.1))

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_sobol_radar(
    input: ArrayLike,
    model: Any,
    model_type: Optional[str] = None,
    n_samples: int = 1000,
    output_dimension: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    include_first_order: bool = True,
    show: bool = True,
    use_cache: bool = True,
) -> Figure:
    """
    绘制 Sobol 雷达图

    参数:
        input: 输入数据用于确定特征范围
        model: 要解释的模型
        model_type: 模型类型
        n_samples: Sobol 分析的样本数量
        output_dimension: 输出维度（对于多输出模型）
        feature_names: 特征名称列表
        title: 图表标题
        figsize: 图表大小
        include_first_order: 是否包含一阶指数
        show: 是否立即显示图表
        use_cache: 是否使用缓存

    返回:
        matplotlib Figure 对象
    """
    if not SALIB_AVAILABLE:
        raise ImportError(
            "SALib library is required. Install with 'pip install SALib'."
        )

    # 获取 Sobol 指数
    sobol_indices = _get_sobol_indices(
        input,
        model,
        model_type,
        n_samples,
        output_dimension,
        feature_names,
        True,
        use_cache,
    )

    # 获取特征名称
    names = handle_feature_names(input, feature_names)

    # 获取一阶和总效应指数
    S1 = sobol_indices["S1"]
    ST = sobol_indices["ST"]

    # 创建图表
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # 准备数据
    ST_values = np.append(ST, ST[0])  # 闭合雷达图

    if include_first_order:
        S1_values = np.append(S1, S1[0])  # 闭合雷达图

    # 绘制雷达图
    ax.plot(angles, ST_values, "o-", linewidth=2, label="Total-effect")
    ax.fill(angles, ST_values, alpha=0.25)

    if include_first_order:
        ax.plot(angles, S1_values, "o-", linewidth=2, label="First-order")
        ax.fill(angles, S1_values, alpha=0.25)

    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names)

    # 设置 y 轴范围
    ax.set_ylim(0, max(1.0, max(ST) * 1.1))

    # 添加图例
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    # 设置标题
    if title:
        ax.set_title(title, y=1.08)
    else:
        ax.set_title("Sobol Sensitivity Indices", y=1.08)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_sobol_interactions(
    input: ArrayLike,
    model: Any,
    model_type: Optional[str] = None,
    n_samples: int = 1000,
    output_dimension: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    top_n: int = 10,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    show: bool = True,
    use_cache: bool = True,
) -> Figure:
    """
    绘制 Sobol 交互热图

    参数:
        input: 输入数据用于确定特征范围
        model: 要解释的模型
        model_type: 模型类型
        n_samples: Sobol 分析的样本数量
        output_dimension: 输出维度（对于多输出模型）
        feature_names: 特征名称列表
        top_n: 显示的顶部交互数量
        title: 图表标题
        figsize: 图表大小
        show: 是否立即显示图表
        use_cache: 是否使用缓存

    返回:
        matplotlib Figure 对象
    """
    if not SALIB_AVAILABLE:
        raise ImportError(
            "SALib library is required. Install with 'pip install SALib'."
        )

    # 获取 Sobol 指数
    sobol_indices = _get_sobol_indices(
        input,
        model,
        model_type,
        n_samples,
        output_dimension,
        feature_names,
        True,
        use_cache,
    )

    # 获取特征名称
    names = handle_feature_names(input, feature_names)

    # 检查是否计算了二阶交互
    if "S2" not in sobol_indices:
        raise ValueError(
            "Second-order indices not available. Set calc_second_order=True."
        )

    # 获取二阶交互指数
    S2 = sobol_indices["S2"]

    # 创建交互矩阵
    n_features = len(names)
    interaction_matrix = np.zeros((n_features, n_features))

    # 填充交互矩阵
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                interaction_matrix[i, j] = sobol_indices["S1"][i]
            else:
                interaction_matrix[i, j] = S2[min(i, j), max(i, j)]

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热图
    im = ax.imshow(interaction_matrix, cmap="viridis")

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sensitivity Index")

    # 设置刻度标签
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    # 旋转 x 轴标签以避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个单元格中添加数值
    for i in range(n_features):
        for j in range(n_features):
            text_color = "white" if interaction_matrix[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{interaction_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color=text_color,
            )

    # 设置标题
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Sobol Interaction Indices")

    # 获取顶部交互
    top_interactions = _get_top_interactions(S2, names, top_n)

    # 在图表下方添加顶部交互表格
    if len(top_interactions) > 0:
        table_data = []
        for _, row in top_interactions.iterrows():
            table_data.append(
                [f"{row['Feature1']} × {row['Feature2']}", f"{row['Interaction']:.4f}"]
            )

        # 创建表格
        table_ax = fig.add_axes((0.15, 0.01, 0.7, min(0.2, 0.02 * len(table_data))))
        table = table_ax.table(
            cellText=table_data,
            colLabels=["Interaction", "Index"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        table_ax.axis("off")

    plt.tight_layout()

    if show:
        plt.show()

    return fig

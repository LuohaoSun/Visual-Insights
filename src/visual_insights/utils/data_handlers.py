"""
数据处理工具

提供数据预处理、转换和验证功能。
"""

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# 类型别名
ArrayLike = NDArray | pd.DataFrame | pd.Series
FeatureNames = list[str]
TargetNames = list[str]


def ensure_numpy_array(data: ArrayLike) -> NDArray[Any]:
    """
    将输入数据转换为 numpy 数组

    参数:
        data: 输入数据，可以是 numpy 数组、pandas DataFrame 或 Series

    返回:
        numpy 数组
    """
    if isinstance(data, pd.DataFrame | pd.Series):
        return data.to_numpy()
    return np.asarray(data)


def handle_feature_names(
    data: ArrayLike,
    feature_names: Sequence[str] | None = None,
    prefix: str = "Feature",
) -> FeatureNames:
    """
    提取或生成特征名称

    参数:
        data: 输入数据
        feature_names: 显式指定的特征名称（可选）
        prefix: 自动生成的特征名称前缀

    返回:
        特征名称列表
    """
    if isinstance(data, pd.DataFrame) and feature_names is None:
        return list(data.columns)
    elif feature_names is not None:
        return list(feature_names)
    else:
        # 生成默认特征名称
        if hasattr(data, "shape") and len(data.shape) > 1:
            return [f"{prefix}{i}" for i in range(data.shape[1])]
        return [f"{prefix}0"]


def handle_target_names(
    data: ArrayLike,
    target_names: Sequence[str] | None = None,
    prefix: str = "Target",
) -> TargetNames:
    """
    提取或生成目标变量名称

    参数:
        data: 输入数据
        target_names: 显式指定的目标变量名称（可选）
        prefix: 自动生成的目标变量名称前缀

    返回:
        目标变量名称列表
    """
    if isinstance(data, pd.DataFrame) and target_names is None:
        return list(data.columns)
    elif isinstance(data, pd.Series) and data.name and target_names is None:
        return [str(data.name)]
    elif target_names is not None:
        return list(target_names)
    else:
        # 生成默认目标变量名称
        if (
            hasattr(data, "shape")
            and len(cast(Sequence[int], data.shape)) > 1
            and cast(tuple[int, ...], data.shape)[1] > 1
        ):
            return [f"{prefix}{i}" for i in range(cast(tuple[int, ...], data.shape)[1])]
        return [f"{prefix}"]


def extract_dimension(data: ArrayLike, dimension: int | None = None) -> NDArray[Any]:
    """
    从多维数据中提取特定维度

    参数:
        data: 输入数据
        dimension: 要提取的维度（可选）

    返回:
        提取的数据，作为 numpy 数组

    异常:
        ValueError: 如果维度无效
    """
    data_array = ensure_numpy_array(data)

    if dimension is None or data_array.ndim == 1:
        return data_array

    if data_array.ndim > 1 and dimension < data_array.shape[1]:
        return data_array[:, dimension]

    raise ValueError(
        f"Invalid dimension {dimension} for data with shape {data_array.shape}"
    )


def prepare_data_pair(
    input_data: ArrayLike,
    target_data: ArrayLike,
    output_dimension: int | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    准备一对输入和输出数据用于分析

    参数:
        input_data: 输入特征
        target_data: 目标值
        output_dimension: 要提取的特定输出维度（可选）

    返回:
        (input_array, target_array) 元组，均为 numpy 数组

    异常:
        ValueError: 如果数据形状不兼容
    """
    input_array = ensure_numpy_array(input_data)
    target_array = ensure_numpy_array(target_data)

    # 处理多维输出
    if output_dimension is not None and target_array.ndim > 1:
        target_array = extract_dimension(target_array, output_dimension)

    # 确保 target 是一维的，用于某些分析
    if target_array.ndim > 1 and target_array.shape[1] == 1:
        target_array = target_array.flatten()

    # 检查形状兼容性
    if input_array.shape[0] != target_array.shape[0]:
        raise ValueError(
            f"Input and target have incompatible shapes: \
                {input_array.shape} vs {target_array.shape}"
        )

    return input_array, target_array

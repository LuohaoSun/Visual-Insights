from typing import Dict

import numpy as np

from .utils.data_handlers import ArrayLike, ensure_numpy_array


def calculate_regression_metrics(
    input: ArrayLike, target: ArrayLike
) -> Dict[str, float]:
    """
    计算预测值与真实值之间的误差

    参数:
        input: 预测值 of shape (n_samples, n_features) or (n_samples,)
        target: 真实值 of shape (n_samples, n_features) or (n_samples,)
        如果需要计算指定维度的误差，请自行进行索引

    返回:
        误差字典, 包含 "MAE", "MSE", "RMSE", "MAPE", "R2", "ME"
    """
    # 转换为 numpy 数组
    input = ensure_numpy_array(input)
    target = ensure_numpy_array(target)

    # 计算误差

    mae = np.mean(np.abs(input - target))
    mse = np.mean((input - target) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((input - target) / target)) * 100
    r2 = 1 - np.sum((target - input) ** 2) / np.sum((target - np.mean(target)) ** 2)
    me = np.mean(input - target)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "ME": float(me),
    }

# 1. 从 typing 导入 Literal
from typing import Literal, overload

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

from .typing import ArrayLike


@overload
def parse_1darray(
    data: ArrayLike, parse_data_name: Literal[True]
) -> tuple[ndarray, str | None]: ...


@overload
def parse_1darray(
    data: ArrayLike, parse_data_name: Literal[False] = False
) -> ndarray: ...


@overload
def parse_1darray(data: ArrayLike) -> ndarray: ...


def parse_1darray(
    data: ArrayLike, parse_data_name: bool = False
) -> tuple[ndarray, str | None] | ndarray:
    if isinstance(data, DataFrame):
        assert len(data.columns) == 1, "DataFrame must have exactly one column"
        ret, name = data.iloc[:, 0].to_numpy(), str(data.columns[0])
    elif isinstance(data, Series):
        ret, name = data.to_numpy(), str(data.name)
    elif isinstance(data, ndarray):
        if data.ndim == 1:
            ret, name = data, None
        elif data.ndim == 2 and data.shape[1] == 1:
            ret, name = data.squeeze(axis=1), None
        else:
            raise ValueError("ndarray must be 1D or 2D with one column")
    else:
        ret, name = np.asarray(data), None

    return (ret, name) if parse_data_name else ret

"""extended functions, especially for pandas Series
"""
from typing import Tuple, Union

from pandas import DataFrame, Index, Series

from .classes import VT


def transform_pandas(
    vtype: VT,
    data: Union[DataFrame, Series, float],
    weights: Union[Index, Series, Tuple[Union[Index, Series]]],
    dim_in: Union[Index, Series] = None,
    dim_out: Union[Index, Series] = None,
    threshold: float = 0.0,
    validate: bool = True,
) -> Union[DataFrame, Series, float]:
    pass

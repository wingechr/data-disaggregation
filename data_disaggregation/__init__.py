__version__ = "0.8.0"

from .base import apply_map
from .dataframe import apply_map_df
from .vartype import VarTypeCategorical, VarTypeMetric, VarTypeMetricExt, VarTypeOrdinal

__all__ = [
    "VarTypeCategorical",
    "VarTypeOrdinal",
    "VarTypeMetric",
    "VarTypeMetricExt",
    "apply_map",
    "apply_map_df",
]

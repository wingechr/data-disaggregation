__version__ = "0.8.2"

from .base import (
    VarTypeCategorical,
    VarTypeMetric,
    VarTypeMetricExt,
    VarTypeOrdinal,
    apply_map,
    apply_map_df,
)

__all__ = [
    "VarTypeCategorical",
    "VarTypeOrdinal",
    "VarTypeMetric",
    "VarTypeMetricExt",
    "apply_map",
    "apply_map_df",
]

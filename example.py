import pandas as pd

from data_disaggregation import VT_NumericExt

ds = pd.Series([10]).rename_axis(["year"])
ds_map = pd.Series({(0, 0): 1, (0, 1): 3}).rename_axis(["year", "month"])


var = {"A": 100}
vmap = {("A", "B"): 1, ("A", "C"): 4}
res = VT_NumericExt.disagg(var, vmap)
print(res)


var = pd.Series({"A": 100})
vmap = {(0, "B"): 1, (0, "C"): 4}
res = VT_NumericExt.disagg(var, vmap)
print(res)


var = [100]
# var = dict(enumerate(var))
vmap = {(0, "B"): 1, (0, "C"): 4}
res = VT_NumericExt.disagg(var, vmap)
print(res)

var = 100
# var = {None: var}
vmap = {(None, "B"): 1, (None, "C"): 4}
res = VT_NumericExt.disagg(var, vmap)
print(res)

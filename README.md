# Data (dis)aggregation

## Install

```bash
pip install data-disaggregation
```

## Example usage (with pandas)

Conceptually, aggregation/disaggregation operations are

- start with _indexed data_ (index can be multidimensional)
- use a _weight map_ to map data to a new (multidimensional) index.
- group values for each unique in the new index and use a _weighted aggregation_,
- which depends on the _variable type_, e.g.nominal, ordinal, numerical (intensive, extensive)

```python
## Examples Using pandas

# although not required, we use pandas Series for data and weights using named Index/MultiIndex
# We start by setting up some dimensions (2 spatial, 1 temporal) using named Index

from pandas import Series, Index, MultiIndex
from data_disaggregation import VT_Numeric, VT_NumericExt, disagg

dim_region = Index(["r1", "r2"], name="region")
dim_subregion = Index(["r11", "r12", "r21", "r22"], name="subregion")
dim_time = Index(["t1", "t2", "t3"], name="time")

# We can use MultiIndex to create cross products:

dim_region_subregion = MultiIndex.from_product([dim_region, dim_subregion])
dim_region_time = MultiIndex.from_product([dim_region, dim_time])
dim_region_time
```

    MultiIndex([('r1', 't1'),
                ('r1', 't2'),
                ('r1', 't3'),
                ('r2', 't1'),
                ('r2', 't2'),
                ('r2', 't3')],
               names=['region', 'time'])

```python
# now we create Series for data and weights (which also includes relationships between dimensions)
# using a value of 1 here because all the subregions have the same weight relatively
w_region_subregion = Series({("r1", "r11"): 1, ("r1", "r12"): 1, ("r2", "r21"): 1, ("r2", "r22"): 1}, index=dim_region_subregion)

# define some data on the regional level
d_region = Series({"r1": 100, "r2": 200}, index=dim_region)

# use extensive disaggregation:
d_subregion = disagg(VT_NumericExt, d_region, weights=w_region_subregion)
d_subregion
```

    subregion
    r11     50
    r12     50
    r21    100
    r22    100
    dtype: int64

```python
# applying the same weight map aggregates it back.
disagg(VT_NumericExt, d_subregion, weights=w_region_subregion)
```

    region
    r1    100
    r2    200
    dtype: int64

```python
# using Intensive distribution, the values for the regions in the disaggregation are duplicated
disagg(VT_Numeric, d_subregion, weights=w_region_subregion)
```

    region
    r1     50
    r2    100
    dtype: int64

```python
# distribute over a new dimension (time)
w_time = Series({"t1": 2, "t2": 3, "t3": 5}, index=dim_time)
disagg(VT_NumericExt, d_region, weights=w_time, dim_out=dim_region_time)
```

    region  time
    r1      t1       20
            t2       30
            t3       50
    r2      t1       40
            t2       60
            t3      100
    dtype: int64

```python
# what about scalar
s_time = disagg(VT_NumericExt, 100, weights=w_time)
s_time
```

    time
    t1    20
    t2    30
    t3    50
    dtype: int64

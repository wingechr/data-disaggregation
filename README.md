# Data (dis)aggregation

[![docs](https://img.shields.io/readthedocs/data-disaggregation?logo=readthedocs&logoColor=white)](https://data-disaggregation.readthedocs.io)
[![tests](https://github.com/wingechr/data-disaggregation/actions/workflows/unittest.yml/badge.svg)](https://github.com/wingechr/data-disaggregation/actions/workflows/unittest.yml)
[![python](https://img.shields.io/pypi/pyversions/data-disaggregation?logo=python&logoColor=white)](https://pypi.org/project/data-disaggregation)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/github/license/wingechr/data-disaggregation)](https://github.com/wingechr/data-disaggregation/blob/main/LICENSE)
[![release](https://img.shields.io/github/v/release/wingechr/data-disaggregation?include_prereleases)](https://github.com/wingechr/data-disaggregation/releases)
[![pypi](https://img.shields.io/pypi/v/data-disaggregation.svg)](https://pypi.org/project/data-disaggregation)

## Install

```bash
pip install data-disaggregation
```

## Quickstart

```python
from data_disaggregation import Dimension, Variable

# create dimension hierarchies
time = Dimension("time")
hour = time.add_level("hour", [1, 2, 3])
space = Dimension("space")
region = space.add_level("region", ["r1", "r2"])
subregion = region.add_level(
    "subregion", {"r1": ["sr1_1", "sr1_2"], "r2": ["sr2_1"]}
)

# create extensive variable
v1 = Variable(
    name="v1",
    data={
        (1, "sr1_1"): 2,
        (1, "sr1_2"): 3,
        (2, "sr1_2"): 4,
        (2, "sr2_1"): 5,
    },
    domain=[hour, subregion],
    vartype="extensive",
)

# transform (aggregate) fo target dimension
v2 = v1.transform(domain=[region])
# print as pandas series
print(v2.to_series())


```

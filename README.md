## Install

```bash
pip install data-disaggregation
```

## Quickstart

```python
from data_disaggregation import Dimension, Variable

time = Dimension("time")
hours = time.add_level("hours", [1, 2, 3])


```

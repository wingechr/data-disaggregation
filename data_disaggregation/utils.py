import pandas as pd


def as_items(x):
    if isinstance(x, (dict, pd.Series)):
        return x.items()
    return x


def as_index(x):
    if isinstance(x, pd.Series):
        return x.index
    return x


def as_multi_index(x):
    if isinstance(x, pd.Series):
        return pd.Series(x.values, index=as_multi_index(x.index))
    elif isinstance(x, pd.MultiIndex):
        return x
    elif isinstance(x, pd.Index):
        return pd.MultiIndex.from_product([x])
    return x


def as_single_index(x):
    if isinstance(x, pd.Series):
        return pd.Series(x.values, index=as_single_index(x.index))
    elif isinstance(x, pd.MultiIndex):
        if len(x.names) != 1:
            raise Exception("number of levels must be 1")
        return pd.Index([i[0] for i in x], name=x.names[0])
    return x


def group_sum(key_vals, get_key=None):
    """simple group sum

    Args:
        key_vals (list): non empty list of (key, value) pairs
          * keys can be anything hashable,
          * values must be numerical

    Returns
        list of (unique key, sum of values) pairs
    """

    key_vals = as_items(key_vals)

    res = {}
    if not get_key:
        for k, v in key_vals:
            res[k] = res.get(k, 0) + v
    else:
        # custom get key
        for k, v in key_vals:
            k = get_key(k)
            res[k] = res.get(k, 0) + v

    return res


def weighted_sum(value_normweights):
    """get sum product

    Args:
        value_normweights (list): non empty list of (value, weight) pairs
          * values must be numerical
          * weights must be numerical, positive, and sum up to 1.0

    Returns
        key from original list with highest weight
    """
    # TODO faster methods with numpy or ?
    return sum(v * w for v, w in value_normweights)


def weighted_mode(value_normweights):
    """get most common value (but by weight)

    Args:
        value_normweights (list): non empty list of (value, weight) pairs
          * values can be anything sortable
          * weights must be numerical, positive, and sum up to 1.0

    Returns
        key from original list with highest weight
    """
    # make values unique (sum weights)
    value_normweights = group_sum(value_normweights).items()
    # first element of item with highest value
    return sorted(value_normweights, key=lambda vw: vw[1], reverse=True)[0][0]


def weighted_percentile(value_normweights, p=0.5):
    """get most median (but by weight)

    Args:
        value_normweights (list): non empty list of (value, weight) pairs
          * values can be anything sortable
          * weights must be numerical, positive, and sum up to 1.0

    Returns
        key from original list with highest weight
    """
    # make values unique (sum weights)
    value_normweights = group_sum(value_normweights).items()
    # get cumulative values
    wsum = 0
    for v, w in sorted(value_normweights, key=lambda vw: vw[0]):
        wsum += w
        if wsum >= p:
            return v
    raise ValueError()


def weighted_median(value_normweights):
    """get most median (but by weight)

    Args:
        value_normweights (list): non empty list of (value, weight) pairs
          * values can be anything sortable
          * weights must be numerical, positive, and sum up to 1.0

    Returns
        key from original list with highest weight
    """
    return weighted_percentile(value_normweights, p=0.5)


def is_na(x):
    return x is None


def get_levels_dict(idx):
    if isinstance(idx, pd.MultiIndex):
        return dict(zip(idx.names, idx.levels))
    else:
        return dict({idx.name: idx})

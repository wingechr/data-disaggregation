def group_sum(key_vals):
    """simple group sum

    Args:
        key_vals (list): non empty list of (key, value) pairs
          * keys can be anything hashable,
          * values must be numerical

    Returns
        list of (unique key, sum of values) pairs
    """

    res = {}
    for k, v in key_vals:
        res[k] = res.get(k, 0) + v
    return list(res.items())


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
    value_normweights = group_sum(value_normweights)
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
    value_normweights = group_sum(value_normweights)
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

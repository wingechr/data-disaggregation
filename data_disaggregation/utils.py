import logging
from collections import OrderedDict

from pandas import DataFrame, Index, MultiIndex, Series


def check_index_names(names: list):
    assert all(names), "missing names: %s" % str(names)
    assert len(names) == len(set(names))


def norm_map(mp: Series, dim_norm: str = None) -> Series:
    """

    Args:
        mp (Series): _description_
        dim_norm (str): _description_

    Returns:
        Series: _description_


    Example:

    >>> d = {'a': ["a1", "a1", "a2"], 'b': ["b1", "b2", "b2"], 'v': [1, 3, 2]}
    >>> mp = DataFrame(d).set_index(['a', 'b'])['v']
    >>> norm_map(mp, 'a').to_dict()
    {('a1', 'b1'): 0.25, ('a1', 'b2'): 0.75, ('a2', 'b2'): 1.0}
    >>> norm_map(mp, 'b').to_dict()
    {('a1', 'b1'): 1.0, ('a1', 'b2'): 0.6, ('a2', 'b2'): 0.4}

    >>> mp = Series({"a1": 1, "a2": 3}).rename_axis(index='a')
    >>> norm_map(mp, 'a').to_dict()
    {'a1': 1.0, 'a2': 1.0}
    >>> norm_map(mp, None).to_dict()
    {'a1': 0.25, 'a2': 0.75}

    """
    check_index_names(mp.index.names)

    if dim_norm in mp.index.names:
        # NOTE: if dim_norm is only dim, result is all 1
        # sum over dimension
        msum = mp.groupby(by=dim_norm).sum().to_frame("sum")
        assert (msum["sum"] > 0).all()
        # must convert series frame so we can join (repeat) sums
        nmap = mp.to_frame("map").join(msum)
        nmap = (nmap["map"] / nmap["sum"]).rename(mp.name)
    else:
        # sum over dimension
        msum = mp.sum()
        assert msum > 0
        nmap = mp / msum
    logging.debug("== norm map\n%s", nmap)
    return nmap


def ensure_multi_index(df):
    if not isinstance(df.index, MultiIndex):
        df.index = MultiIndex.from_tuples(
            tuple((x,) for x in df.index), names=(df.index.name,)
        )
    return df


def flatten_single_index(df):
    if isinstance(df.index, MultiIndex) and len(df.index.levels) == 1:
        df.index = df.index.levels[0]
    return df


def get_dims(df):
    dims = OrderedDict(zip(df.index.names, df.index.levels))
    return dims


def get_input_dims(df):
    assert df.index.is_unique
    if isinstance(df, DataFrame):
        df = ensure_multi_index(df)
        check_index_names(df.index.names)
        dims_df = get_dims(df)
    elif isinstance(df, Series):
        dims_df = None
    else:
        raise TypeError(df)
    logging.debug("=== input dimensions")
    logging.debug(dims_df)
    return df, dims_df


def get_map_dims(mp):
    assert isinstance(mp, Series)
    assert mp.index.is_unique
    mp = ensure_multi_index(mp)
    check_index_names(mp.index.names)
    assert (mp >= 0).all()
    dims_mp = get_dims(mp)
    logging.debug("=== map dimensions")
    logging.debug(dims_mp)
    return mp, dims_mp


def get_shared_new_dims(dims_df, dims_mp):
    dims_df = dims_df or OrderedDict()
    # check compatibility of dimension names
    dims_shared = list(set(dims_mp) & set(dims_df))
    dims_new = list(set(dims_mp) - set(dims_df))
    len_shared_new = len(dims_shared), len(dims_new)
    dims_result = dims_df.copy()
    if len_shared_new == (1, 1):
        # map s -> n
        dim_shared = dims_shared[0]
        dim_new = dims_new[0]
        # replace dimension in position
        idx = list(dims_result).index(dim_shared)
        dims_result = list(dims_result.items())
        dims_result[idx] = (dim_new, dims_mp[dim_new])
        dims_result = OrderedDict(dims_result)
    elif len_shared_new == (1, 0):
        # drop s
        dim_shared = dims_shared[0]
        dim_new = None
        del dims_result[dim_shared]
    elif len_shared_new == (0, 1):
        # add n
        dim_shared = None
        dim_new = dims_new[0]
        dims_result[dim_new] = dims_mp[dim_new]
    else:
        raise TypeError(map)
    if dim_shared:
        assert set(dims_mp[dim_shared]) >= set(dims_df[dim_shared])
    dims_tmp = dims_df.copy()
    if dim_new:
        dims_tmp[dim_new] = dims_mp[dim_new]
    idx_tmp = MultiIndex.from_product(dims_tmp.values(), names=dims_tmp.keys())
    check_index_names(idx_tmp.names)
    if dims_result:
        idx_result = MultiIndex.from_product(
            dims_result.values(), names=dims_result.keys()
        )
        check_index_names(idx_result.names)
    else:
        idx_result = None
    logging.debug(f"dim_shared: {dim_shared}, dim_new: {dim_new}")
    return dim_shared, dim_new, idx_result, idx_tmp


def expand_df(idx, df=None):
    result = DataFrame(index=idx)
    if df is not None:
        result = result.join(df)
    # make sure order is correct after join
    result = result.reorder_levels(idx.names)
    return result


def expand_s(idx, s):
    result = DataFrame(index=idx).join(s.to_frame("_name"))["_name"]
    result = result.reorder_levels(idx.names)
    return result


def make_dim(items, name):
    check_index_names(items)
    idx = Index(items, name)
    return idx


def make_domain(indices, name):
    for index in indices:
        check_index_names(index)
    idx = MultiIndex.from_product(indices, names=indices.names)
    check_index_names(idx.names)
    return idx


def transform(
    df: DataFrame | Series, mp: Series, intensive=False
) -> DataFrame | Series:
    """map data to new index

    Args:
        df (DataFrame | float): data
        mp (Series): index mapper
        intensive (bool, optional): normalize target dimension

    Returns:
        DataFrame | float

    res = transform(df, map, intensive=True)
    res2b = transform(res, map, intensive=True)

    res2 = transform(res, map2, intensive=True)

    res3 = transform(res2, map3, intensive=True)
    res4 = transform(res3, map, intensive=True)


    Example (distribute number):

    >>> df1 = Series({"s": 12, "x": 16})
    >>> mp_a = Series({'a1': 1, 'a2': 3}).rename_axis(index='a')
    >>> mp_b = Series({'b1': 1, 'b2': 3}).rename_axis(index='b')
    >>> d = {"c": ["c1", "c1", "c2"], "b": ["b1", "b2", "b2"], "m": 1}
    >>> mp_bc = DataFrame(d).set_index(["c", "b"])["m"]
    >>> df2 = transform(df1, mp_a)
    >>> df2.to_dict('index')
    {'a1': {'s': 3.0, 'x': 4.0}, 'a2': {'s': 9.0, 'x': 12.0}}
    >>> s3 = transform(df2, mp_a)
    >>> s3.to_dict()
    {'s': 12.0, 'x': 16.0}
    >>> df4 = transform(df2, mp_b)
    >>> df4.to_dict('index')
    {\
('a1', 'b1'): {'s': 0.75, 'x': 1.0}, \
('a1', 'b2'): {'s': 2.25, 'x': 3.0}, \
('a2', 'b1'): {'s': 2.25, 'x': 3.0}, \
('a2', 'b2'): {'s': 6.75, 'x': 9.0}}
    >>> df5 = transform(df4, mp_bc)
    >>> df5.to_dict('index')
    {\
('a1', 'c1'): {'s': 1.875, 'x': 2.5}, \
('a1', 'c2'): {'s': 1.125, 'x': 1.5}, \
('a2', 'c1'): {'s': 5.625, 'x': 7.5}, \
('a2', 'c2'): {'s': 3.375, 'x': 4.5}}
    >>> df6 = transform(df5, mp_a)
    >>> df6.to_dict('index')
    {'c1': {'s': 7.5, 'x': 10.0}, 'c2': {'s': 4.5, 'x': 6.0}}

    >>> df1 = Series({"s": 12, "x": 16})
    >>> mp_a = Series({'a1': 1, 'a2': 3}).rename_axis(index='a')
    >>> mp_b = Series({'b1': 1, 'b2': 3}).rename_axis(index='b')
    >>> d = {"c": ["c1", "c1", "c2"], "b": ["b1", "b2", "b2"], "m": 1}
    >>> mp_bc = DataFrame(d).set_index(["c", "b"])["m"]
    >>> df2 = transform(df1, mp_a, intensive=True)
    >>> df2.to_dict('index')
    {'a1': {'s': 12, 'x': 16}, 'a2': {'s': 12, 'x': 16}}
    >>> s3 = transform(df2, mp_a, intensive=True)
    >>> s3.to_dict()
    {'s': 12.0, 'x': 16.0}
    >>> df4 = transform(df2, mp_b, intensive=True)
    >>> df4.to_dict('index')
    {\
('a1', 'b1'): {'s': 12, 'x': 16}, \
('a1', 'b2'): {'s': 12, 'x': 16}, \
('a2', 'b1'): {'s': 12, 'x': 16}, \
('a2', 'b2'): {'s': 12, 'x': 16}}
    >>> df5 = transform(df4, mp_bc, intensive=True)
    >>> df5.to_dict('index')
    {\
('a1', 'c1'): {'s': 12.0, 'x': 16.0}, \
('a1', 'c2'): {'s': 12.0, 'x': 16.0}, \
('a2', 'c1'): {'s': 12.0, 'x': 16.0}, \
('a2', 'c2'): {'s': 12.0, 'x': 16.0}}
    >>> df6 = transform(df5, mp_a, intensive=True)
    >>> df6.to_dict('index')
    {'c1': {'s': 12.0, 'x': 16.0}, 'c2': {'s': 12.0, 'x': 16.0}}

    """

    df, dims_df = get_input_dims(df)
    mp, dims_mp = get_map_dims(mp)
    d_shared, d_new, idx_res, idx_tmp = get_shared_new_dims(dims_df, dims_mp)

    choice = (
        dims_df is not None,
        d_shared is not None,
        d_new is not None,
        idx_res is not None,
    )

    if choice == (False, False, True, True):
        logging.debug("******* expand from scalars")
        # todo: do it without the loop?
        result = expand_df(idx_tmp)
        for name, val in df.iteritems():
            result[name] = val
        if not intensive:
            mp = norm_map(mp, None)
            result = result.mul(mp, axis="index")
    elif choice == (True, False, True, True):
        logging.debug("******* expand normal")
        result = expand_df(idx_tmp, df)
        if not intensive:
            mp = norm_map(mp, None)
            mp = expand_s(idx_tmp, mp)
            result = result.mul(mp, axis="index")
    elif choice == (True, True, False, False):
        logging.debug("******* squeeze to scalar")
        result = expand_df(idx_tmp, df)
        if intensive:
            mp = norm_map(mp, None)
            result = result.mul(mp, axis="index")
        result = result.sum()
    elif choice == (True, True, False, True):
        logging.debug("******* squeeze normal")
        result = expand_df(idx_tmp, df)
        if intensive:
            mp = norm_map(mp, None)
            mp = expand_s(idx_tmp, mp)
            result = result.mul(mp, axis="index")
        result = result.groupby(idx_res.names).sum()
    elif choice == (True, True, True, True):
        logging.debug("******* normal map")
        if intensive:
            mp = norm_map(mp, d_new)
        else:
            mp = norm_map(mp, d_shared)
        mp = expand_s(idx_tmp, mp)
        result = expand_df(idx_tmp, df)
        result = result.mul(mp, axis="index")
        result = result.groupby(idx_res.names).sum()
    else:
        raise NotImplementedError()

    # final checks
    result, dims_result = get_input_dims(result)
    assert (idx_res is None) == (dims_result is None)
    if dims_result:
        assert tuple(dims_result.keys()) == tuple(idx_res.names)

    result = flatten_single_index(result)

    logging.debug("== result\n%s", result)
    return result


if __name__ == "__main__":

    logging.basicConfig(
        format="[%(asctime)s %(levelname)7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    import doctest

    doctest.testmod()

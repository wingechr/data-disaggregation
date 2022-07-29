import logging
from collections import OrderedDict

from pandas import DataFrame, Index, MultiIndex, Series

DEFAULT_VALUE_NAME = "value"


def norm_map(mp: Series, dim_norm: str = None) -> Series:
    """

    Args:
        mp (Series): _description_
        dim_norm (OrderedDict): _description_

    Returns:
        Series: normalised map for transformation

    """
    assert Index(mp.index.names).is_unique

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
    return nmap


def ensure_multi_index(df):
    if not isinstance(df.index, MultiIndex):
        logging.debug("convert to multiindex")
        df.index = MultiIndex.from_tuples(
            tuple((x,) for x in df.index), names=(df.index.name,)
        )
    return df


def flatten_single_index(df):
    if isinstance(df.index, MultiIndex) and len(df.index.levels) == 1:
        logging.debug("flatten index")
        df.index = df.index.levels[0]
    return df


def get_dims(df):
    dims = OrderedDict(zip(df.index.names, df.index.levels))
    return dims


def get_input_dims(df):

    if isinstance(df, Series):
        logging.debug("convert series to frame")
        df = df.to_frame()
        df.columns = [None]

    if isinstance(df, DataFrame):
        df = ensure_multi_index(df)
        assert df.index.is_unique
        assert Index(df.index.names).is_unique
        assert all(df.index.names), "all map index levels must be named"
        dims_df = get_dims(df)
    elif isinstance(df, (int, float)):
        logging.debug("convert number to series")
        dims_df = None
        df = Series({None: df})
    elif isinstance(df, dict):
        logging.debug("convert dict to series")
        dims_df = None
        df = Series(df)
    else:
        raise TypeError(df)

    df = df.fillna(0)

    logging.debug("input dimensions")
    logging.debug(dims_df)
    if isinstance(df, Series):
        logging.debug("input series name: %s", df.name)
    else:
        logging.debug("input frame column names: %s", df.columns)
    return df, dims_df


def get_map_dims(mp):
    if isinstance(mp, Index):
        logging.debug("convert index to series with value 1")
        mp = Series(1, index=mp)

    assert isinstance(mp, Series)
    assert mp.index.is_unique
    assert all(mp.index.names), "all map index levels must be named"
    mp = ensure_multi_index(mp)
    assert Index(mp.index.names).is_unique
    mp = mp.fillna(0)
    assert (mp >= 0).all()
    dims_mp = get_dims(mp)

    logging.debug("map dimensions")
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

    logging.debug(f"dim_shared: {dim_shared}, dim_new: {dim_new}")

    if dim_shared:
        missing_elems = (set(dims_mp[dim_shared]) >= set(dims_df[dim_shared]),)
        assert missing_elems, f"missing keys: {missing_elems}"
    dims_tmp = dims_df.copy()
    if dim_new:
        dims_tmp[dim_new] = dims_mp[dim_new]

    idx_tmp = MultiIndex.from_product(dims_tmp.values(), names=dims_tmp.keys())
    assert Index(idx_tmp.names).is_unique
    if dims_result:
        idx_result = MultiIndex.from_product(
            dims_result.values(), names=dims_result.keys()
        )
        assert Index(idx_result.names).is_unique
    else:
        idx_result = None

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


def transform(
    df: DataFrame | Series, mp: Series, intensive=False
) -> DataFrame | Series:
    """map data to new index

    Args:
        df (DataFrame | Series | dict | float): data
        mp (Series): index mapper
        intensive (bool, optional): normalize target dimension

    Returns:
        DataFrame | Series | dict | float


    """
    logging.info("==============================")

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
        logging.debug("expand from scalars")
        # todo: do it without the loop?
        result = expand_df(idx_tmp)
        for name, val in df.iteritems():
            result[name] = val
        if not intensive:
            mp = norm_map(mp, None)
            result = result.mul(mp, axis="index")
    elif choice == (True, False, True, True):
        logging.debug("expand normal")
        result = expand_df(idx_tmp, df)
        if not intensive:
            mp = norm_map(mp, None)
            mp = expand_s(idx_tmp, mp)
            result = result.mul(mp, axis="index")
    elif choice == (True, True, False, False):
        logging.debug("squeeze to scalar")
        result = expand_df(idx_tmp, df)
        if intensive:
            mp = norm_map(mp, None)
            result = result.mul(mp, axis="index")
        result = result.sum()
    elif choice == (True, True, False, True):
        logging.debug("squeeze normal")
        result = expand_df(idx_tmp, df)
        if intensive:
            mp = norm_map(mp, None)
            mp = expand_s(idx_tmp, mp)
            result = result.mul(mp, axis="index")
        result = result.groupby(idx_res.names).sum()
    elif choice == (True, True, True, True):
        logging.debug("normal map")
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

    if idx_res is not None:  # not scalar
        assert tuple(result.index.names) == tuple(idx_res.names)
        result = flatten_single_index(result)
        # if frame with single column [None] -> make into series
        if tuple(result.columns) == (None,):
            result = result[None]
            logging.debug("series name: %s", result.name)
        else:
            logging.debug("frame columns name: %s", result.columns)
    else:  # make into dict
        result = dict(result.iteritems())
        if tuple(result.keys()) == (None,):  # make into value
            result = result[None]

    return result


if __name__ == "__main__":

    logging.basicConfig(
        format="[%(asctime)s %(levelname)7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    import doctest

    doctest.testmod()

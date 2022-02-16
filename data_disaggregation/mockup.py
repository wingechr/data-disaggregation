"""
TODO: also add unit checks
"""
import math
from itertools import chain, product
from collections import OrderedDict
import numpy as np
import pandas as pd


def create_group_matrix(group_sizes):
    """
    Examples:
    >>> create_group_matrix([1, 2])
    array([[1., 0.],
           [0., 1.],
           [0., 1.]])
    """
    return create_weighted_group_matrix(
        [[1] * n for n in group_sizes], on_group_sum_ne_1="ignore"
    )


def create_weighted_group_matrix(
    grouped_weights, on_group_sum_ne_1="error", rel_tol=1e-09, abs_tol=0.0
):
    """
    Examples:
    >>> create_weighted_group_matrix([[1], [0.6, 0.4]])
    array([[1. , 0. ],
           [0. , 0.6],
           [0. , 0.4]])

    >>> create_weighted_group_matrix([[1], [6, 4]], on_group_sum_ne_1="rescale")
    array([[1. , 0. ],
           [0. , 0.6],
           [0. , 0.4]])
    """
    assert on_group_sum_ne_1 in ("error", "ignore", "rescale")

    n_rows = sum(len(gw) for gw in grouped_weights)
    n_columns = len(grouped_weights)
    matrix = np.zeros(shape=(n_rows, n_columns))
    row_i_start = 0
    for col_i, weights in enumerate(grouped_weights):
        # check weights
        sum_weights = sum(weights)
        if not math.isclose(1, sum_weights, rel_tol=rel_tol, abs_tol=abs_tol):
            if on_group_sum_ne_1 == "error":
                raise ValueError("Sum of weights != 1")
            elif on_group_sum_ne_1 == "rescale":
                if not sum_weights:
                    raise ValueError("Sum of weights == 0")
                weights = np.array(weights) / sum_weights
        n_rows = len(weights)
        row_i_end = row_i_start + n_rows
        matrix[row_i_start:row_i_end, col_i] = weights
        row_i_start = row_i_end
    return matrix


class DimensionLevel:
    def __init__(self, parent, name, grouped_elements):
        self._parent = parent
        self._name = name
        self._children = dict()

        # create/check elements
        if self.is_dimension_root:
            self._elements = tuple([None])
            self._levels = {self.name: self}
            group_sizes = [1]
        else:
            self._levels = self.dimension._levels  # only store once
            # register by name
            if self.name in self._levels:
                raise KeyError("Name already used: %s" % self.name)
            self._levels[self.name] = self
            # register as child
            self.parent._children[self.name] = self

            # elements
            if self.parent.is_dimension_root and not isinstance(grouped_elements, dict):
                grouped_elements = {None: grouped_elements}
            parent_elements = self.parent.elements
            assert set(grouped_elements.keys()) == set(parent_elements)
            grouped_elements = [grouped_elements[pe] for pe in parent_elements]
            group_sizes = [len(g) for g in grouped_elements]
            assert all(gs > 0 for gs in group_sizes)
            self._elements = tuple(chain(*grouped_elements))
            assert len(self.elements) == len(set(self.elements))
            assert len(self.elements) > 0

        # create grouping matrix
        self._group_matrix = create_group_matrix(group_sizes)

        self._size = len(self.elements)
        self._indices = tuple(range(self.size))
        self._element2index = dict(zip(self.elements, self.indices))

    def add_level(self, name, grouped_elements):
        return DimensionLevel(parent=self, name=name, grouped_elements=grouped_elements)

    def get_level(self, name):
        return self._levels[name]

    def get_child(self, name):
        return self._children[name]

    @property
    def group_matrix(self):
        return self._group_matrix.copy()

    @property
    def is_dimension_root(self):
        return self.parent is None

    @property
    def parent(self):
        return self._parent

    @property
    def dimension_name(self):
        return self.dimension.name

    @property
    def name(self):
        return self._name

    @property
    def indices(self):
        return self._indices

    @property
    def elements(self):
        return self._elements

    @property
    def size(self):
        return self._size

    @property
    def dimension(self):
        return self if self.is_dimension_root else self.parent.dimension

    @property
    def path(self):
        if self.is_dimension_root:
            return [self.name]
        else:
            return self.parent.path + [self.name]

    def __str__(self):
        return "/".join(self.path)


class Dimension(DimensionLevel):
    def __init__(self, name):
        super().__init__(parent=None, name=name, grouped_elements=None)


class Domain:
    def __init__(self, dimension_levels):
        dimension_levels = dimension_levels or []

        # dimensions are added by dimension name!
        # TODO: what if we want multiple spacial dimensions? we could use alias,
        # but then it would be harder/impossible to automatically trnasform variables
        self._dimension_levels = OrderedDict(
            (d.dimension_name, d) for d in dimension_levels
        )
        assert len(dimension_levels) == len(self.dimension_levels)

        self._dimension_name2index = dict(
            (n, i) for i, n in enumerate(self.dimension_names)
        )
        assert len(dimension_levels) == len(self._dimension_name2index)

        self._size = len(self.dimension_levels)
        self._shape = tuple(d.size for d in self.dimension_levels)
        self._keys = tuple(product(*[d.elements for d in self.dimension_levels]))
        self._indices = tuple(product(*[d.indices for d in self.dimension_levels]))
        self._key2index = dict(zip(self.keys, self.indices))

    def get_dimension_index(self, dimension_name):
        return self._dimension_name2index[dimension_name]

    def get_dimension_level(self, dimension_name):
        return self._dimension_levels[dimension_name]

    @property
    def dimension_levels(self):
        return tuple(self._dimension_levels.values())

    @property
    def dimensions(self):
        return tuple(d.dimension for d in self.dimension_levels)

    @property
    def dimension_names(self):
        return tuple(d.name for d in self.dimensions)

    @property
    def dimension_level_names(self):
        return tuple(d.name for d in self.dimension_levels)

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    @property
    def keys(self):
        return self._keys

    @property
    def indices(self):
        return self._indices

    def iter_indices_keys(self):
        yield from zip(self.indices, self.keys)

    def get_index(self, key):
        return self._key2index[key]

    def __str__(self):
        return (
            "("
            + ", ".join(
                "%s=%s(%d)" % (n, d, d.size) for n, d in self._dimension_levels.items()
            )
            + ")"
        )

    # create data matrix from dict data
    def dict_to_matrix(self, data):
        d_matrix = np.zeros(shape=self.shape)
        for key, val in data.items():
            idx = self.get_index(key)
            d_matrix[idx] = val
        return d_matrix

    # create data matrix from records data
    def records_to_matrix(self, data, value="value"):
        dimension_level_names = self.dimension_level_names
        data_dict = {}
        for rec in data:
            key = tuple(rec[n] for n in dimension_level_names)
            if key in data_dict:
                raise KeyError(key)
            data_dict[key] = rec[value]
        return self.dict_to_matrix(data_dict)


class Unit:
    def __init__(self, *args, **kwargs):
        pass


class Variable:
    def __init__(self, domain, data, unit=None, is_intensive=False):

        if isinstance(domain, Domain):
            self._domain = domain
        else:
            self._domain = Domain(domain)

        if isinstance(data, dict):
            self._data_matrix = self._domain.dict_to_matrix(data)
        elif isinstance(data, list):
            self._data_matrix = self._domain.records_to_matrix(data)
        elif isinstance(data, pd.Series):
            self._data_matrix = self._domain.dict_to_matrix(data.to_dict())
        elif isinstance(data, np.ndarray):
            self._data_matrix = data.copy()
        else:
            raise TypeError(type(data))

        assert self._domain.shape == self._data_matrix.shape

        if isinstance(unit, Unit):
            self._unit = unit
        else:
            self._unit = Unit(unit)

        assert is_intensive in (True, False, None)
        self._is_intensive = is_intensive

    @property
    def is_intensive(self):
        return self._is_intensive == True

    @property
    def is_extensive(self):
        return self._is_intensive == False

    @property
    def is_weight(self):
        # TODO: also: unit is None or 1
        return self._is_intensive == None

    @property
    def domain(self):
        return self._domain

    @property
    def unit(self):
        return self._unit

    def as_dict(self, skip_0=False):
        res = {}
        for idx, key in self.domain.iter_indices_keys():
            val = self._data_matrix[idx]
            if val or not skip_0:
                res[key] = val
        return res

    def as_records(self, skip_0=False, value="value"):
        res = []
        dimension_level_names = self.domain.dimension_level_names
        for key, val in self.as_dict(skip_0=skip_0).items():
            rec = dict(zip(dimension_level_names, key))
            rec[value] = val
            res.append(rec)

        return res

    def as_weight(self):
        if self.is_weight:
            return self

        assert self.domain.size == 1
        dimension_level = self.domain.dimension_levels[0]
        assert self._data_matrix.shape == (dimension_level.size,)

        # create sum for groups. shapes: (m, n) * (n,) = (m,)
        sums = dimension_level.group_matrix.transpose().dot(self._data_matrix)
        assert np.all(sums != 0)

        # create inverse, repeat, sum: (m,) => (1, m) => (n, m)
        # values = np.repeat(1 / sums, dimension_level.size, axis=1)
        sums = np.reshape(1 / sums, (1, sums.size))
        sums = np.repeat(sums, dimension_level.size, axis=0)

        assert sums.shape == dimension_level.group_matrix.shape
        sums = np.sum(sums * dimension_level.group_matrix, axis=1)

        assert sums.shape == self._data_matrix.shape
        weights = sums * self._data_matrix
        # is_intensive=None ==> weights
        return Variable(domain=self.domain, data=weights, unit=None, is_intensive=None)

    def aggregate(self, dimension_name, weights=None):

        if self.is_intensive and not weights:
            raise Exception("intensive aggregation without weights")
        dimension_levels = list(self.domain.dimension_levels)
        dim_idx = self.domain.get_dimension_index(dimension_name)
        dim_idx_last = len(dimension_levels) - 1
        dimension_level = dimension_levels[dim_idx]
        if dimension_level.is_dimension_root:
            raise Exception("dimension root cannot be aggregated further")

        new_dimension_level = dimension_level.parent
        group_matrix = dimension_level.group_matrix

        # print("aggregate %s => %s" % (dimension_level, new_dimension_level))

        if weights:
            if (
                len(weights.domain.dimensions) != 1
                or weights.domain.dimension_levels[0] != new_dimension_level
            ):
                raise Exception(
                    "weights domain must be %s" % Domain([new_dimension_level])
                )
            # weights is one dimensional, and number must be the same as
            # columns in group matrix
            n_rows, n_cols = group_matrix.shape
            weights_matrix = weights._data_matrix
            assert weights._data_matrix.shape == (n_cols,)

            weights_matrix = np.repeat(
                np.reshape(weights_matrix, (1, n_cols)), n_rows, axis=0
            )
            # todo check sum of groups = 1
            assert weights_matrix.shape == group_matrix.shape
            group_matrix *= weights_matrix
            # if group sums are all 1, sums in each row in group_matrix must be 1
            row_sums = np.sum(group_matrix, axis=1)
            assert np.all(np.isclose(row_sums, 1))

        dimension_levels[dim_idx] = new_dimension_level
        new_domain = Domain(dimension_levels)

        data_matrix = np.matmul(
            self._data_matrix.swapaxes(dim_idx, dim_idx_last), group_matrix
        ).swapaxes(dim_idx_last, dim_idx)

        return Variable(new_domain, data_matrix, self.unit, self.is_intensive)

    def disaggregate(self, dimension_name, dimension_level_name, weights=None):
        if self.is_extensive and not weights:
            raise Exception(
                "extensive disaggregation without weights for %s" % dimension_level_name
            )

        dimension_levels = list(self.domain.dimension_levels)
        dim_idx = self.domain.get_dimension_index(dimension_name)
        dim_idx_last = len(dimension_levels) - 1
        dimension_level = dimension_levels[dim_idx]

        try:
            new_dimension_level = dimension_level.get_child(dimension_level_name)
        except KeyError:
            raise Exception(
                "dimension level %s has no sublevel named %s"
                % (dimension_level, dimension_level_name)
            )

        dimension_levels[dim_idx] = new_dimension_level
        new_domain = Domain(dimension_levels)

        group_matrix = new_dimension_level.group_matrix.transpose()

        # print("disaggregate %s => %s" % (dimension_level, new_dimension_level))

        if weights:
            if (
                len(weights.domain.dimensions) != 1
                or weights.domain.dimension_levels[0] != new_dimension_level
            ):
                raise Exception(
                    "weights domain must be %s" % Domain([new_dimension_level])
                )
            # weights is one dimensional, and number must be the same as
            # columns in group matrix
            n_rows, n_cols = group_matrix.shape
            weights_matrix = weights._data_matrix
            assert weights._data_matrix.shape == (n_cols,)

            weights_matrix = np.repeat(
                np.reshape(weights_matrix, (1, n_cols)), n_rows, axis=0
            )
            # todo check sum of groups = 1
            assert weights_matrix.shape == group_matrix.shape
            group_matrix *= weights_matrix
            # if group sums are all 1, sums in each row in group_matrix must be 1
            row_sums = np.sum(group_matrix, axis=1)
            assert np.all(np.isclose(row_sums, 1))

        data_matrix = np.matmul(
            self._data_matrix.swapaxes(dim_idx, dim_idx_last), group_matrix
        ).swapaxes(dim_idx_last, dim_idx)

        return Variable(new_domain, data_matrix, self.unit, self.is_intensive)

    def expand(self, dimension):
        # print("adding %s" % (dimension.name,))
        assert dimension.is_dimension_root
        data_matrix = np.expand_dims(self._data_matrix, axis=self.domain.size)
        domain = Domain(list(self.domain.dimension_levels) + [dimension])
        return Variable(domain, data_matrix, self.unit, self.is_intensive)

    def squeeze(self, dimension_name):
        # print("removing %s" % (dimension_name,))

        dim_idx = self.domain.get_dimension_index(dimension_name)
        assert self.domain.dimension_levels[dim_idx].is_dimension_root
        data_matrix = np.squeeze(self._data_matrix, axis=dim_idx)
        domain = Domain(
            self.domain.dimension_levels[:dim_idx]
            + self.domain.dimension_levels[dim_idx + 1 :]
        )
        return Variable(domain, data_matrix, self.unit, self.is_intensive)

    def reorder(self, dimension_names):
        assert set(dimension_names) == set(self.domain.dimension_names)
        indices = [self.domain.get_dimension_index(n) for n in dimension_names]
        data_matrix = self._data_matrix.transpose(indices)
        domain = Domain([self.domain.dimension_levels[i] for i in indices])
        return Variable(domain, data_matrix, self.unit, self.is_intensive)

    def transform(self, target_domain, weights=None):
        """
        root_dimensions that are in current domain and NOT in target_domain will be aggregated, then dimension will be dropped.
        root_dimensions that are NOT in current domain but in target_domain will be first added at dimension level and then disaggregated
        root_dimensions that are in both will be tranformed along the transformation path: up until nearest common ancestor, then down
        """

        add_dimensions = set(target_domain.dimensions) - set(self.domain.dimensions)
        drop_dimensions = set(self.domain.dimensions) - set(target_domain.dimensions)
        change_dimensions = set(self.domain.dimensions) & set(target_domain.dimensions)

        weights = weights or {}
        for k, v in weights.items():
            weights[k] = v.as_weight()

        result = self
        for dim in drop_dimensions:
            # aggregate to root, then drop
            path = self.domain.get_dimension_level(dim.name).path
            level_names = list(reversed(path))[1:]
            for level_name in level_names:
                result = result.aggregate(dim.name, weights=weights.get(level_name))
            result = result.squeeze(dim.name)

        for dim in add_dimensions:
            # add, then disaggregate
            result = result.expand(dim)
            path = target_domain.get_dimension_level(dim.name).path
            level_names = path[1:]
            for level_name in level_names:
                result = result.disaggregate(
                    dim.name, level_name, weights=weights.get(level_name)
                )

        for dim in change_dimensions:
            # aggregate up to closest common ancestor, then down
            path_up = self.domain.get_dimension_level(dim.name).path
            path_down = target_domain.get_dimension_level(dim.name).path
            # find common part of path
            path_shared = []
            for pu, pd in zip(path_up, path_down):
                if pu != pd:
                    break
                path_shared.append(pu)
            n = len(path_shared)  # root is always shared
            path_down = path_down[n:]
            path_up = list(reversed(path_up[n:]))
            for level_name in path_up:
                result = result.aggregate(dim.name, weights=weights.get(level_name))
            for level_name in path_down:
                result = result.disaggregate(
                    dim.name, level_name, weights=weights.get(level_name)
                )

        result = result.reorder(target_domain.dimension_names)

        return result

    def __str__(self):
        return str(self._data_matrix)


class Scalar(Variable):
    def __init__(self, value, unit=None, is_intensive=False):
        super().__init__(
            domain=[], data={tuple(): value}, unit=unit, is_intensive=is_intensive
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)

    # test example

    d_t = Dimension("time")
    l_day = d_t.add_level("day", ["d1", "d2"])  # first level
    l_day_hour = l_day.add_level(
        "day_hour", {"d1": ["d1_1", "d1_2"], "d2": ["d2_1", "d2_2"]}
    )
    l_year_hour = d_t.add_level("year_hour", ["1", "2", "3", "4"])

    d_s = Dimension("space")
    l_reg = d_s.add_level("region", ["r1", "r2"])

    dom = Domain([l_reg, l_day_hour])
    dom2 = Domain([l_year_hour])

    var = Variable(
        dom,
        pd.DataFrame(
            [
                {"r": "r2", "day_hour": "d2_1", "value": 4},
                {"r": "r1", "day_hour": "d1_1", "value": 2},
                {"r": "r1", "day_hour": "d2_2", "value": 3},
            ]
        )
        .set_index(["r", "day_hour"])
        .value,
    )
    w = Variable(
        [l_year_hour],
        [{"year_hour": "1", "value": 4}, {"year_hour": "4", "value": 6}],
        None,
    )

    print(w.as_records())

    var2 = var.transform(dom2, {"year_hour": w})
    print(var2.as_records())

    sca = Scalar(100)

    var2 = sca.transform(dom2, {"year_hour": w})
    print(pd.DataFrame(var2.as_records()))

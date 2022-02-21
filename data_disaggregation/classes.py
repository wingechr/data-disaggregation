"""
TODO: also add unit checks
"""
import logging
from collections import OrderedDict
from itertools import product

import numpy as np

from .exceptions import AggregationError, DimensionStructureError, DuplicateNameError
from .functions import create_group_matrix, group

try:
    import pandas as pd
except ImportError:
    pd = None


class DimensionLevel:
    """Comparable to a pd.Index, but grouped and linked to other Levels in a tree"""

    @staticmethod
    def _parse_grouped_elements(parent, grouped_elements):
        # convert grouped_elements into dict
        if parent.is_dimension_root:  # parent is root
            if isinstance(grouped_elements, (list, tuple)):
                grouped_elements = {None: grouped_elements}
            elif pd and isinstance(grouped_elements, pd.Series):
                grouped_elements = {None: grouped_elements.values}
        else:
            if pd and isinstance(grouped_elements, pd.Series):
                grouped_elements = group(grouped_elements.iteritems())

        if not isinstance(grouped_elements, dict):
            raise DimensionStructureError(
                """grouped_elements must be a mapping of parent to child elements.
                    This should be either
                    * a dict mapping parents to a list of children each
                    * a pandas series, where (non-unique) key are parents and
                      values are children
                    * for the first level, it can just be a list of children
                """
            )

        parent_elements = parent.elements
        set_parent_elements = set(parent_elements)
        for p in set_parent_elements:
            if p not in grouped_elements:
                raise DimensionStructureError("Missing elements for parent: %s for" % p)

        elements = []
        group_sizes = []
        set_elements = set()
        for p, elems in grouped_elements.items():
            if p not in set_parent_elements:
                raise DimensionStructureError("Parent does not exist: %s" % p)
            group_size = len(elems)
            if not group_size:
                raise DimensionStructureError("No elements for parent: %s" % p)
            group_sizes.append(group_size)
            for e in elems:
                if e in set_elements:
                    raise DuplicateNameError("Duplicate element: %s" % e)
                set_elements.add(e)
                elements.append(e)
        elements = tuple(elements)

        return elements, group_sizes

    def __init__(self, parent, name, grouped_elements):
        self._parent = parent
        self._name = name
        self._children = dict()

        # create/check elements
        if self.is_dimension_root:
            elements, group_sizes = tuple([None]), [1]
            self._levels = {}
        else:
            self._levels = self.dimension._levels  # only store once
            # register by name
            if self.name in self._levels:
                raise DuplicateNameError(
                    "Name for dimension level already used: %s" % self.name
                )
            elements, group_sizes = self._parse_grouped_elements(
                self.parent, grouped_elements
            )

        self._group_matrix = create_group_matrix(group_sizes)
        self._elements = elements
        self._size = len(self.elements)

        self._indices = tuple(range(self.size))
        self._element2index = dict(zip(self.elements, self.indices))

        # register/link name
        self._levels[self.name] = self
        # register as child
        if not self.is_dimension_root:
            self.parent._children[self.name] = self

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
    """Comparaable to pandas.MultiIndex"""

    def __init__(self, dimension_levels):
        dimension_levels = dimension_levels or []

        # dimensions are added by dimension name!
        # TODO: what if we want multiple spacial dimensions? we could use alias,
        # but then it would be harder/impossible to automatically trnasform variables
        self._dimension_levels = OrderedDict()
        for d in dimension_levels:
            key = d.dimension_name
            if key in self._dimension_levels:
                raise DuplicateNameError("Duplicate dimension: %s" % d)
            self._dimension_levels[key] = d

        self._dimension_name2index = dict(
            (n, i) for i, n in enumerate(self.dimension_names)
        )

        self._size = len(self.dimension_levels)
        self._shape = tuple(d.size for d in self.dimension_levels)
        self._keys = tuple(product(*[d.elements for d in self.dimension_levels]))
        self._indices = tuple(product(*[d.indices for d in self.dimension_levels]))
        self._key2index = dict(zip(self.keys, self.indices))

    def get_dimension_index(self, dimension_name):
        return self._dimension_name2index[dimension_name]

    def get_dimension_level(self, dimension_name):
        return self._dimension_levels[dimension_name]

    def to_pandas_multi_index(self):
        if not pd:
            raise ImportError("pandas could not be imported")
        # special case scalar:
        if not self.dimensions:
            return None
        return pd.MultiIndex.from_tuples(self.keys, names=self.dimension_level_names)

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
            # TODO: create new function?
            # fix for 1-dim
            if not isinstance(key, tuple):
                key = (key,)
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
    """Comparable to pandas.Series with multiindex
    and/or multidimensional numpy.ndarray"""

    def __init__(self, name, data, domain, vartype, unit=None):

        if isinstance(unit, Unit):
            self._unit = unit
        else:
            self._unit = Unit(unit)

        if vartype not in ("intensive", "extensive", "weight"):
            raise ValueError("vartype not in ('intensive', 'extensive', 'weight')")

        self._vartype = vartype
        self._name = name

        # set domain
        if not domain:  # Scalar
            self._domain = Domain([])
        elif isinstance(domain, Domain):
            self._domain = domain
        else:
            self._domain = Domain(domain)

        # set data
        if isinstance(data, dict):
            self._data_matrix = self._domain.dict_to_matrix(data)
        elif isinstance(data, list):
            self._data_matrix = self._domain.records_to_matrix(data)
        elif pd and isinstance(data, pd.Series):
            if not data.index.is_unique:
                raise KeyError("Index must be unique")
            self._data_matrix = self._domain.dict_to_matrix(data.to_dict())
        elif isinstance(data, np.ndarray):
            self._data_matrix = data.copy()
        elif self.is_scalar and isinstance(data, (int, float)):
            self._data_matrix = self._domain.dict_to_matrix({tuple(): data})
        else:
            raise TypeError(type(data))

        if self._domain.shape != self._data_matrix.shape:
            raise DimensionStructureError(
                "Shape of data %s != shape of domain %s"
                % (self._data_matrix.shape, self._domain.shape)
            )

        if self.is_weight:
            if self.domain.size != 1:
                raise DimensionStructureError("Weights must have exactly one dimension")
            # check values
            group_matrix = self.domain.dimension_levels[0].group_matrix
            # create sum for groups. shapes: (m, n) * (n,) = (m,)
            sums = group_matrix.transpose().dot(self._data_matrix)
            shape = (group_matrix.shape[1],)
            if not np.allclose(np.ones(shape), sums):
                raise ValueError("Values in some groups don't add up to 1.0")

    @property
    def name(self):
        return self._name

    @property
    def is_intensive(self):
        return self._vartype == "intensive"

    @property
    def is_extensive(self):
        return self._vartype == "extensive"

    @property
    def is_weight(self):
        # TODO: also: unit is None or 1
        return self._vartype == "weight"

    @property
    def is_scalar(self):
        return self._domain.size == 0

    @property
    def domain(self):
        return self._domain

    @property
    def unit(self):
        return self._unit

    def to_dict(self, skip_0=False):
        res = {}
        for idx, key in self.domain.iter_indices_keys():
            val = self._data_matrix[idx]
            if val or not skip_0:
                res[key] = val
        return res

    def to_records(self, skip_0=False, value="value"):
        res = []
        dimension_level_names = self.domain.dimension_level_names
        for key, val in self.to_dict(skip_0=skip_0).items():
            rec = dict(zip(dimension_level_names, key))
            rec[value] = val
            res.append(rec)

        return res

    def as_normalized(self):
        s = np.sum(self._data_matrix)
        if not s:
            raise Exception("sum = 0")
        data = self._data_matrix / s
        return Variable(self.name, self.domain, data=data, unit=None, is_intensive=True)

    def as_weight(self):
        if self.is_weight:
            return self

        if self.domain.size != 1:
            raise DimensionStructureError("Weights must have exactly one dimension")
        dimension_level = self.domain.dimension_levels[0]

        # create sum for groups. shapes: (m, n) * (n,) = (m,)
        sums = dimension_level.group_matrix.transpose().dot(self._data_matrix)
        if not np.all(sums != 0):
            raise ValueError("some groups add up to 0")

        # create inverse, repeat, sum: (m,) => (1, m) => (n, m)
        # values = np.repeat(1 / sums, dimension_level.size, axis=1)
        sums = np.reshape(1 / sums, (1, sums.size))
        sums = np.repeat(sums, dimension_level.size, axis=0)
        sums = np.sum(sums * dimension_level.group_matrix, axis=1)
        data = sums * self._data_matrix
        # is_intensive=None ==> weights
        return Variable(
            name=self.name,
            domain=self.domain,
            data=data,
            unit=None,
            vartype="weight",
        )

    def aggregate(self, dimension_name, weights=None):

        if self.is_intensive:
            raise AggregationError("intensive aggregation without weights")

        dimension_levels = list(self.domain.dimension_levels)
        dim_idx = self.domain.get_dimension_index(dimension_name)
        dim_idx_last = len(dimension_levels) - 1
        dimension_level = dimension_levels[dim_idx]
        if dimension_level.is_dimension_root:
            raise AggregationError("dimension root cannot be aggregated further")

        new_dimension_level = dimension_level.parent
        group_matrix = dimension_level.group_matrix

        logging.debug("aggregate %s => %s" % (dimension_level, new_dimension_level))

        if weights:
            if (
                len(weights.domain.dimensions) != 1
                or weights.domain.dimension_levels[0] != new_dimension_level
            ):
                raise AggregationError(
                    "weights domain must be %s" % Domain([new_dimension_level])
                )
            if not weights.is_weight:
                raise TypeError("weight is not of type Weight")

            # weights is one dimensional, and number must be the same as
            # columns in group matrix
            n_rows, n_cols = group_matrix.shape
            weights_matrix = weights._data_matrix

            weights_matrix = np.repeat(
                np.reshape(weights_matrix, (1, n_cols)), n_rows, axis=0
            )
            # todo check sum of groups = 1
            group_matrix *= weights_matrix
            # if group sums are all 1, sums in each row in group_matrix must be 1
            row_sums = np.sum(group_matrix, axis=1)

            if not np.all(np.isclose(row_sums, 1)):
                raise ValueError(
                    """Aggregation checksum failed.
                    This should not happen (if weights are of type Weight)
                    """
                )

        dimension_levels[dim_idx] = new_dimension_level
        new_domain = Domain(dimension_levels)

        data_matrix = np.matmul(
            self._data_matrix.swapaxes(dim_idx, dim_idx_last), group_matrix
        ).swapaxes(dim_idx_last, dim_idx)

        return Variable(
            name=self.name,
            domain=new_domain,
            data=data_matrix,
            unit=self.unit,
            vartype=self._vartype,
        )

    def disaggregate(self, dimension_name, dimension_level_name, weights=None):
        if self.is_extensive and not weights:
            raise AggregationError(
                "extensive disaggregation without weights for %s" % dimension_level_name
            )

        dimension_levels = list(self.domain.dimension_levels)
        dim_idx = self.domain.get_dimension_index(dimension_name)
        dim_idx_last = len(dimension_levels) - 1
        dimension_level = dimension_levels[dim_idx]

        try:
            new_dimension_level = dimension_level.get_child(dimension_level_name)
        except KeyError:
            raise AggregationError(
                "dimension level %s has no sublevel named %s"
                % (dimension_level, dimension_level_name)
            )

        dimension_levels[dim_idx] = new_dimension_level
        new_domain = Domain(dimension_levels)

        group_matrix = new_dimension_level.group_matrix.transpose()

        logging.debug("disaggregate %s => %s" % (dimension_level, new_dimension_level))

        if weights:
            if (
                len(weights.domain.dimensions) != 1
                or weights.domain.dimension_levels[0] != new_dimension_level
            ):
                raise AggregationError(
                    "weights domain must be %s" % Domain([new_dimension_level])
                )
            # weights is one dimensional, and number must be the same as
            # columns in group matrix
            n_rows, n_cols = group_matrix.shape
            weights_matrix = weights._data_matrix

            weights_matrix = np.repeat(
                np.reshape(weights_matrix, (1, n_cols)), n_rows, axis=0
            )
            group_matrix *= weights_matrix
            # if group sums are all 1, sums in each row in group_matrix must be 1
            row_sums = np.sum(group_matrix, axis=1)
            if not np.all(np.isclose(row_sums, 1)):
                raise ValueError(
                    """Aggregation checksum failed.
                    This should not happen (if weights are of type Weight)
                    """
                )

        data_matrix = np.matmul(
            self._data_matrix.swapaxes(dim_idx, dim_idx_last), group_matrix
        ).swapaxes(dim_idx_last, dim_idx)

        return Variable(
            name=self.name,
            domain=new_domain,
            data=data_matrix,
            unit=self.unit,
            vartype=self._vartype,
        )

    def expand(self, dimension):
        if not dimension.is_dimension_root:
            logging.warning(
                "You should use dimension root %s instead of level %s"
                % (dimension, dimension.dimension)
            )
            dimension = dimension.dimension

        logging.debug("adding %s" % (dimension.name,))
        data_matrix = np.expand_dims(self._data_matrix, axis=self.domain.size)
        domain = Domain(list(self.domain.dimension_levels) + [dimension])
        return Variable(
            name=self.name,
            domain=domain,
            data=data_matrix,
            unit=self.unit,
            vartype=self._vartype,
        )

    def squeeze(self, dimension_name):
        logging.debug("removing %s" % (dimension_name,))

        dim_idx = self.domain.get_dimension_index(dimension_name)

        if not self.domain.dimension_levels[dim_idx].is_dimension_root:
            raise DimensionStructureError(
                "can only squeeze dimension if it is fully aggregated"
            )

        data_matrix = np.squeeze(self._data_matrix, axis=dim_idx)
        domain = Domain(
            self.domain.dimension_levels[:dim_idx]
            + self.domain.dimension_levels[dim_idx + 1 :]
        )
        return Variable(
            name=self.name,
            domain=domain,
            data=data_matrix,
            unit=self.unit,
            vartype=self._vartype,
        )

    def reorder(self, dimension_names):
        if set(dimension_names) != set(self.domain.dimension_names):
            raise DimensionStructureError(
                "reordering must contain all dimensions of origin al data"
            )

        indices = [self.domain.get_dimension_index(n) for n in dimension_names]
        data_matrix = self._data_matrix.transpose(indices)
        domain = Domain([self.domain.dimension_levels[i] for i in indices])
        return Variable(
            name=self.name,
            domain=domain,
            data=data_matrix,
            unit=self.unit,
            vartype=self._vartype,
        )

    def transform(self, domain, level_weights=None):
        """TODO"""

        if not isinstance(domain, Domain):
            domain = Domain(domain)

        add_dimensions = set(domain.dimensions) - set(self.domain.dimensions)
        drop_dimensions = set(self.domain.dimensions) - set(domain.dimensions)
        change_dimensions = set(self.domain.dimensions) & set(domain.dimensions)

        level_weights = level_weights or {}

        def get_weights(dimension_level):
            level_name = dimension_level.name
            weights = level_weights.get(level_name)
            if not weights:
                return None
            weights = weights.transform(Domain([dimension_level]))
            weights = weights.as_weight()
            return weights

        result = self
        for dim in drop_dimensions:
            # aggregate to root, then drop
            path = self.domain.get_dimension_level(dim.name).path
            level_names = list(reversed(path))[1:]
            for level_name in level_names:
                weights = get_weights(dim.get_level(level_name))
                result = result.aggregate(dim.name, weights=weights)
            result = result.squeeze(dim.name)

        for dim in add_dimensions:
            # add, then disaggregate
            result = result.expand(dim)
            path = domain.get_dimension_level(dim.name).path
            level_names = path[1:]
            for level_name in level_names:
                weights = get_weights(dim.get_level(level_name))
                result = result.disaggregate(dim.name, level_name, weights=weights)

        for dim in change_dimensions:
            # aggregate up to closest common ancestor, then down
            path_up = self.domain.get_dimension_level(dim.name).path
            path_down = domain.get_dimension_level(dim.name).path
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
                weights = weights = get_weights(dim.get_level(level_name))
                result = result.aggregate(dim.name, weights=weights)
            for level_name in path_down:
                weights = weights = get_weights(dim.get_level(level_name))
                result = result.disaggregate(dim.name, level_name, weights=weights)

        result = result.reorder(domain.dimension_names)

        return result

    def __str__(self):
        return str(self._data_matrix)

    def to_series(self):
        if not pd:
            raise ImportError("pandas could not be imported")

        # TODO: REALLY check that alignment of keys and values is correct!!
        # special case scalar

        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
        # flatten into array C-style
        data = self._data_matrix.flatten("C")
        index = self.domain.to_pandas_multi_index()
        return pd.Series(data, index=index, name=self.name)


class ExtensiveVariable(Variable):
    def __init__(self, name, data, domain, unit=None):
        super().__init__(
            name=name, data=data, unit=unit, domain=domain, vartype="extensive"
        )


class IntensiveVariable(Variable):
    def __init__(self, name, data, domain, unit=None):
        super().__init__(
            name=name, data=data, unit=unit, domain=domain, vartype="intensive"
        )


class ExtensiveScalar(ExtensiveVariable):
    def __init__(self, name, value, unit=None):
        super().__init__(name=name, data=value, unit=unit, domain=[])


class IntensiveScalar(IntensiveVariable):
    def __init__(self, name, value, unit=None):
        super().__init__(name=name, data=value, unit=unit, domain=[])


class Weight(Variable):
    def __init__(self, name, data, dimension_level):
        super().__init__(
            name=name, data=data, unit=None, domain=[dimension_level], vartype="weight"
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)

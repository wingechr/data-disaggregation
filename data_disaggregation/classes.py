"""Wrapper classes around multidimensional numpy matrices
for (dis)aggregation of data.
"""

import logging
from collections import OrderedDict
from itertools import product

import numpy as np

from .exceptions import AggregationError, DimensionStructureError, DuplicateNameError
from .functions import create_group_matrix, group_unique_values

# pandas is no a required dependency
try:
    import pandas as pd
except ImportError:
    pd = None


class DimensionLevel:
    """Dimension at a specific hierarchy level.

    Comparable to a pd.Index, but grouped and linked to other levels in a tree

    Users can but should rather not instance this class directly.
    a better way is to create a new top level Dimension and then
    add new levels using the add_level() method.


    Args:
        parent(DimensionLevel): parent dimension level
        name(str): level name
        grouped_elements: must be a mapping of parent to child elements.
          This should be either:

          * a dict mapping parents to a list of children each
          * a pandas series, where (non-unique) key are parents and
            values are children
          * for the first level, it can just be a list of children

    """

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

    def __str__(self):
        return "/".join(self.path)

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
                grouped_elements = group_unique_values(grouped_elements.iteritems())

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

    def add_level(self, name, grouped_elements):
        """Add a new child level.

        Args:
            name(str) name of dimension level
            grouped_elements: must be a mapping of parent to child elements.
              This should be either:

                * a dict mapping parents to a list of children each
                * a pandas series, where (non-unique) key are parents and
                  values are children
                * for the first level, it can just be a list of children

        """
        return DimensionLevel(parent=self, name=name, grouped_elements=grouped_elements)

    def get_level(self, name):
        """get level by name (anywhere in the dimension)"""
        return self._levels[name]

    def get_child(self, name):
        """get child level by name"""
        return self._children[name]

    @property
    def group_matrix(self):
        """2 dimensional numpy matrix to group elements to the parent level"""
        return self._group_matrix.copy()

    @property
    def is_dimension_root(self):
        """True if level is root level"""
        return self.parent is None

    @property
    def parent(self):
        """parent level"""
        return self._parent

    @property
    def dimension_name(self):
        """name of the dimension (=name of the root level)"""
        return self.dimension.name

    @property
    def name(self):
        """name of the level"""
        return self._name

    @property
    def indices(self):
        """indices of elements"""
        return self._indices

    @property
    def elements(self):
        """list of elements"""
        return self._elements

    @property
    def size(self):
        """number of elements"""
        return self._size

    @property
    def dimension(self):
        """link to root level"""
        return self if self.is_dimension_root else self.parent.dimension

    @property
    def path(self):
        """list of levels from root down to this level"""
        if self.is_dimension_root:
            return [self.name]
        else:
            return self.parent.path + [self.name]


class Dimension(DimensionLevel):
    """Dimension root level

    Args:
        name: name of dimension
    """

    def __init__(self, name):
        super().__init__(parent=None, name=name, grouped_elements=None)


class Domain:
    """List of DimensionLevel

    Comparable to pandas.MultiIndex
    """

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

    def __str__(self):
        dims = self._dimension_levels.items()
        dims_str = ", ".join("%s=%s(%d)" % (n, d, d.size) for n, d in dims)
        return "(" + dims_str + ")"

    def iter_indices_keys(self):
        """TODO: docstring"""
        yield from zip(self.indices, self.keys)

    def get_index(self, key):
        """TODO: docstring"""
        return self._key2index[key]

    # create data matrix from dict data
    def dict_to_matrix(self, data):
        """TODO: docstring"""
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
        """TODO: docstring"""
        dimension_level_names = self.dimension_level_names
        data_dict = {}
        for rec in data:
            key = tuple(rec[n] for n in dimension_level_names)
            if key in data_dict:
                raise KeyError(key)
            data_dict[key] = rec[value]
        return self.dict_to_matrix(data_dict)

    def get_dimension_index(self, dimension_name):
        """TODO: docstring"""
        return self._dimension_name2index[dimension_name]

    def get_dimension_level(self, dimension_name):
        """TODO: docstring"""
        return self._dimension_levels[dimension_name]

    def to_pandas_multi_index(self):
        """TODO: docstring"""

        if not pd:
            raise ImportError("pandas could not be imported")
        # special case scalar:
        if not self.dimensions:
            return None
        return pd.MultiIndex.from_tuples(self.keys, names=self.dimension_level_names)

    @property
    def dimension_levels(self):
        """TODO: docstring"""
        return tuple(self._dimension_levels.values())

    @property
    def dimensions(self):
        """TODO: docstring"""
        return tuple(d.dimension for d in self.dimension_levels)

    @property
    def dimension_names(self):
        """TODO: docstring"""
        return tuple(d.name for d in self.dimensions)

    @property
    def dimension_level_names(self):
        """TODO: docstring"""
        return tuple(d.name for d in self.dimension_levels)

    @property
    def size(self):
        """TODO: docstring"""
        return self._size

    @property
    def shape(self):
        """TODO: docstring"""
        return self._shape

    @property
    def keys(self):
        """TODO: docstring"""
        return self._keys

    @property
    def indices(self):
        """TODO: docstring"""
        return self._indices


class Variable:
    """Data container over a multidimensional domain.

    Comparable to pandas.Series with multiindex
    and/or multidimensional numpy.ndarray

    Args:
        name(str): name of variable
        data: data can be passed in different ways:

          * dictionary of key -> value,
            where key is a tuple that is an element of the domain
          * list of records. records must have fields for each
            dimension in the domain and a column named `value`
          * pandas Series with a MultiIndex
          * if variable is a scalar: simple number

        domain: list of DimensionLevel instances (or Domain instance)
        vartype(str): one of "intensive", "extensive" or "weight":

          * extensive variables can be simply aggregated by summing values
            in each group
          * intensive variables can simply be disaggregated by duplicating
            values to all child elements in a group
          * weights sum up to 1.0 in each group

        unit(optional): not implemented yet
          (maybe use https://pint.readthedocs.io/en/stable/)

    """

    def __init__(self, name, data, domain, vartype, unit=None):
        self._unit = unit

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
        """name of variable"""
        return self._name

    @property
    def is_intensive(self):
        """TODO: docstring"""
        return self._vartype == "intensive"

    @property
    def is_extensive(self):
        """TODO: docstring"""
        return self._vartype == "extensive"

    @property
    def is_weight(self):
        """TODO: docstring"""
        return self._vartype == "weight"

    @property
    def is_scalar(self):
        """TODO: docstring"""
        return self._domain.size == 0

    @property
    def domain(self):
        """TODO: docstring"""
        return self._domain

    @property
    def unit(self):
        """TODO: docstring"""
        return self._unit

    def to_dict(self, skip_0=False):
        res = {}
        for idx, key in self.domain.iter_indices_keys():
            val = self._data_matrix[idx]
            if val or not skip_0:
                res[key] = val
        return res

    def to_records(self, skip_0=False, value="value"):
        """TODO: docstring"""
        res = []
        dimension_level_names = self.domain.dimension_level_names
        for key, val in self.to_dict(skip_0=skip_0).items():
            rec = dict(zip(dimension_level_names, key))
            rec[value] = val
            res.append(rec)

        return res

    def as_normalized(self):
        """TODO: docstring"""
        s = np.sum(self._data_matrix)
        if not s:
            raise Exception("sum = 0")
        data = self._data_matrix / s
        return Variable(self.name, self.domain, data=data, unit=None, is_intensive=True)

    def as_weight(self):
        """TODO: docstring"""

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
        """TODO: docstring"""

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
        """TODO: docstring"""

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
        """TODO: docstring"""
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
        """TODO: docstring"""

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
        """Return new Variable with reordered Dimensions

        Args:
            dimension_names(list): list of names of existing dimensions
              in desired order
        """
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
        """Main function to map variable to a new domain.

        Args:
            domain: list of DimensionLevel instances (or Domain instance)
            level_weights(dict, optional):
               dimension level names -> one dimensional variables that will be used
               as weights for this level

        """

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
        """Return indexed pandas Series"""
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
    """Shorthand for `Variable(vartype="extensive")`

    Args:
        name(str): name of variable
        data: data can be passed in different ways:

          * dictionary of key -> value, where key is a tuple that is an element
            of the domain
          * list of records. records must have fields for each dimension
            in the domain and a column named `value`
          * pandas Series with a MultiIndex
          * if variable is a scalar: simple number

        domain: list of DimensionLevel instances (or Domain instance)
        unit(optional): not implemented yet

    """

    def __init__(self, name, data, domain, unit=None):
        super().__init__(
            name=name, data=data, unit=unit, domain=domain, vartype="extensive"
        )


class IntensiveVariable(Variable):
    """Shorthand for `Variable(vartype="intensive")`

    Args:
        name(str): name of variable
        data: data can be passed in different ways:

          * dictionary of key -> value,
            where key is a tuple that is an element of the domain
          * list of records. records must have fields for each
            dimension in the domain and a column named `value`
          * pandas Series with a MultiIndex
          * if variable is a scalar: simple number

        domain: list of DimensionLevel instances (or Domain instance)
        unit(optional): not implemented yet

    """

    def __init__(self, name, data, domain, unit=None):
        super().__init__(
            name=name, data=data, unit=unit, domain=domain, vartype="intensive"
        )


class Weight(Variable):
    """Shorthand for `Variable(vartype="weight")`

    In addition, weights are special Variables with:

    * domain is exactly one dimension
    * values in each group add up to 1.0

    Args:
        name(str): name of variable
        data: data can be passed in different ways:

          * dictionary of key -> value,
            where key is a tuple that is an element of the domain
          * list of records. records must have fields for each
            dimension in the domain and a column named `value`
          * pandas Series with a MultiIndex

        dimension_level(DimensionLevel): domain of data (weights have only one)
        unit(optional): not implemented yet

    """

    def __init__(self, name, data, dimension_level):
        super().__init__(
            name=name, data=data, unit=None, domain=[dimension_level], vartype="weight"
        )


class ExtensiveScalar(ExtensiveVariable):
    """Shorthand for `ExtensiveVariable(domain=None)`

    Args:
        name(str): name of variable
        value: number
        unit(optional): not implemented yet

    """

    def __init__(self, name, value, unit=None):
        super().__init__(name=name, data=value, unit=unit, domain=[])


class IntensiveScalar(IntensiveVariable):
    """Shorthand for `IntensiveVariable(domain=None)`

    Args:
        name(str): name of variable
        value: number
        unit(optional): not implemented yet

    """

    def __init__(self, name, value, unit=None):
        super().__init__(name=name, data=value, unit=unit, domain=[])


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)

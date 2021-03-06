"""Wrapper classes around multidimensional numpy matrices
for (dis)aggregation of data.
"""

import logging
from collections import OrderedDict
from itertools import product

import numpy as np

from .exceptions import (
    AggregationError,
    ArithmeticError,
    DimensionStructureError,
    DuplicateNameError,
)
from .functions import (
    create_group_matrix,
    get_path_up_down,
    group_unique_values,
    normalize_groups,
)

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
        self._children = OrderedDict()

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
        self._element2index = dict(zip(self.elements, self._indices))

        # register/link name
        self._levels[self.name] = self
        # register as child
        if not self.is_dimension_root:
            self.parent._children[self.name] = self

    def __str__(self):
        return "/".join(self.path)

    @staticmethod
    def _parse_grouped_elements(parent, grouped_elements):
        """
        only used in __init__
        """
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
        set_grouped_elements = set(grouped_elements)
        if set_grouped_elements - set_parent_elements:
            raise DimensionStructureError(
                "Elements not in parent level: %s"
                % (set_grouped_elements - set_parent_elements)
            )
        elif set_parent_elements - set_grouped_elements:
            raise DimensionStructureError(
                "Missing elements from parent level: %s"
                % (set_parent_elements - set_grouped_elements)
            )

        elements = []
        group_sizes = []
        set_elements = set()
        # IMPORTANT: group order defined by parent!
        for p in parent_elements:
            elems = grouped_elements[p]
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
    def children(self):
        return OrderedDict(self._children)

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

    Args:
        dimension_levels: either

          * list of DimensionLevel
          * a single DimensionLevel
          * None (Scalar)

    """

    def __init__(self, dimension_levels):
        dimension_levels = dimension_levels or []
        if isinstance(dimension_levels, DimensionLevel):
            dimension_levels = [dimension_levels]

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
        self._indices = tuple(product(*[d._indices for d in self.dimension_levels]))
        self._key2index = dict(zip(self._keys, self._indices))

    def __str__(self):
        dims = self._dimension_levels.items()
        dims_str = ", ".join("%s=%s(%d)" % (n, d, d.size) for n, d in dims)
        return "(" + dims_str + ")"

    def __eq__(self, other):
        return self._size == other._size and all(
            d1 == d2 for d1, d2 in zip(self.dimension_levels, other.dimension_levels)
        )

    def dict_to_matrix(self, data):
        """create data matrix from dict data

        Args:
            data(dict)
        """
        d_matrix = np.zeros(shape=self.shape)
        for key, val in data.items():
            # TODO: create new function?
            # fix for 1-dim
            if not isinstance(key, tuple):
                key = (key,)
            idx = self._key2index[key]
            d_matrix[idx] = val
        return d_matrix

    def records_to_matrix(self, data, value="value"):
        """create data matrix from records data

        Args:
            data(list): list of records
            value(str, optional): value column
        """
        dimension_level_names = self.dimension_level_names
        data_dict = {}
        for rec in data:
            key = tuple(rec[n] for n in dimension_level_names)
            if key in data_dict:
                raise KeyError(key)
            data_dict[key] = rec[value]
        return self.dict_to_matrix(data_dict)

    def get_dimension_index(self, dimension_name):
        """get numerical index of dimension from name

        Args:
            dimension_name(str): name of dimension
        """
        return self._dimension_name2index[dimension_name]

    def get_dimension(self, dimension_name):
        """get dimension object by name

        Args:
            dimension_name(str): name of dimension
        """
        return self.get_dimension_level(dimension_name).dimension

    def has_dimension(self, dimension_name):
        """check if dimension is in domain

        Args:
            dimension_name(str): name of dimension
        """
        return dimension_name in self._dimension_levels

    def get_dimension_level(self, dimension_name):
        """get current level of dimension

        Args:
            dimension_name(str)
        """
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
    def is_scalar(self):
        return self.size == 0


class VariableBase:
    pass


class MixinNumpyMath:
    """delegate math operations to numpy"""

    def _get_unit_add(self, other):
        unit = self._unit
        if isinstance(other, VariableBase):
            other_unit = other._unit
        else:
            other_unit = unit
        if unit != other_unit:
            raise ArithmeticError("unit mismatch")
        return unit

    def _get_unit_mult(self, other):
        unit = self._unit
        if isinstance(other, VariableBase):
            other_unit = other._unit
        else:
            other_unit = unit
        if unit or other_unit:
            raise NotImplementedError("unit multiplication")
        return unit

    def _get_unit_div(self, other):
        unit = self._unit
        if isinstance(other, VariableBase):
            other_unit = other._unit
        else:
            other_unit = unit
        if unit or other_unit:
            raise NotImplementedError("unit division")
        return unit

    def _get_unit_rdiv(self, other):
        unit = self._unit
        if isinstance(other, VariableBase):
            other_unit = other._unit
        else:
            other_unit = unit
        if unit or other_unit:
            raise NotImplementedError("unit division")
        return unit

    def _get_vartype(self, other):
        vartype = self._vartype
        if isinstance(other, VariableBase):
            other_vartype = other._vartype
        else:
            other_vartype = vartype
        if vartype != other_vartype:
            raise ArithmeticError("vartype mismatch")
        if vartype != "extensive":
            raise ArithmeticError("invalid vartype for operation: %s" % vartype)
        return vartype

    def _get_domain(self, other):
        # domain must be either equal, or one must be a scalar
        domain = self._domain
        if isinstance(other, VariableBase):
            other_domain = other._domain
        else:
            other_domain = domain
        if domain.is_scalar:
            return other_domain
        elif other_domain.is_scalar:
            return domain
        if domain != other_domain:
            raise ArithmeticError("vartype mismatch")
        return domain

    @staticmethod
    def _get_data_matrix_other(other):
        if isinstance(other, VariableBase):
            data_matrix = other._data_matrix
        else:
            data_matrix = other
        return data_matrix

    def __add__(self, other):
        unit = self._get_unit_add(other)
        vartype = self._get_vartype(other)
        domain = self._get_domain(other)
        # use numpy to do actual calculation
        data = self._data_matrix + self._get_data_matrix_other(other)
        return Variable(data, domain, vartype, unit=unit)

    def __sub__(self, other):
        unit = self._get_unit_add(other)
        vartype = self._get_vartype(other)
        domain = self._get_domain(other)
        # use numpy to do actual calculation
        data = self._data_matrix - self._get_data_matrix_other(other)
        return Variable(data, domain, vartype, unit=unit)

    def __mul__(self, other):
        unit = self._get_unit_mult(other)
        vartype = self._get_vartype(other)
        domain = self._get_domain(other)
        # use numpy to do actual calculation
        data = self._data_matrix * self._get_data_matrix_other(other)
        return Variable(data, domain, vartype, unit=unit)

    def __rmul__(self, other):
        unit = self._get_unit_mult(other)
        vartype = self._get_vartype(other)
        domain = self._get_domain(other)
        # use numpy to do actual calculation
        data = self._get_data_matrix_other(other) * self._data_matrix
        return Variable(data, domain, vartype, unit=unit)

    def __truediv__(self, other):
        unit = self._get_unit_rdiv(other)
        vartype = self._get_vartype(other)
        domain = self._get_domain(other)
        # use numpy to do actual calculation
        data = self._data_matrix / self._get_data_matrix_other(other)
        return Variable(data, domain, vartype, unit=unit)

    def __rtruediv__(self, other):
        unit = self._get_unit_rdiv(other)
        vartype = self._get_vartype(other)
        domain = self._get_domain(other)
        # use numpy to do actual calculation
        data = self._get_data_matrix_other(other) / self._data_matrix
        return Variable(data, domain, vartype, unit=unit)

    # unary methods
    def __neg__(self):
        data = -self._data_matrix
        return Variable(data, self._domain, self._vartype, unit=self._unit)

    def fillna(self, value=0):
        data = np.nan_to_num(self._data_matrix, nan=value)
        return Variable(data, self._domain, self._vartype, unit=self._unit)


class Variable(VariableBase, MixinNumpyMath):
    """Data container over a multidimensional domain.

    Comparable to pandas.Series with multiindex
    and/or multidimensional numpy.ndarray

    Args:
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

    def __init__(self, data, domain, vartype, unit=None):
        self._unit = unit

        if vartype not in ("intensive", "extensive", "weight", "weight_w_nan"):
            raise ValueError(
                "vartype not in ('intensive', 'extensive', 'weight_w_nan')"
            )

        self._vartype = vartype

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
        elif self.is_scalar:
            data = float(data)
            self._data_matrix = self._domain.dict_to_matrix({tuple(): data})
        else:
            raise TypeError(type(data))

        if self._domain.shape != self._data_matrix.shape:
            raise DimensionStructureError(
                "Shape of data %s != shape of domain %s"
                % (self._data_matrix.shape, self._domain.shape)
            )

        if self.is_weight:
            if self._domain.size != 1:
                raise DimensionStructureError("Weights must have exactly one dimension")

            # check values
            if np.any(self._data_matrix < 0):
                raise ValueError("all values must be >= 0")
            if np.any(np.isnan(self._data_matrix)) and not self.is_weight_w_nan:
                raise ValueError("all values must be >= 0")

            group_matrix = self._domain.dimension_levels[0].group_matrix
            # create sum for groups. shapes: (m, n) * (n,) = (m,)
            sums = group_matrix.transpose().dot(self._data_matrix)
            shape = (group_matrix.shape[1],)
            if not np.allclose(np.ones(shape), sums):
                if self.is_weight_w_nan:
                    # sums must be either 0 or 1
                    pass  # FIXME
                else:
                    raise ValueError("Values in some groups don't add up to 1.0")

    def to_series(self):
        """Return indexed pandas Series"""
        if not pd:
            raise ImportError("pandas could not be imported")

        # TODO: REALLY check that alignment of keys and values is correct!!
        # special case scalar
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
        # flatten into array C-style
        data = self._data_matrix.flatten("C")

        if not self._domain.dimensions:
            index = None
        elif self._domain.size == 1:  # one dimensional index
            dimension_level = self._domain.dimension_levels[0]
            index = pd.Index(dimension_level.elements, name=dimension_level.name)
        else:  # multilevel index
            index = pd.MultiIndex.from_tuples(
                self._domain._keys, names=self._domain.dimension_level_names
            )

        return pd.Series(data, index=index)

    def to_dict(self, skip_0=False):
        """Return data as an dict

        Args:
            skip_0(bool, optional): if True: do not include cells with value == 0
        """
        res = {}
        for idx, key in zip(self._domain._indices, self._domain._keys):
            val = self._data_matrix[idx]
            if val or not skip_0:
                res[key] = val
        return res

    def to_records(self, skip_0=False, value="value"):
        """Return data as an iterable of records

        Args:
            skip_0(bool, optional): if True: do not include cells with value == 0
            value(str, optional): name of the value column
        """
        res = []
        dimension_level_names = self._domain.dimension_level_names
        for key, val in self.to_dict(skip_0=skip_0).items():
            rec = dict(zip(dimension_level_names, key))
            rec[value] = val
            res.append(rec)

        return res

    def to_weight(self, on_group_0="error"):
        """Convert variable into weight.
        Only possible if Domain has only one dimension.

        Args:
        on_group_0(str, optional): how to handle group sums of 0: must be one of

            * 'error': the default, raises an Error
            * 'nan': sets resulting weights to nan.
               this can lead to a loss of values on disaggregation
            * 'equal': sets resulting weights to 1/n
        """

        if self._domain.size != 1:
            raise DimensionStructureError("Weights must have exactly one dimension")
        dimension_level = self._domain.dimension_levels[0]
        group_matrix = dimension_level.group_matrix
        data_matrix = self._data_matrix

        data = normalize_groups(group_matrix, data_matrix, on_group_0=on_group_0)

        # is_intensive=None ==> weights

        vartype = "weight"
        if np.any(np.isnan(data)) and on_group_0 == "nan":
            vartype = "weight_w_nan"

        return Variable(
            domain=self._domain,
            data=data,
            unit=None,
            vartype=vartype,
        )

    def aggregate(self, dimension_name, weights=None):
        """aggregate one dimension lvel to the next

        Args:
            dimension_name(str): name of dimension to be aggregated
            weights(Weights, optional): weights variable for aggregation,
              must have the the target dimension level as domain
        """

        # get the current level of the dimension
        dimension_levels = list(self._domain.dimension_levels)
        dim_idx = self._domain.get_dimension_index(dimension_name)
        dim_idx_last = len(dimension_levels) - 1
        dimension_level = dimension_levels[dim_idx]

        if dimension_level.is_dimension_root:
            raise AggregationError(
                "dimension cannot be aggregated further: %s" % dimension_level
            )
        if self.is_intensive and not weights:
            raise AggregationError(
                "intensive aggregation without weights on level: %s" % dimension_level
            )

        new_dimension_level = dimension_level.parent
        group_matrix = dimension_level.group_matrix

        logging.debug("aggregate %s => %s" % (dimension_level, new_dimension_level))

        if weights:

            logging.debug("weights: %s", weights._domain)

            if weights._domain.size != 1:
                raise AggregationError("weights domain must be exacly one dimension")
            if weights._domain.dimension_levels[0] != dimension_level:
                raise AggregationError(
                    "weights domain for aggregation must be %s, not %s"
                    % (Domain([dimension_level]), weights._domain)
                )
            if not weights.is_weight:
                raise TypeError("weight is not of type Weight")

            # weights is one dimensional, and number must be the same as
            # columns in group matrix

            n_rows, n_cols = group_matrix.shape
            weights_matrix = weights._data_matrix

            weights_matrix = np.repeat(
                np.reshape(weights_matrix, (n_rows, 1)), n_cols, axis=1
            )

            group_matrix *= weights_matrix

        dimension_levels[dim_idx] = new_dimension_level
        new_domain = Domain(dimension_levels)

        data_matrix = np.matmul(
            self._data_matrix.swapaxes(dim_idx, dim_idx_last), group_matrix
        ).swapaxes(dim_idx_last, dim_idx)

        return Variable(
            domain=new_domain,
            data=data_matrix,
            unit=self._unit,
            vartype=self._vartype,
        )

    def disaggregate(self, dimension_name, dimension_level_name, weights=None):
        """disaggregate one dimension to one of the next child levels

        Args:
            dimension_name(str): name of dimension to be disaggregated
            dimension_level_name(str): name of new child level
            weights(Weights, optional): weights variable for disaggregation,
              must have the the target dimension level as domain
        """

        if self.is_extensive and not weights:
            raise AggregationError(
                "extensive disaggregation without weights for %s" % dimension_level_name
            )

        dimension_levels = list(self._domain.dimension_levels)
        dim_idx = self._domain.get_dimension_index(dimension_name)
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

        logging.debug(
            "disaggregate %s (%s) => %s",
            dimension_level,
            self._vartype,
            new_dimension_level,
        )

        if weights:
            if (
                len(weights._domain.dimensions) != 1
                or weights._domain.dimension_levels[0] != new_dimension_level
            ):
                raise AggregationError(
                    "weights domain for disaggregation must be %s"
                    % Domain([new_dimension_level])
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
            group_matrix *= np.nan_to_num(weights_matrix)
            # if group sums are all 1, sums in each row in group_matrix must be 1
            row_sums = np.sum(group_matrix, axis=1)

            if not np.isclose(
                np.sum(group_matrix), np.sum(np.nan_to_num(weights._data_matrix))
            ):
                raise Exception(
                    """Aggregation checksum failed
                    sum(weights) should be sum(group_matrix)
                    This should not happen (if weights are of type Weight)
                    """
                )

            if not np.all(np.isclose(row_sums, 1)):
                if weights.is_weight_w_nan and np.all(
                    np.isclose(row_sums, 1) | np.isclose(row_sums, 0)
                ):
                    # partially defined weights: row_sums can be 1 or 0
                    pass
                else:
                    raise ValueError(
                        """Aggregation checksum failed
                    """
                    )

        data_matrix = np.matmul(
            self._data_matrix.swapaxes(dim_idx, dim_idx_last), group_matrix
        ).swapaxes(dim_idx_last, dim_idx)

        return Variable(
            domain=new_domain,
            data=data_matrix,
            unit=self._unit,
            vartype=self._vartype,
        )

    def expand(self, dimension):
        """add dimension on aggregated level

        Args:
            dimension(Dimension): root level dimension
        """

        if not dimension.is_dimension_root:
            logging.warning(
                "You should use dimension root %s instead of level %s"
                % (dimension, dimension.dimension)
            )
            dimension = dimension.dimension

        logging.debug("adding %s" % (dimension.name,))
        data_matrix = np.expand_dims(self._data_matrix, axis=self._domain.size)
        domain = Domain(list(self._domain.dimension_levels) + [dimension])
        return Variable(
            domain=domain,
            data=data_matrix,
            unit=self._unit,
            vartype=self._vartype,
        )

    def squeeze(self, dimension_name):
        """remove aggregated dimension

        Args:
            dimension_name: names of existing dimensions to be dropped
        """

        logging.debug("removing %s" % (dimension_name,))

        dim_idx = self._domain.get_dimension_index(dimension_name)

        if not self._domain.dimension_levels[dim_idx].is_dimension_root:
            raise DimensionStructureError(
                "can only squeeze dimension if it is fully aggregated"
            )

        data_matrix = np.squeeze(self._data_matrix, axis=dim_idx)
        domain = Domain(
            self._domain.dimension_levels[:dim_idx]
            + self._domain.dimension_levels[dim_idx + 1 :]
        )
        return Variable(
            domain=domain,
            data=data_matrix,
            unit=self._unit,
            vartype=self._vartype,
        )

    def reorder(self, dimension_names):
        """Return new Variable with reordered Dimensions

        Args:
            dimension_names(list): list of names of existing dimensions
              in desired order
        """
        if set(dimension_names) != set(self._domain.dimension_names):
            raise DimensionStructureError(
                "reordering must contain all dimensions of origin al data"
            )

        indices = [self._domain.get_dimension_index(n) for n in dimension_names]
        data_matrix = self._data_matrix.transpose(indices)
        domain = Domain([self._domain.dimension_levels[i] for i in indices])
        return Variable(
            domain=domain,
            data=data_matrix,
            unit=self._unit,
            vartype=self._vartype,
        )

    def get_transform_steps(self, domain, level_weights=None, on_group_0="error"):
        """
        Args:
            domain: list of DimensionLevel instances (or Domain instance)
            level_weights(dict, optional):
               dimension level names -> one dimensional variables that will be used
               as weights for this level
            on_group_0(str, optional): how to handle group sums of 0: must be one of
            * 'error': the default, raises an Error
            * 'nan': sets resulting weights to nan.
               this can lead to a loss of values on disaggregation
            * 'equal': sets resulting weights to 1/n
        Returns:
            OrderedDict: dimension -> steps
              * each element contains steps for a dimension
              * dimensions are all dimensions in source and target domain
              * each step is (from_level, to_level, action, (weight_level, weight_var))
        """
        if not isinstance(domain, Domain):
            domain = Domain(domain)
        level_weights = level_weights or {}

        def get_weights(dimension_level):
            level_name = dimension_level.name
            weights = level_weights.get(level_name)
            if not weights:
                return None
            try:
                weights = weights.transform(Domain([dimension_level]))
                weights = weights.to_weight(on_group_0=on_group_0)
            except AggregationError:
                raise AggregationError("Cannot transform weight: %s" % weights)
            return weights

        result = OrderedDict()
        for dim in domain.dimensions + self._domain.dimensions:
            # do not add dimension twice (if in source AND target)
            if dim in result:
                continue
            steps = []
            result[dim] = steps

            try:
                source_level = self._domain.get_dimension_level(dim.name)
                expand = False
            except KeyError:
                # new dimension => first step is to add (expand)
                source_level = dim
                expand = True

            try:
                target_level = domain.get_dimension_level(dim.name)
                squeeze = False
            except KeyError:
                # remove dimension => last step is to remove (squeeze)
                target_level = dim
                squeeze = True

            if expand:
                steps.append((None, dim, "expand", None))

            path_up, peak, path_down = get_path_up_down(
                source_level.path, target_level.path
            )
            for i, p_l_n in enumerate(path_up):
                if i == len(path_up) - 1:
                    p_u_n = peak
                else:
                    p_u_n = path_up[i + 1]
                p_l = dim.get_level(p_l_n)
                p_u = dim.get_level(p_u_n) if p_u_n else None
                if self.is_intensive:
                    weight = get_weights(p_l)
                else:
                    weight = None
                steps.append((p_l, p_u, "aggregate", weight))

            for i, p_l_n in enumerate(path_down):
                if i == 0:
                    p_u_n = peak
                else:
                    p_u_n = path_down[i - 1]
                p_l = dim.get_level(p_l_n) if p_l_n else None
                p_u = dim.get_level(p_u_n)
                weight = None

                if self.is_extensive:
                    weight = get_weights(p_l)
                else:
                    weight = None
                steps.append((p_u, p_l, "disaggregate", weight))

            if squeeze:
                steps.append((dim, None, "squeeze", None))

            if not steps:
                # no change: still add a dummy step (for visualization)
                steps.append((source_level, source_level, "keep", None))

        return result

    def transform(self, domain, level_weights=None, on_group_0="error"):
        """Main function to map variable to a new domain.

        Args:
            domain: list of DimensionLevel instances (or Domain instance)
            level_weights(dict, optional):
               dimension level names -> one dimensional variables that will be used
               as weights for this level
            on_group_0(str, optional): how to handle group sums of 0: must be one of

                * 'error': the default, raises an Error
                * 'nan': sets resulting weights to nan.
                   WARNING: this can lead to a loss of values on disaggregation
                * 'equal': sets resulting weights to 1/n
        """

        if not isinstance(domain, Domain):
            domain = Domain(domain)
        level_weights = level_weights or {}

        result = self
        dim_steps = self.get_transform_steps(
            domain, level_weights=level_weights, on_group_0=on_group_0
        )
        for dim, steps in dim_steps.items():
            dim_name = dim.name

            for (_from_level, to_level, action, weight) in steps:
                if action == "expand":
                    result = result.expand(dim)
                elif action == "squeeze":
                    result = result.squeeze(dim_name)
                elif action == "aggregate":
                    result = result.aggregate(dim_name, weights=weight)
                elif action == "disaggregate":
                    result = result.disaggregate(
                        dim_name, to_level.name, weights=weight
                    )
                elif action == "keep":
                    # do nothing
                    pass
                else:
                    raise NotImplementedError(action)

        result = result.reorder(domain.dimension_names)
        return result

    @property
    def is_intensive(self):
        return self._vartype == "intensive"

    @property
    def is_extensive(self):
        return self._vartype == "extensive"

    @property
    def is_weight(self):
        return self._vartype in ("weight", "weight_w_nan")

    @property
    def is_weight_w_nan(self):
        """A weight where group sums can also be nan (partially defined)"""
        return self._vartype == "weight_w_nan"

    @property
    def is_scalar(self):
        return self._domain.is_scalar


class ExtensiveVariable(Variable):
    """Shorthand for `Variable(vartype="extensive")`

    Args:
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

    def __init__(self, data, domain, unit=None):
        super().__init__(data=data, unit=unit, domain=domain, vartype="extensive")


class IntensiveVariable(Variable):
    """Shorthand for `Variable(vartype="intensive")`

    Args:
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

    def __init__(self, data, domain, unit=None):
        super().__init__(data=data, unit=unit, domain=domain, vartype="intensive")


class Weight(Variable):
    """Shorthand for `Variable(vartype="weight")`

    In addition, weights are special Variables with:

    * domain is exactly one dimension
    * values in each group add up to 1.0

    Args:
        data: data can be passed in different ways:

          * dictionary of key -> value,
            where key is a tuple that is an element of the domain
          * list of records. records must have fields for each
            dimension in the domain and a column named `value`
          * pandas Series with a MultiIndex

        dimension_level(DimensionLevel): domain of data (weights have only one)
        unit(optional): not implemented yet

    """

    def __init__(self, data, dimension_level):
        super().__init__(
            data=data, unit=None, domain=[dimension_level], vartype="weight"
        )


class ExtensiveScalar(ExtensiveVariable):
    """Shorthand for `ExtensiveVariable(domain=None)`

    Args:
        value: number
        unit(optional): not implemented yet

    """

    def __init__(self, value, unit=None):
        super().__init__(data=value, unit=unit, domain=[])


class IntensiveScalar(IntensiveVariable):
    """Shorthand for `IntensiveVariable(domain=None)`

    Args:
        value: number
        unit(optional): not implemented yet

    """

    def __init__(self, value, unit=None):
        super().__init__(data=value, unit=unit, domain=[])


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)

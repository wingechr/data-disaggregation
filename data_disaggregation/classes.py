import logging  # noqa
from enum import Enum
from itertools import product

import numpy as np
import pint

ureg = pint.UnitRegistry()


def get_unit(x):
    if not x:
        return ureg.dimensionless
    return getattr(ureg, str(x))


DIMENSION_ROOT_ELEMENT = None


class TransformName(Enum):
    FROM = "FROM"
    TO = "TO"
    SQUEEZE = "SQUEEZE"
    EXPAND = "EXPAND"


class FrozenMap:
    """immutable, ordered map of unique hashables mapping to variables of a type"""

    __slots__ = ["__values", "__indices", "__keys"]

    def __init__(self, items, value_class):

        self.__indices = {}  # key -> idx

        keys = []
        values = []

        for idx, (key, val) in enumerate(items):
            if not isinstance(val, value_class):  # ensure type of values
                raise TypeError(f"{val} is not of type {value_class}")
            if key in self.__indices:  # ensure uniqueness
                raise KeyError(f"{key} not unique")
            self.__indices[key] = idx
            keys.append(key)
            values.append(val)

        self.__keys = tuple(keys)
        self.__values = tuple(values)

    def __len__(self):
        return len(self.__keys)

    def __getitem__(self, key):
        idx = self.__indices[key]
        return self.__values[idx]

    def __contains__(self, key):
        return key in self.__indices

    def keys(self):
        # return self.__data.keys()
        return self.__keys

    def values(self):
        return self.__values

    def items(self):
        return zip(self.__keys, self.__values)

    def index(self, key):
        return self.__indices[key]


class UniquelyNamedFrozenMap(FrozenMap):
    __slots__ = ["__name"]

    __instances = {}  # name -> instance

    def __init__(self, name, items, value_class):
        # elements: map elem -> index
        self.__name = name
        if name in self.__instances:
            raise KeyError(f"{name} is not a unique name")
        self.__instances[self.name] = self

        super().__init__(items, value_class)

    @property
    def name(self):
        return self.__name

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class DimensionLevel(UniquelyNamedFrozenMap):
    __slots__ = ["__dimension"]

    def __init__(self, name, dimension, elements):
        elements = ((e, i) for i, e in enumerate(elements))

        super().__init__(name, elements, int)
        # elements: map elem -> index
        if not isinstance(dimension, Dimension):
            raise TypeError(f"{dimension} is not of type Dimension")
        self.__dimension = dimension

    def __str__(self):
        return f"DimensionLevel({self.dimension.name}={self.name})"

    @property
    def dimension(self):
        return self.__dimension

    @property
    def size(self):
        return len(self)

    def alias(self, name):
        return Alias(name, self)

    def aliasFrom(self):
        if self.is_dimension_root:
            return Alias(TransformName.EXPAND, self)
        else:
            return Alias(TransformName.FROM, self)

    def aliasTo(self):
        if self.is_dimension_root:
            return Alias(TransformName.SQUEEZE, self)
        else:
            return Alias(TransformName.TO, self)

    @property
    def is_dimension_root(self):
        return self == self.__dimension


class Dimension(DimensionLevel):
    def __init__(self, name):
        super().__init__(name=name, dimension=self, elements=[DIMENSION_ROOT_ELEMENT])


class Alias:
    __slots__ = ["__name", "__reference"]

    def __init__(self, name, reference):
        self.__name = name
        self.__reference = reference

    @property
    def name(self):
        return self.__name

    @property
    def reference(self):
        return self.__reference


def parse_dimension_levels(dimension_levels):
    if isinstance(dimension_levels, (DimensionLevel, Alias)):
        # only one passed
        dimension_levels = [dimension_levels]
    elif dimension_levels is None:  # scalar
        dimension_levels = []
    elif isinstance(dimension_levels, (dict, Domain)):
        dimension_levels = dimension_levels.items()
    for x in dimension_levels:
        if isinstance(x, Alias):
            yield x.name, x.reference
        elif isinstance(x, DimensionLevel):
            yield x.dimension.name, x
        else:
            yield x


class Domain(FrozenMap):
    __slots__ = ["__indices"]

    def __init__(self, dimension_levels):
        dimension_levels = parse_dimension_levels(dimension_levels)
        super().__init__(dimension_levels, DimensionLevel)

        # generate index mappings
        if self.size == 0:
            self.__indices = FrozenMap([], int)
        elif self.size == 1:
            self.__indices = FrozenMap(self.values()[0].items(), int)
        else:
            self.__indices = FrozenMap(
                zip(
                    product(*[d.keys() for d in self.values()]),
                    product(*[d.values() for d in self.values()]),
                ),
                tuple,
            )

    def __eq__(self, other):
        return tuple(self.values()) == tuple(other.values())

    def __str__(self):
        return f"Domain({list(self.keys())})"

    @property
    def shape(self):
        return tuple(d.size for d in self.values())

    @property
    def size(self):
        return len(self)

    @property
    def indices(self):
        return self.__indices

    @classmethod
    def as_domain(cls, x):
        if isinstance(x, Domain):
            return x
        else:
            return Domain(x)


class TransformDomain(Domain):
    __slots__ = ["__dimension_name"]

    def __init__(self, dimension_levels, dimension_name=None):
        dimension_levels = list(parse_dimension_levels(dimension_levels))
        dimension_levels = list(x[1] for x in dimension_levels)

        if len(dimension_levels) != 2:
            raise TypeError("must have 2 dims")
        dim, dim2 = [d.dimension for d in dimension_levels]
        if dim != dim2:
            raise TypeError("must be in same dimension")
        self.__dimension_name = dimension_name or dim.name
        dimension_levels = (
            dimension_levels[0].aliasFrom(),
            dimension_levels[1].aliasTo(),
        )
        super().__init__(dimension_levels)

    @property
    def dimension_name(self):
        return self.__dimension_name

    @property
    def dimension(self):
        return self.values()[0]  # both are the same


def assertEqual(items1, items2):
    list1 = list(items1)
    list2 = list(items2)
    set1 = set(list1)
    set2 = set(list2)
    if len(set1) != len(list1):
        raise KeyError(f"Duplicate elements in first list: {list1}")
    if len(set2) != len(list2):
        raise KeyError(f"Duplicate elements in second list: {list2}")
    if set2 - set1:
        raise KeyError(f"Missing elements in first list: {set2 - set1}")
    if set1 - set2:
        raise KeyError(f"Missing elements in second list: {set1 - set2}")


class Variable:

    __slots__ = [
        "__data",
        "__domain",
        "__unit",
        "__is_extensive",
    ]

    def __init__(self, data, domain, unit=None, is_extensive=False):
        self.__domain = Domain.as_domain(domain)  # must be first
        self.__data = self.__parse_data(data)
        self.__unit = get_unit(unit)
        self.__is_extensive = bool(is_extensive)

    def __str__(self):
        return f"Variable({self.domain})"

    def __parse_data(self, data):
        if isinstance(data, np.ndarray):
            if self.shape != data.shape:
                raise TypeError(
                    f"wrong data shape, expected {self.shape}, got {data.shape}"
                )
            result = data.copy()
        elif self.domain.size == 0:  # scalar
            result = np.array(data)  # data must be single value, shape is ()
            assert result.shape == self.shape
        else:  # assume data is dict like structure (like pandas Series)
            result = np.zeros(shape=self.shape, dtype=np.dtype(float))
            for key, val in data.items():
                idx = self.domain.indices[key]
                result[idx] = val
        result = result.astype(np.dtype(float))
        result = self._validate_data(result)
        return result

    def _validate_data(self, data):
        return data

    @property
    def domain(self):
        return self.__domain

    @property
    def data(self):
        return self.__data

    @property
    def shape(self):
        return self.domain.shape

    @property
    def unit(self):
        return self.__unit

    @property
    def is_extensive(self):
        return self.__is_extensive

    def transpose(self, dimension_names):
        indices = [self.domain.index(n) for n in dimension_names]

        assertEqual(dimension_names, self.domain.keys())
        dimension_levels = [(n, self.domain[n]) for n in dimension_names]
        domain = Domain(dimension_levels)

        data = np.transpose(self.__data, axes=indices)

        return Variable(
            data,
            domain,
            self.unit,
            self.is_extensive,
        )

    def _squeeze(self):
        if not self.domain.size:
            raise Exception("can not squeeze scalar")

        last_dimension_level = self.domain.values()[self.domain.size - 1]
        if not isinstance(last_dimension_level, Dimension):
            raise Exception("can only remove last dimension if fully aggregated")
        dimension_levels = list(self.domain.items())
        domain = Domain(dimension_levels[:-1])

        index = self.domain.size - 1
        data = np.squeeze(self.data, axis=index)

        return Variable(
            data,
            domain,
            self.unit,
            self.is_extensive,
        )

    def _expand(self, dimension):

        if not (
            isinstance(dimension, Dimension)
            or (
                isinstance(dimension, Alias),
                isinstance(dimension.reference, Dimension),
            )
        ):
            raise Exception("can only add root level dimension")
        dimension_levels = list(self.domain.items())
        dimension_levels.append(dimension)
        domain = Domain(dimension_levels)

        index = self.domain.size
        data = np.expand_dims(self.data, axis=index)

        return Variable(
            data,
            domain,
            self.unit,
            self.is_extensive,
        )

    def items(self):
        data = self.__data.flatten("C")
        # TODO: fully test that flatten works the same in all versions
        keys = self.domain.indices.keys()
        # alternatively: return [(k, self.__data[i]) for k, i in self.domain.indices.items()] # noqa
        return zip(keys, data)

    def records(self, value_column="value"):
        names = self.domain.keys()
        if len(names) == 1:

            def create_record(key):
                return {names[0]: key}

        else:

            def create_record(key):
                return dict(zip(names, key))

        for key, val in self.items():
            rec = create_record(key)
            rec[value_column] = val
            yield rec

    def transform(self, other):

        """
        TODO:
        transformvariable must fullfill special conditions
        * must be normalized for extensive/intensive
        * must be 2d in same dimension
        * first level name: FROM or EXPAND
        * second level name: TO or SQUEEZE

        * if dim_name/alias not in self: expand
        * if second level is root and empty level name : squeeze

        """
        if isinstance(other, Variable):

            if not isinstance(other.domain, TransformDomain):
                raise TypeError("Dom2 must be of type TransformDomain")

            result = self
            o_first_dim_name, o_second_dim_name = other.domain.keys()
            o_first_dim_level, o_second_dim_level = other.domain.values()

            if isinstance(result.domain, TransformDomain):  # combine transformation
                pass
            else:  # normal
                if o_first_dim_name == TransformName.EXPAND:
                    result = result._expand(
                        o_first_dim_level.alias(other.domain.dimension_name)
                    )
                # make sure dimension is last
                dimension_names = list(result.domain.keys())
                index = dimension_names.index(other.domain.dimension_name)
                if index != result.domain.size - 1:
                    # move to end
                    dimension_names.append(dimension_names.pop(index))
                    result = result.transpose(dimension_names)

            if not result.domain.size:
                raise Exception("Dom1 cannot be scalar")

            r_last_dim_level = result.domain.values()[-1]

            if r_last_dim_level != o_first_dim_level:
                raise Exception(
                    f"Last dimension level of Dom1({r_last_dim_level}) must match first dimension level of Dom2({o_first_dim_level})"  # noqa
                )  # noqa

            dimension_levels_1 = list(result.domain.items())

            if isinstance(self.domain, TransformDomain):
                dimension_levels = dimension_levels_1[:-1] + [o_second_dim_level]
                domain = TransformDomain(dimension_levels, self.domain.dimension_name)
            else:
                dimension_levels = dimension_levels_1[:-1] + [
                    o_second_dim_level.alias(other.domain.dimension_name)
                ]
                domain = Domain(dimension_levels)

            data = np.matmul(result.data, other.data)

            unit = result.unit * other.unit

            result = Variable(
                data,
                domain,
                unit,
                result.is_extensive,
            )

            if (
                not isinstance(self.domain, TransformDomain)
                and o_second_dim_name == TransformName.SQUEEZE
            ):
                result = result._squeeze()

            return result
        else:
            raise NotImplementedError(f"Variable + {other.__class__.name}")

    def normalize_transform(self, for_extensive=True, dimension_name=None):
        if not self.is_extensive:
            raise TypeError("must be extensive")
        domain = TransformDomain(self.domain, dimension_name=dimension_name)

        data = self.data
        transposed = not (for_extensive)
        if transposed:
            data = data.transpose()
        rows, cols = data.shape
        sums = np.sum(data, axis=1).reshape((rows, 1))
        if (sums == 0).any():
            raise ValueError("sum = 0")
        sums = np.repeat(sums, cols, axis=1)
        data = data / sums
        if transposed:
            data = data.transpose()
        # result has no unit and is NOT extensive

        return Variable(data, domain, None, False)

    def to_unit(self, unit):
        unit = get_unit(unit)
        return Variable(
            unit.from_(self.data * self.unit).magnitude,
            self.domain,
            unit,
            self.is_extensive,
        )

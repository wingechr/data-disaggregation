from collections import OrderedDict
from itertools import product

import numpy as np


class FrozenMap:
    """immutable, ordered map of unique hashables mapping to variables of a type"""

    __slots__ = ["__data"]

    def __init__(self, items, value_class):
        self.__data = OrderedDict()  # immutable: only change in __init__
        for key, val in items:
            if not isinstance(val, value_class):  # ensure type of values
                raise TypeError(f"{val} is not of type {value_class}")
            if key in self.__data:  # ensure uniqueness
                raise KeyError(f"{key} not unique")
            self.__data[key] = val  # ensure hashable

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, key):
        return self.__data[key]

    def __contains__(self, key):
        return key in self.__data

    def keys(self):
        return self.__data.keys()

    def values(self):
        return self.__data.values()

    def items(self):
        return self.__data.items()


class UniquelyNamedFrozenMap(FrozenMap):
    __slots__ = ["__name"]

    __instances = {}  # name -> instance

    def __init__(self, name, items, value_class):
        # elements: map elem -> index
        if name in self.__instances:
            raise KeyError(f"{name} is not a unique name")
        self.__name = name
        self.__instances[self.name] = self

        super().__init__(items, value_class)

    @property
    def name(self):
        return self.__name


class DimensionLevel(UniquelyNamedFrozenMap):
    __slots__ = ["__dimension"]

    def __init__(self, name, dimension, elements):
        elements = ((e, i) for i, e in enumerate(elements))

        super().__init__(name, elements, int)
        # elements: map elem -> index
        if not isinstance(dimension, Dimension):
            raise TypeError(f"{dimension} is not of type Dimension")
        self.__dimension = dimension

        # register
        # self.dimension.add_dimension_level(self)

    @property
    def dimension(self):
        return self.__dimension

    @property
    def size(self):
        return len(self)

    # def add_dimension_level(self, dimension_level):
    #    if not isinstance(dimension_level, DimensionLevel):
    #        raise TypeError(f"{dimension_level} is not of type DimensionLevel")


class Dimension(DimensionLevel):
    pass

    def __init__(self, name):
        super().__init__(name=name, dimension=self, elements=[])


def parse_dimension_levels(dimension_levels):
    if isinstance(dimension_levels, DimensionLevel):
        # only one passed
        dimension_levels = [dimension_levels]
    elif dimension_levels is None:  # scalar
        dimension_levels = []
    for x in dimension_levels:
        if isinstance(x, DimensionLevel):
            yield x.name, x
        else:  # assume it's a dictionary
            yield x, dimension_levels[x]


class Domain(FrozenMap):
    __slots__ = ["__indices"]

    def __init__(self, dimension_levels):
        dimension_levels = parse_dimension_levels(dimension_levels)
        super().__init__(dimension_levels, DimensionLevel)

        # generate index mappings
        if self.size == 0:
            self.__indices = FrozenMap([], int)
        elif self.size == 1:
            self.__indices = FrozenMap(self.dimension_levels[0].items(), int)
        else:
            self.__indices = FrozenMap(
                zip(
                    product(*[d.keys() for d in self.dimension_levels]),
                    product(*[d.values() for d in self.dimension_levels]),
                ),
                tuple,
            )

    @property
    def shape(self):
        return tuple(d.size for d in self.dimension_levels)

    @property
    def size(self):
        return len(self)

    @property
    def indices(self):
        return self.__indices

    @property
    def dimension_levels(self):
        return tuple(self.values())

    @classmethod
    def as_domain(cls, x):
        if isinstance(x, Domain):
            return x
        else:
            return Domain(x)


class VarType:
    __slots__ = []

    @classmethod
    def as_vartype(cls, x):
        if isinstance(x, VarType):
            return x
        else:
            return Domain(x)


class Variable:

    __slots__ = [
        "__data",
        "__domain",
        "__vartype",
    ]

    def __init__(self, data, domain, vartype):
        self.__domain = Domain.as_domain(domain)
        self.__vartype = VarType.as_vartype(vartype)
        self.__data = self.__parse_data(data)

    def __parse_data(self, data):
        if isinstance(data, np.ndarray):
            if self.shape != data.shape:
                raise TypeError(
                    f"wrong data shape, expected {self.shape}, got {data.shape}"
                )
            result = data.astype(np.dtype(float), copy=True)
        elif self.domain.size == 0:  # scalar
            result = np.array(data)  # data must be single value, shape is ()
            assert result.shape == self.shape
        else:  # assume data is dict like structure (like pandas Series)
            result = np.zeros(shape=self.shape, dtype=np.dtype(float))
            for key, val in data.items():
                idx = self.domain.indices[key]
                result[idx] = val
        result = self._validate_data(result)
        return result

    def _validate_data(self, data):
        # no n/a
        if np.isnan(data).any():
            raise ValueError("cannot have na")
        return data

    @property
    def domain(self):
        return self.__domain

    @property
    def shape(self):
        return self.domain.shape

    @property
    def vartype(self):
        return self.__vartype


class PosVariable(Variable):
    def _validate_data(self, data):
        data = super()._validate_data(data)
        if (data < 0).any():
            raise ValueError("cannot have negative values")
        return data


class BoolVariable(PosVariable):
    def _validate_data(self, data):
        data = super()._validate_data(data)
        if not (np.isin(data, (0, 1))).all():
            raise ValueError("values must be 0 or 1")
        return data

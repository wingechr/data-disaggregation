import logging  # noqa
from itertools import product

import numpy as np

DIMENSION_ROOT_ELEMENT = None


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
        if name in self.__instances:
            raise KeyError(f"{name} is not a unique name")
        self.__name = name
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

    @property
    def dimension(self):
        return self.__dimension

    @property
    def size(self):
        return len(self)

    def alias(self, name):
        return Alias(name, self)


class Dimension(DimensionLevel):
    pass

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
    if isinstance(dimension_levels, DimensionLevel):
        # only one passed
        dimension_levels = [dimension_levels]
    elif dimension_levels is None:  # scalar
        dimension_levels = []
    elif isinstance(dimension_levels, dict):
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
            self.__indices = FrozenMap(self.dimension_levels[0].items(), int)
        else:
            self.__indices = FrozenMap(
                zip(
                    product(*[d.keys() for d in self.dimension_levels]),
                    product(*[d.values() for d in self.dimension_levels]),
                ),
                tuple,
            )

    def __eq__(self, other):
        return tuple(self.values()) == tuple(other.values())

    def __str__(self):
        return f"Domain({list(self.keys())})"

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

    def transpose(self, dimension_names):
        assertEqual(dimension_names, self.keys())
        dimension_levels = [(n, self[n]) for n in dimension_names]
        return Domain(dimension_levels)

    def squeeze(self):
        if not self.size:
            raise Exception("can not squeeze scalar")
        last_dimension_level = self.values()[self.size - 1]
        if not isinstance(last_dimension_level, Dimension):
            raise Exception("can only remove last dimension if fully aggregated")
        dimension_levels = list(self.items())
        return Domain(dimension_levels[:-1])

    def expand(self, dimension):
        if not (
            isinstance(dimension, Dimension)
            or (
                isinstance(dimension, Alias),
                isinstance(dimension.reference, Dimension),
            )
        ):
            raise Exception("can only add root level dimension")
        dimension_levels = list(self.items())
        dimension_levels.append(dimension)
        return Domain(dimension_levels)


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


class VarType:
    __slots__ = []

    @classmethod
    def as_vartype(cls, x):
        if isinstance(x, VarType):
            return x
        else:
            return VarType(x)

    def __init__(self, *arg, **kwargs):
        pass


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
    def shape(self):
        return self.domain.shape

    @property
    def vartype(self):
        return self.__vartype

    def __new_domain(self, data, domain):
        return Variable(
            data,
            domain,
            self.__vartype,
        )

    def transpose(self, dimension_names):
        indices = [self.__domain.index(n) for n in dimension_names]
        return self.__new_domain(
            np.transpose(self.__data, axes=indices),
            self.__domain.transpose(dimension_names),
        )

    def squeeze(self):
        index = self.domain.size - 1
        return self.__new_domain(
            np.squeeze(self.__data, axis=index), self.__domain.squeeze()
        )

    def expand(self, dimension):
        index = self.domain.size
        return self.__new_domain(
            np.expand_dims(self.__data, axis=index),
            self.__domain.expand(dimension),
        )

    def items(self):
        data = self.__data.flatten("C")
        # TODO: fully test that flatten works the same in all versions
        keys = self.domain.indices.keys()
        # alternatively: return [(k, self.__data[i]) for k, i in self.domain.indices.items()] # noqa
        return zip(keys, data)

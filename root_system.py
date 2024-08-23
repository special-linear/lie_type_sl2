# Absolutely minimal class for working with root systems
# Almost no guardrails

from itertools import chain, combinations, combinations_with_replacement
from more_itertools import distinct_permutations
from numbers import Number
from operator import neg, methodcaller, attrgetter
from functools import cached_property
from typing import Iterable, Self


class WeightSpaceElement:
    """
    Weight space elements (mostly roots) presented via coefficients in a simple roots basis.
    No check for the sizes compatibility is done.
    """

    def __init__(self, coefficients: Iterable[Number]) -> None:
        self.coefficients = tuple(coefficients)

    def __iter__(self):
        return iter(self.coefficients)

    def __eq__(self, other):
        return all(x == y for x, y in zip(self, other))

    def __bool__(self):
        return any(self.coefficients)

    def __hash__(self):
        return hash(self.coefficients)

    def __str__(self):
        return '({})'.format(','.join(map(str, self)))

    def __add__(self, other: Self) -> Self:
        return WeightSpaceElement(x + y for x, y in zip(self, other))

    def __neg__(self) -> Self:
        return WeightSpaceElement(-x for x in self)

    def __sub__(self, other: Self) -> Self:
        return WeightSpaceElement(x - y for x, y in zip(self, other))

    def __mul__(self, other: Number) -> Self:
        if isinstance(other, Number):
            return WeightSpaceElement(x * other for x in self)
        else:
            return NotImplemented

    def __rmul__(self, other: Number):
        return self * other

    @cached_property
    def ht(self) -> Number:
        return sum(self)

    @cached_property
    def is_positive(self) -> bool:
        return all(x >= 0 for x in self)


class RootSystem:
    """
    Root systems with root lists generators.
    No checks for ranks parameters when creating an instance.
    """

    def __init__(self, series: str, rank: int):
        self.series = series
        self.rank = rank
        roots_generators = {
            'A': self._generate_roots_type_a,
            'B': self._generate_roots_type_b,
            'C': self._generate_roots_type_c,
            'D': self._generate_roots_type_d,
            'E': self._generate_roots_type_e,
            'F': self._generate_roots_type_f,
            'G': self._generate_roots_type_g
        }
        self._positive_roots = frozenset(map(WeightSpaceElement, roots_generators[series](rank)))
        gram_matrix_generators = {
            'A': self._gram_matrix_type_a,
            'B': self._gram_matrix_type_b,
            'C': self._gram_matrix_type_c,
            'D': self._gram_matrix_type_d,
            'E': self._gram_matrix_type_e,
            'F': self._gram_matrix_type_f,
            'G': self._gram_matrix_type_g
        }
        self._gram_matrix = tuple(map(tuple, gram_matrix_generators[series](rank)))

    def __contains__(self, item: WeightSpaceElement) -> bool:
        return item in self._positive_roots or -item in self._positive_roots

    def simple_roots(self) -> Iterable[WeightSpaceElement]:
        return (WeightSpaceElement(int(i == k) for i in range(self.rank)) for k in range(self.rank))

    @cached_property
    def zero_weight(self) -> WeightSpaceElement:
        return WeightSpaceElement([0] * self.rank)

    def positive_roots(self) -> Iterable[WeightSpaceElement]:
        return iter(self._positive_roots)

    def all_roots(self) -> Iterable[WeightSpaceElement]:
        return chain(self.positive_roots(), map(neg, self.positive_roots()))

    @cached_property
    def maximal_root(self) -> WeightSpaceElement:
        return max(self._positive_roots, key=attrgetter('ht'))

    def scalar_product(self, u, v):
        bv = [sum(bij * vj for bij, vj in zip(bi, v)) for bi in self._gram_matrix]
        return sum(ui * bvi for ui, bvi in zip(u, bv))

    @staticmethod
    def _generate_roots_type_a(rank: int) -> Iterable[Iterable[int]]:
        if rank >= 1:
            return (chain([0] * i, [1] * (j - i), [0] * (rank - j)) for i, j in
                    combinations(range(rank + 1), r=2))
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _gram_matrix_type_a(rank: int) -> Iterable[Iterable[Number]]:
        if rank >= 1:
            return ((2 if i == j else -1 if abs(i - j) == 1 else 0 for i in range(rank)) for j in range(rank))
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _generate_roots_type_b(rank: int) -> Iterable[Iterable[int]]:
        if rank >= 2:
            return chain(
                (chain([0] * i, [1] * (j - i), [0] * (rank - j))
                 for i, j in combinations(range(rank + 1), r=2)),
                (chain([0] * i, [1] * (j - i), [2] * (rank - j))
                 for i, j in combinations(range(rank), r=2))
            )
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _gram_matrix_type_b(rank: int) -> Iterable[Iterable[Number]]:
        if rank >= 2:
            gram_matrix = [[2 if i == j else -1 if abs(i - j) == 1 else 0 for i in range(rank)] for j in range(rank)]
            gram_matrix[rank - 1][rank - 1] = 1
            return gram_matrix
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _generate_roots_type_c(rank: int) -> Iterable[Iterable[int]]:
        if rank >= 2:
            return chain(
                (chain([0] * i, [1] * (j - i), [0] * (rank - j))
                 for i, j in combinations(range(rank + 1), r=2)),
                (chain([0] * i, [1] * (j - i), [2] * (rank - 1 - j), [1])
                 for i, j in combinations_with_replacement(range(rank - 1), r=2))
            )
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _gram_matrix_type_c(rank: int) -> Iterable[Iterable[Number]]:
        if rank >= 2:
            gram_matrix = [[2 if i == j else -1 if abs(i - j) == 1 else 0 for i in range(rank)] for j in range(rank)]
            gram_matrix[rank - 1][rank - 1] = 4
            gram_matrix[rank - 1][rank - 2], gram_matrix[rank - 2][rank - 1] = -2, -2
            return gram_matrix
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _generate_roots_type_d(rank: int) -> Iterable[Iterable[int]]:
        if rank >= 4:
            return chain(
                (chain([0] * i, [1] * (j - i), [0] * (rank - j))
                 for i, j in combinations(range(rank), r=2)),
                (chain([0] * (i - 1), [1] * (rank - i - 1), [0, 1]) for i in range(1, rank)),
                (chain([0] * i, [1] * (j - i), [2] * (rank - j - 2), [1, 1])
                 for i, j in combinations(range(rank - 1), r=2))
            )
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _gram_matrix_type_d(rank: int) -> Iterable[Iterable[Number]]:
        if rank >= 4:
            gram_matrix = [
                [2 if i == j else -1 if i < rank - 1 and j < rank - 1 and abs(i - j) == 1 else 0 for i in range(rank)]
                for j in range(rank)]
            gram_matrix[rank-1][rank-3], gram_matrix[rank-3][rank-1] = -1, -1
            return gram_matrix
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _generate_roots_type_e(rank: int) -> Iterable[Iterable[int]]:
        if rank in (6, 7, 8):
            # transformation from x-tuples to root basis coordinates
            def transform(x):
                d = sum(x) // 3
                m = [d] + [i * d - sum(x[:i]) for i in range(1, 3)] + [sum(x[i:]) for i in range(3, rank)]
                m[0], m[1] = m[1], m[0]
                return m
            # x-tuples for positive roots of A_{rank-1} subsystem
            a_xs = (tuple(chain([0] * i, [-1], [0] * (j - i), [1], [0] * (rank - 2 - j)))
                    for i, j in combinations_with_replacement(range(rank - 1), r=2))
            # x-tuples representatives for positive roots not in the A subsystem
            e_xs_representatives = [
                [1] * 3 + [0] * (rank - 3),
                [1] * 6 + [0] * (rank - 6)
            ]
            if rank == 8:
                e_xs_representatives.append([2] + [1] * 7)
            e_xs = chain.from_iterable(map(distinct_permutations, e_xs_representatives))
            return map(transform, chain(a_xs, e_xs))
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _gram_matrix_type_e(rank: int) -> Iterable[Iterable[Number]]:
        if rank in (6, 7, 8):
            gram_matrix = [
                [2 if i == j else -1 if i != 1 and j != 1 and abs(i - j) == 1 else 0 for i in range(rank)]
                for j in range(rank)]
            gram_matrix[0][2], gram_matrix[2][0], gram_matrix[1][3], gram_matrix[3][1] = -1, -1, -1, -1
            return gram_matrix
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _generate_roots_type_f(rank: int) -> Iterable[Iterable[int]]:
        if rank == 4:
            root_strings = ('1000 0100 0010 0001 1100 0110 '
                            '0011 1110 0120 0111 1120 1111 '
                            '0121 1220 1121 0122 1221 1122 '
                            '1231 1222 1232 1242 1342 2342')
            return map(lambda root_str: map(int, root_str), root_strings.split())
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _gram_matrix_type_f(rank: int) -> Iterable[Iterable[Number]]:
        if rank == 4:
            return (
                (4, -2, 0, 0),
                (-2, 4, -2, 0),
                (0, -2, 2, -1),
                (0, 0, -1, 2)
            )
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _generate_roots_type_g(rank: int) -> Iterable[Iterable[int]]:
        if rank == 2:
            return [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)]
        else:
            raise ValueError('Incorrect rank parameter.')

    @staticmethod
    def _gram_matrix_type_g(rank: int) -> Iterable[Iterable[Number]]:
        if rank == 2:
            return (
                (2, -3),
                (-3, 6)
            )
        else:
            raise ValueError('Incorrect rank parameter.')

from typing import Container

Boolean = bool
Integer = int
IntegerTuple = tuple[Integer, Integer]
Numerical = Integer | IntegerTuple
IntegerSet = frozenset[Integer]
Grid = tuple[tuple[Integer]]
Cell = tuple[Integer, IntegerTuple]
Object = frozenset[Cell]
Objects = frozenset[Object]
Indices = frozenset[IntegerTuple]
IndicesSet = frozenset[Indices]
Patch = Object | Indices
Element = Object | Grid
Piece = Grid | Patch
TupleTuple = tuple[tuple]
ContainerContainer = Container[Container]

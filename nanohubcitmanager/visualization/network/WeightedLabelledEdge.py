from __future__ import annotations

from .Edge import Edge
from .Vertex import Vertex


class WeightedLabelledEdge(Edge):
    def __init__(self, v1: Vertex, v2: Vertex):
        super().__init__(v1, v2)
        self.label = ""

    def __hash__(self):
        return hash((self.id, self.label))

    def __eq__(self, other):
        if not isinstance(other, WeightedLabelledEdge):
            return False
        return self.id == other.id and self.label == other.label

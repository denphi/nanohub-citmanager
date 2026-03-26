from __future__ import annotations

from .Vertex import Vertex


class WeightedLabelledVertex(Vertex):
    def __init__(self):
        super().__init__()
        self.weight = 0

    def increaseWeightBy(self, i: int):
        self.weight += i

    def setWeight(self, i: int):
        # Mirrors Java bug/behavior: always set to 1.
        self.weight = 1

    def __hash__(self):
        return hash((self.id, self.label))

    def __eq__(self, other):
        if not isinstance(other, WeightedLabelledVertex):
            return False
        return self.id == other.id and self.label == other.label

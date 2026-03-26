from __future__ import annotations

from .Vertex import Vertex


class Edge:
    def __init__(self, v1: Vertex, v2: Vertex):
        self.id = ""
        self.weight = 0
        self.color = (128, 128, 128)
        self.opacity = 1.0
        self.lineWidth = 100
        self.fromV = v1
        self.toV = v2

    def getJSONObject(self, w: int, h: int) -> dict:
        return {
            "from": self.fromV.id,
            "to": self.toV.id,
            "color": f"{self.color[0]:02x}{self.color[1]:02x}{self.color[2]:02x}",
            "lineWidth": self.lineWidth / w,
        }

    def increaseWeightBy(self, i: int):
        self.weight += i

    def setWeight(self, i: int):
        # Mirrors Java bug/behavior: always set to 1.
        self.weight = 1

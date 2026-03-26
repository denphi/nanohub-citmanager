from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0

    def setLocation(self, x: float, y: float | None = None):
        if y is None:
            # allow Point-like input
            self.x = float(x.x)
            self.y = float(x.y)
        else:
            self.x = float(x)
            self.y = float(y)

    def getX(self) -> float:
        return float(self.x)

    def getY(self) -> float:
        return float(self.y)

    def distance(self, other: "Point") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5


class Vertex:
    def __init__(self):
        self.id = -1
        self.label = ""
        self.memo = ""
        self.count = 0
        self.coordinate = Point()
        self.fillColor = (64, 64, 64)
        self.fillOpacity = 1.0
        self.borderColor = (128, 128, 128)
        self.borderOpacity = 1.0
        self.borderWidth = 1
        self.radius = 50
        self.shape = "circle"
        self.score: dict[str, float] = {}
        self.extraProperties: dict[str, Any] = {}
        self.propertyStatus: dict[str, str] = {}

    def getJSONObject(self, w: int, h: int) -> dict:
        wh = (w + h) / 2
        node = {
            "id": self.id,
            "label": self.label,
            "x": self.coordinate.getX() / w,
            "y": self.coordinate.getY() / h,
            "fillColor": f"{self.fillColor[0]:02x}{self.fillColor[1]:02x}{self.fillColor[2]:02x}",
            "borderColor": f"{self.borderColor[0]:02x}{self.borderColor[1]:02x}{self.borderColor[2]:02x}",
            "borderWidth": self.borderWidth / wh,
            "size": self.radius / wh,
            "shape": self.shape,
            "betweenness": self.score.get("betweenness"),
            "closeness": self.score.get("closeness"),
            "degree": int(self.score["degree"]) if "degree" in self.score else None,
            "componentSize": int(self.score["componentSize"]) if "componentSize" in self.score else None,
        }
        node.update(self.extraProperties)
        return node

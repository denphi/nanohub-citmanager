from __future__ import annotations

from typing import List

from .Circle import Circle


class CirclePacking:
    def __init__(self):
        self.circles: List[Circle] = []

    def addCircle(self, c: Circle):
        self.circles.append(c)

    def getCircle(self, i: int) -> Circle:
        return self.circles[i]

    def layout(self, overlap: float):
        while self.occluded():
            # Build adjacency on occluded circles.
            adjacency: dict[int, set[int]] = {i: set() for i in range(len(self.circles))}
            for i in range(len(self.circles)):
                for n in range(i + 1, len(self.circles)):
                    ci = self.circles[i]
                    cn = self.circles[n]
                    if ci.occluded(cn):
                        adjacency[i].add(n)
                        adjacency[n].add(i)

            # Connected components.
            visited = set()
            for i in range(len(self.circles)):
                if i in visited:
                    continue
                stack = [i]
                comp: list[int] = []
                while stack:
                    v = stack.pop()
                    if v in visited:
                        continue
                    visited.add(v)
                    comp.append(v)
                    stack.extend(adjacency[v] - visited)
                if len(comp) >= 2:
                    self.expand(comp)

    def expand(self, vids: list[int]):
        accuX = 0.0
        accuY = 0.0
        minX = float("inf")
        maxX = float("-inf")
        minY = float("inf")
        maxY = float("-inf")

        for v in vids:
            c = self.circles[v]
            accuX += c.x
            accuY += c.y
            minX = min(minX, c.x)
            maxX = max(maxX, c.x)
            minY = min(minY, c.y)
            maxY = max(maxY, c.y)

        centerX = accuX / len(vids)
        centerY = accuY / len(vids)
        stepDist = ((maxX - minX) ** 2 + (maxY - minY) ** 2) ** 0.5 / 10.0

        maxIteration = 100
        i = 0
        while self.occludedInGroup(vids):
            for v in vids:
                c = self.circles[v]
                dx = c.x - centerX
                dy = c.y - centerY
                if abs(dx) < 1e-12:
                    arc = 1.5707963267948966 if dy > 0 else -1.5707963267948966
                else:
                    arc = __import__("math").atan(dy / dx)
                if dx < 0:
                    arc += __import__("math").pi
                c.x = c.x + (stepDist / c.r) * __import__("math").cos(arc)
                c.y = c.y + (stepDist / c.r) * __import__("math").sin(arc)
            i += 1
            if i > maxIteration:
                break

    def occludedInGroup(self, vids: list[int]) -> bool:
        for i in range(len(vids)):
            for n in range(i + 1, len(vids)):
                if self.circles[vids[i]].occluded(self.circles[vids[n]]):
                    return True
        return False

    def occluded(self) -> bool:
        for i in range(len(self.circles)):
            for n in range(i + 1, len(self.circles)):
                if self.circles[i].occluded(self.circles[n]):
                    return True
        return False

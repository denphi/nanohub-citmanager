class Circle:
    def __init__(self, x: float, y: float, r: float):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)

    def occluded(self, c: "Circle") -> bool:
        return ((c.x - self.x) ** 2 + (c.y - self.y) ** 2) ** 0.5 < (self.r + c.r)

    def __str__(self) -> str:
        return f"x={self.x} y={self.y} r={self.r}"

    def __eq__(self, other):
        return isinstance(other, Circle) and other.x == self.x and other.y == self.y and other.r == self.r

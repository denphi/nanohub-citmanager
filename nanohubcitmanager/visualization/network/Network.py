from __future__ import annotations

import math
import time
from typing import Any

import networkx as nx

from .Vertex import Vertex, Point
from .WeightedLabelledVertex import WeightedLabelledVertex
from .Edge import Edge
from .WeightedLabelledEdge import WeightedLabelledEdge


def _darker(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        max(0, int(c[0] * 0.7)),
        max(0, int(c[1] * 0.7)),
        max(0, int(c[2] * 0.7)),
    )


class Network:
    def __init__(self):
        self.g: nx.Graph = nx.Graph()
        self.currentGraph: nx.Graph = self.g
        self.verticesInfo: dict[int, Vertex] = {}
        self.edgesInfo: dict[str, Edge] = {}
        self.computationTime: dict[str, int] = {}
        self.stats: dict[str, float] = {}
        self.componentSizeCount: dict[int, int] = {}

        self.components: list[set[int]] | None = None
        self.canvasWidth = 100000
        self.canvasHeight = 100000
        self.startColorCode = 0
        self.basicNodeSize = 500.0

        self.colors: list[tuple[int, int, int]] = [(0, 0, 0)] * 32
        self.colors[0] = (0, 255, 0)
        self.colors[1] = (255, 255, 0)
        self.colors[2] = (255, 0, 255)
        self.colors[3] = (0, 255, 0)
        self.colors[4] = (0, 255, 255)
        self.colors[5] = (255, 0, 0)
        self.colors[6] = (0, 0, 0)
        self.colors[7] = (255, 0, 0)
        self.colors[8] = (255, 255, 0)
        self.colors[9] = (0, 255, 0)
        self.colors[10] = (0, 255, 0)
        self.colors[11] = (255, 255, 0)
        self.colors[12] = (255, 0, 0)
        self.colors[13] = (0, 255, 255)
        self.colors[14] = (128, 128, 128)
        self.colors[15] = (255, 255, 255)
        for i in range(16):
            self.colors[16 + i] = _darker(self.colors[i])

    def getColor(self, i: int) -> tuple[int, int, int]:
        return self.colors[i]

    def getGraph(self) -> nx.Graph:
        return self.g

    def getCurrentGraph(self) -> nx.Graph:
        return self.currentGraph

    def getVerticesInfo(self) -> dict[int, Vertex]:
        return self.verticesInfo

    def setVerticesInfo(self, vi: dict[int, Vertex]):
        self.verticesInfo = vi

    def getEdgesInfo(self) -> dict[str, Edge]:
        return self.edgesInfo

    def setGraph(self, g: nx.Graph):
        self.g = g
        self.currentGraph = self.g
        self.components = None

    def _get_components(self) -> list[set[int]]:
        if self.components is None:
            self.components = [set(c) for c in nx.connected_components(self.currentGraph)]
        return self.components

    def useLargestNetwork(self):
        if self.g.number_of_nodes() == 0:
            self.currentGraph = self.g
            self.components = None
            return
        largest = max(nx.connected_components(self.g), key=len)
        self.currentGraph = self.g.subgraph(largest).copy()
        self.components = None

    def computeNodeComponentSize(self):
        for comp in self._get_components():
            size = float(len(comp))
            for v in comp:
                if v in self.verticesInfo:
                    self.verticesInfo[v].score["componentSize"] = size

    def computeNetworkID(self):
        for i, comp in enumerate(self._get_components()):
            for v in comp:
                if v in self.verticesInfo:
                    self.verticesInfo[v].score["networkID"] = float(i)

    def computeBetweennessCentrality(self):
        bgn = time.time()
        scores = nx.betweenness_centrality(self.currentGraph)
        for v, s in scores.items():
            if v in self.verticesInfo:
                self.verticesInfo[v].score["betweenness"] = float(s)
        self.computationTime["betweenness"] = int((time.time() - bgn) * 1000)

    def computeClosenessCentrality(self):
        bgn = time.time()
        for comp in self._get_components():
            sub = self.currentGraph.subgraph(comp)
            scores = nx.closeness_centrality(sub)
            for v, s in scores.items():
                if v not in self.verticesInfo:
                    continue
                self.verticesInfo[v].score["closeness"] = -1.0 if math.isnan(s) else float(s)
        self.computationTime["closeness"] = int((time.time() - bgn) * 1000)

    def computeDegree(self):
        bgn = time.time()
        for v, deg in self.currentGraph.degree():
            if v in self.verticesInfo:
                self.verticesInfo[v].score["degree"] = float(deg)
        self.computationTime["degree"] = int((time.time() - bgn) * 1000)

    def computeComponentSize(self):
        sizeCnt: dict[int, int] = {}
        for comp in self._get_components():
            size = len(comp)
            sizeCnt[size] = sizeCnt.get(size, 0) + 1
        self.componentSizeCount = sizeCnt

    def _set_positions_from_layout(self, pos: dict[int, tuple[float, float]]):
        for v, (x, y) in pos.items():
            if v in self.verticesInfo:
                self.verticesInfo[v].coordinate = Point(float(x), float(y))

    def setLayout(self, layoutName: str, iteration: int, use_kamada_kawai: bool = False):
        canvas_w = float(self.canvasWidth)
        canvas_h = float(self.canvasHeight)

        if layoutName == "Kamada-Kawai":
            if self.currentGraph.number_of_nodes() == 0:
                return
            pos = nx.kamada_kawai_layout(self.currentGraph)
            mapped: dict[int, tuple[float, float]] = {}
            for v, (x, y) in pos.items():
                mapped[v] = (
                    (x + 1.0) * 0.5 * canvas_w,
                    (y + 1.0) * 0.5 * canvas_h,
                )
            self._set_positions_from_layout(mapped)
            return

        if layoutName == "Dust&Magnet":
            from nanohubcitmanager.visualization.layout.DustMagnetLayout import DustMagnetLayout

            layout = DustMagnetLayout(self.g)
            layout.setSize((self.canvasWidth, self.canvasHeight))
            layout.setNetwork(self)
            layout.filterEdgeByWeight(2.0)
            layout.run(iteration, use_kamada_kawai=use_kamada_kawai)
            for v in self.currentGraph.nodes():
                pt = layout.transform(v)
                if v in self.verticesInfo and pt is not None:
                    self.verticesInfo[v].coordinate = Point(int(pt.getX()), int(pt.getY()))
            self.components = None
            return

        if self.currentGraph.number_of_nodes() == 0:
            return

        if layoutName == "spring" or layoutName == "spring2":
            pos = nx.spring_layout(self.currentGraph, iterations=iteration)
        elif layoutName == "circle":
            pos = nx.circular_layout(self.currentGraph)
        elif layoutName == "Fruchterman-Reingold":
            pos = nx.fruchterman_reingold_layout(self.currentGraph, iterations=iteration)
        else:
            pos = nx.spring_layout(self.currentGraph, iterations=iteration)

        mapped: dict[int, tuple[float, float]] = {}
        for v, (x, y) in pos.items():
            mapped[v] = (
                (x + 1.0) * 0.5 * canvas_w,
                (y + 1.0) * 0.5 * canvas_h,
            )
        self._set_positions_from_layout(mapped)

    def colorNodeBy(self, attr: str, colorIndex: int):
        if attr == "same":
            for v in self.currentGraph.nodes():
                if v in self.verticesInfo:
                    self.verticesInfo[v].fillColor = self.colors[colorIndex]
        elif attr == "component size":
            for v in self.currentGraph.nodes():
                if v in self.verticesInfo:
                    size = int(self.verticesInfo[v].score.get("componentSize", 0))
                    colorCode = size % len(self.colors)
                    self.verticesInfo[v].fillColor = self.colors[colorCode]
        elif attr == "degree":
            pass
        elif attr == "default":
            pass
        else:
            maxColorCode = 0
            for v in self.currentGraph.nodes():
                if v not in self.verticesInfo:
                    continue
                vertex = self.verticesInfo[v]
                if attr not in vertex.extraProperties:
                    continue
                val = int(vertex.extraProperties[attr])
                if val < 0:
                    val = -val
                colorCode = (val + colorIndex) % len(self.colors)
                maxColorCode = max(maxColorCode, colorCode)
                vertex.fillColor = self.colors[colorCode]
            self.startColorCode = maxColorCode + 1

    def edgeOpacityBy(self, op: float):
        for _, _, data in self.currentGraph.edges(data=True):
            eID = data.get("id")
            if eID and eID in self.edgesInfo:
                self.edgesInfo[eID].opacity = op

    def opacityBy(self, *args):
        # opacityBy(double)
        if len(args) == 1:
            op = float(args[0])
            for v in self.currentGraph.nodes():
                if v in self.verticesInfo:
                    vi = self.verticesInfo[v]
                    vi.fillOpacity = op
                    vi.borderOpacity = op
            self.edgeOpacityBy(op)
            return

        # opacityBy(String attr, double startOp, double endOp, boolean reset)
        if len(args) != 4:
            raise TypeError("opacityBy expects 1 or 4 args")

        attr = str(args[0])
        startOp = float(args[1])
        endOp = float(args[2])
        reset = bool(args[3])

        total = 0.0
        max_val = -1000000.0
        min_val = 1000000.0
        i = 0
        for v in self.currentGraph.nodes():
            vi = self.verticesInfo[v]
            val = vi.extraProperties.get(attr)
            if isinstance(val, (float, int)):
                fv = float(val)
                total += fv
                max_val = max(max_val, fv)
                min_val = min(min_val, fv)
            i += 1

        if max_val == min_val:
            min_val = max_val - 1.0
        range_val = max_val - min_val
        if range_val <= 0:
            range_val = 1.0

        for v in self.currentGraph.nodes():
            vi = self.verticesInfo[v]
            val = vi.extraProperties.get(attr)
            initOpacity = 1.0 if reset else vi.fillOpacity
            if isinstance(val, (float, int)):
                fv = float(val)
                newOp = min(initOpacity, startOp + (fv - min_val) / range_val * (endOp - startOp))
                vi.fillOpacity = newOp
                vi.borderOpacity = newOp

            for _, _, data in self.currentGraph.edges(v, data=True):
                eID = data.get("id")
                if not eID or eID not in self.edgesInfo:
                    continue
                eData = self.edgesInfo[eID]
                from_v = eData.fromV
                to_v = eData.toV
                eData.opacity = min(from_v.fillOpacity, to_v.fillOpacity)

    def sizeByBinary(self, attr: str, threshold: float):
        numVs = self.currentGraph.number_of_nodes()
        vSpacing = 12.0
        self.basicNodeSize = math.sqrt(self.canvasWidth * self.canvasHeight / (numVs + vSpacing)) / 8
        bigNodeSize = 2.5 * self.basicNodeSize

        total = 0.0
        i = 0
        if attr == "componentSize" or attr == "betweenness":
            for v in self.currentGraph.nodes():
                total += float(self.verticesInfo[v].score.get(attr, 0.0))
                i += 1
            ave = 1.0 if i == 0 else total / i
            for v in self.currentGraph.nodes():
                size = float(self.verticesInfo[v].score.get(attr, 0.0))
                self.verticesInfo[v].radius = int(size / ave * self.basicNodeSize)
        elif attr == "default":
            for v in self.currentGraph.nodes():
                self.verticesInfo[v].radius = int(self.basicNodeSize)
        else:
            for v in self.currentGraph.nodes():
                size = self.verticesInfo[v].extraProperties.get(attr)
                if isinstance(size, (float, int)):
                    self.verticesInfo[v].radius = int(bigNodeSize if float(size) >= threshold else self.basicNodeSize)

    def sizeByRange(self, attr: str, divVals: list[int]):
        numVs = self.currentGraph.number_of_nodes()
        vSpacing = 12.0
        self.basicNodeSize = math.sqrt(self.canvasWidth * self.canvasHeight / (numVs + vSpacing)) / 8
        biggestNodeSize = 3.0 * self.basicNodeSize

        for v in self.currentGraph.nodes():
            size = self.verticesInfo[v].extraProperties.get(attr)
            if size is None:
                continue
            sz = int(size)
            level = 0
            for dVal in divVals:
                if sz <= dVal:
                    break
                level += 1
            self.verticesInfo[v].radius = int(self.basicNodeSize + (biggestNodeSize - self.basicNodeSize) / len(divVals) * level)

    def sizeBy(self, attr: str):
        numVs = self.currentGraph.number_of_nodes()
        vSpacing = 12.0
        basicNodeSize = math.sqrt(self.canvasWidth * self.canvasHeight / (numVs + vSpacing)) / 3.5
        maxNodeSize = basicNodeSize * 4

        total = 0.0
        ave = 0.0
        i = 0

        if attr == "componentSize" or attr == "betweenness":
            min_val = 1000000.0
            max_val = -1000000.0
            for v in self.currentGraph.nodes():
                size = float(self.verticesInfo[v].score.get(attr, 0.0)) if v in self.verticesInfo else 0.0
                total += size
                i += 1
                min_val = min(min_val, size)
                max_val = max(max_val, size)
            ave = 1.0 if i == 0 else total / i
            rng = max_val - min_val
            if rng == 0:
                rng = 1.0
            for v in self.currentGraph.nodes():
                if v in self.verticesInfo:
                    size = float(self.verticesInfo[v].score.get(attr, 0.0))
                    self.verticesInfo[v].radius = int(basicNodeSize + (size - min_val) / rng * (maxNodeSize - basicNodeSize))
        elif attr == "default":
            for v in self.currentGraph.nodes():
                self.verticesInfo[v].radius = int(basicNodeSize)
        else:
            max_val = -1000000.0
            min_val = 1000000.0
            for v in self.currentGraph.nodes():
                size = self.verticesInfo[v].extraProperties.get(attr)
                if isinstance(size, (float, int)):
                    fv = float(size)
                    total += fv
                    max_val = max(max_val, fv)
                    min_val = min(min_val, fv)
                i += 1
            ave = total / i if i else 0.0
            rng = max_val - min_val
            if rng <= 0:
                rng = 1.0
            for v in self.currentGraph.nodes():
                size = self.verticesInfo[v].extraProperties.get(attr)
                if isinstance(size, (float, int)):
                    fv = float(size)
                    self.verticesInfo[v].radius = int(basicNodeSize + (fv - ave) / rng * basicNodeSize * 1.5)

    def outputCurrentNetworkVertices(self):
        return

    def getVisualRadius(self) -> float:
        return 0.0

    def filterByDegree(self, thr: int):
        for v in list(self.currentGraph.nodes()):
            wlv = self.verticesInfo.get(v)
            if wlv and wlv.score.get("degree", 0) < thr:
                self.currentGraph.remove_node(v)
        self.components = None

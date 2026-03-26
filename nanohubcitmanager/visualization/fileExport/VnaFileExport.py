from __future__ import annotations

from .GraphFileExport import GraphFileExport


class VnaFileExport(GraphFileExport):
    def __init__(self, net):
        super().__init__(net)

    def export(self, fileName: str):
        nodeDataSection = '*Node data\n"ID"\n'
        self.writeToFile(fileName, nodeDataSection, False)

        nodeDataSection = ""
        vs = list(self.network.currentGraph.nodes())
        n = 0
        for v in vs:
            wlv = self.network.getVerticesInfo().get(v)
            if wlv is None:
                continue
            nodeDataSection += f"{wlv.id}\n"
            if n % 1000 == 1:
                self.writeToFile(fileName, nodeDataSection, True)
                nodeDataSection = ""
            n += 1
        self.writeToFile(fileName, nodeDataSection, True)

        n = 0
        nodePropertySection = (
            "*Node properties\n"
            "ID x y color shape size labeltext labelsize labelcolor gapx gapy active\n"
        )
        self.writeToFile(fileName, nodePropertySection, True)

        nodePropertySection = ""
        for v in vs:
            wlv = self.network.getVerticesInfo().get(v)
            if wlv is None:
                continue
            fillColor = "13828244"
            nodePropertySection += (
                f'{wlv.id} {wlv.coordinate.getX() / self.network.canvasWidth} '
                f'{wlv.coordinate.getY() / self.network.canvasHeight} {fillColor} 1 6 '
                f'"{wlv.label}" 7 8421504 3 5 TRUE\n'
            )
            if n % 1000 == 0:
                self.writeToFile(fileName, nodePropertySection, True)
                nodePropertySection = ""
            n += 1
        self.writeToFile(fileName, nodePropertySection, True)

        n = 0
        edgeDataSection = "*Tie data\nFrom To \"whatever\"\n"
        for u, v, _ in self.network.currentGraph.edges(data=True):
            edgeDataSection += f"{u}, {v}, 1\n"
            if n % 1000 == 0:
                self.writeToFile(fileName, edgeDataSection, True)
                edgeDataSection = ""
            n += 1
        self.writeToFile(fileName, edgeDataSection, True)

        edgePropertySection = "*Tie properties\nFROM TO color size headcolor headsize active\n"
        self.writeToFile(fileName, edgePropertySection, True)

from __future__ import annotations

from pathlib import Path

from nanohubcitmanager.visualization.network.Network import Network


class GraphFileExport:
    def __init__(self, net: Network):
        self.network = net
        self.options: dict[str, object] = {
            "canvasSize": 5,
            "nodeSize": 5,
            "nodeShape": "circle",
            "nodeLabelFontSize": 1,
            "fixedNodeSize": False,
        }

    def writeToFile(self, fileName: str, fileBody: str, append: bool):
        mode = "a" if append else "w"
        Path(fileName).open(mode, encoding="utf-8").write(fileBody)

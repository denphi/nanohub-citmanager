from __future__ import annotations

import json

from .GraphFileExport import GraphFileExport


class GenericFileExport(GraphFileExport):
    def __init__(self, niNet):
        super().__init__(niNet)

    def export(self, fileName: str):
        all_obj = {}
        nodeArray = []

        for v in list(self.network.currentGraph.nodes()):
            wlv = self.network.getVerticesInfo().get(v)
            if wlv is None:
                continue
            nodeArray.append(wlv.getJSONObject(self.network.canvasWidth, self.network.canvasHeight))

        all_obj["nodes"] = nodeArray

        edgeArray = []
        for _, _, data in self.network.currentGraph.edges(data=True):
            eID = data.get("id")
            if not eID:
                continue
            wle = self.network.getEdgesInfo().get(eID)
            if wle is None:
                continue
            edgeArray.append(wle.getJSONObject(self.network.canvasWidth, self.network.canvasHeight))

        all_obj["edges"] = edgeArray
        self.writeToFile(fileName, json.dumps(all_obj), False)

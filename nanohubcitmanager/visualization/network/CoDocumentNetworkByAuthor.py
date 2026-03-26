from __future__ import annotations

import networkx as nx

from .Network import Network
from .WeightedLabelledVertex import WeightedLabelledVertex
from .WeightedLabelledEdge import WeightedLabelledEdge


class CoDocumentNetworkByAuthor(Network):
    def __init__(self):
        super().__init__()
        self.docAliasResp: dict = {}
        self.authorAliasResp: dict = {}
        self.docAuthorResp: dict = {}
        self.docTitleResp: dict = {}
        self.docInfoResp: dict = {}
        self.extraInfoResp: dict = {}
        self.extraInfo: dict = {}

    def processResponse(self):
        self.extraInfo = self.extraInfoResp.get("result", {}).get("data", {})

        documentAliasMap: dict[int, int] = {}
        data = self.docAliasResp.get("result", {}).get("data", {})
        for key, val in data.items():
            documentAliasMap[int(key)] = int(val)

        authorAliasMap: dict[int, int] = {}
        data = self.authorAliasResp.get("result", {}).get("data", {})
        for key, val in data.items():
            authorAliasMap[int(key)] = int(val)

        docTitleMap: dict[int, str] = {}
        data = self.docTitleResp.get("result", {}).get("data", {})
        for key, val in data.items():
            id_ = int(key)
            title = "" if val is None else str(val)
            docTitleMap[id_] = title
            v = WeightedLabelledVertex()
            v.id = id_
            v.label = title
            self.verticesInfo[id_] = v

        docIDs: set[int] = set()
        authorDocMap: dict[int, list[int]] = {}
        data = self.docAuthorResp.get("result", {}).get("data", {})
        for key, val in data.items():
            documentID = int(key)
            for person in val:
                personID = int(person)
                if personID in authorAliasMap:
                    personID = authorAliasMap[personID]
                if personID not in authorDocMap:
                    authorDocMap[personID] = []
                authorDocMap[personID].append(documentID)
            docIDs.add(documentID)

        self.stats["numAuthors"] = float(len(authorDocMap))
        self.stats["numDocs"] = float(len(docIDs))

        self.g = nx.Graph()
        for dID in docIDs:
            self.g.add_node(dID)

        for docs in authorDocMap.values():
            if len(docs) <= 1:
                continue
            for i in range(len(docs)):
                for n in range(len(docs)):
                    d1ID = docs[i]
                    d2ID = docs[n]
                    if d1ID < d2ID:
                        eID = f"^{d1ID}-{d2ID}$"
                        if eID in self.edgesInfo:
                            self.edgesInfo[eID].increaseWeightBy(1)
                        else:
                            v1 = self.verticesInfo[d1ID]
                            v2 = self.verticesInfo[d2ID]
                            eInfo = WeightedLabelledEdge(v1, v2)
                            eInfo.id = eID
                            eInfo.setWeight(1)
                            self.edgesInfo[eID] = eInfo
                            self.g.add_edge(d1ID, d2ID, id=eID)

        self.currentGraph = self.g
        self.components = None
        self.processVertexInfo()

    def getExtraInfo(self) -> dict:
        return self.extraInfo

    def processVertexInfo(self):
        pass

    def config(self, params):
        pass

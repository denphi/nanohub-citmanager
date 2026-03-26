from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .CoDocumentNetworkByAuthor import CoDocumentNetworkByAuthor


class NanoHUBCoDocumentNetwork(CoDocumentNetworkByAuthor):
    def __init__(
        self,
        inFile: str | None = None,
        client: Any | None = None,
        begin_year: int | None = None,
        end_year: int | None = None,
        cache_dir: str = ".",
        force_refresh: bool = False,
    ):
        super().__init__()
        self.client = None
        self.begin_year = int(begin_year or 0)
        self.end_year = int(end_year or 0)
        self.cache_dir = cache_dir
        self.force_refresh = force_refresh
        self._raw_documents: list[dict[str, Any]] = []

        if inFile is not None:
            self._log(f"Loading documents from {inFile}")
            self._raw_documents = json.loads(Path(inFile).read_text(encoding="utf-8"))
            self._log(f"Loaded {len(self._raw_documents)} documents")
            self._build_responses_from_documents()
            self.processResponse()
            return

        if client is not None:
            self.client = client
            self.extraInfo = {"h-index": "0"}
            return

        raise ValueError("Either inFile or client must be provided")

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"documents_{self.begin_year}_{self.end_year}.json")

    def _log(self, message: str):
        print(f"[network] {message}", flush=True)

    def fetch(self):
        if self.client is None:
            return

        cache_path = self._cache_path()
        fetch_start = time.time()
        if (not self.force_refresh) and os.path.exists(cache_path):
            self._log(f"Loading cached documents from {cache_path}")
            with open(cache_path, encoding="utf-8") as f:
                self._raw_documents = json.load(f)
            self._log(
                f"Loaded {len(self._raw_documents)} documents from cache "
                f"in {time.time() - fetch_start:.1f}s"
            )
            return

        self._log(f"Requesting DocumentNetwork list for years {self.begin_year}-{self.end_year}")
        result = self.client._api_call(
            "DocumentNetwork",
            {
                "action": "list",
                "yearFrom": self.begin_year,
                "yearTo": self.end_year,
            },
        )
        self._raw_documents = result.get("documents", [])
        self._log(f"Received {len(self._raw_documents)} documents in {time.time() - fetch_start:.1f}s")

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self._raw_documents, f)
        self._log(f"Saved cache to {cache_path}")

    def process(self):
        process_start = time.time()
        self._log("Converting documents to graph")
        self._build_responses_from_documents()
        self.processResponse()
        self._log(
            f"Process complete: {len(self.verticesInfo)} vertices, "
            f"{len(self.edgesInfo)} edges in {time.time() - process_start:.1f}s"
        )

    def _compute_h_index(self, citations: list[int]) -> int:
        citations = sorted((int(c) for c in citations), reverse=True)
        h = 0
        for i, c in enumerate(citations, start=1):
            if c >= i:
                h = i
            else:
                break
        return h

    def _canonical_tool_name(self, doc: dict[str, Any]) -> str:
        tool_name = doc.get("toolName")
        if tool_name is not None and str(tool_name).strip() != "":
            return str(tool_name)

        alias = str(doc.get("toolAlias") or "").strip().lower()
        if alias == "workspace":
            return "Workspace"
        if alias == "schred":
            return "Schred"
        if alias == "nanomos":
            return "NanoMOS"
        if alias == "fettoy":
            return "FETToy"
        if alias == "":
            return "NULL"
        return alias

    def _build_responses_from_documents(self):
        doc_alias_data: dict[str, str] = {}
        author_alias_data: dict[str, str] = {}
        doc_title_data: dict[str, str] = {}
        doc_author_data: dict[str, list[str]] = {}
        doc_info_data: dict[str, dict[str, str]] = {}
        citation_values: list[int] = []

        for doc in self._raw_documents:
            raw_id = doc.get("id", 0)
            try:
                doc_id = int(raw_id)
            except Exception:
                continue
            if doc_id <= 0:
                continue

            doc_id_str = str(doc_id)
            doc_title_data[doc_id_str] = str(doc.get("title") or "")

            raw_authors = doc.get("authors") or []
            authors: list[str] = []
            for a in raw_authors:
                try:
                    authors.append(str(int(a)))
                except Exception:
                    continue
            doc_author_data[doc_id_str] = authors

            cit_val = int(doc.get("cnt_citations", doc.get("cit", 0)) or 0)
            citation_values.append(cit_val)

            doc_info_data[doc_id_str] = {
                "exp_list_exp_data": str(int(doc.get("exp_list_exp_data", 0) or 0)),
                "year": str(int(doc.get("year", 0) or 0)),
                "exp_data": str(int(doc.get("exp_data", 0) or 0)),
                "affiliated": str(int(doc.get("affiliated", 0) or 0)),
                "ID": doc_id_str,
                "ref_type": str(doc.get("ref_type") or ""),
                "cit": str(cit_val),
                "toolName": self._canonical_tool_name(doc),
            }

        h_index = self._compute_h_index(citation_values)

        self.docAliasResp = {"result": {"data": doc_alias_data, "status": "OK"}, "id": None, "exeTime": 0}
        self.authorAliasResp = {"result": {"data": author_alias_data, "status": "OK"}, "id": None, "exeTime": 0}
        self.docTitleResp = {"result": {"data": doc_title_data, "status": "OK"}, "id": None, "exeTime": 0}
        self.docAuthorResp = {"result": {"data": doc_author_data, "status": "OK"}, "id": None, "exeTime": 0}
        self.docInfoResp = {"result": {"data": doc_info_data, "status": "OK"}, "id": None, "exeTime": 0}
        self.extraInfoResp = {
            "result": {"data": {"h-index": str(h_index)}, "status": "OK"},
            "id": None,
            "exeTime": 0,
        }

    def processVertexInfo(self):
        data = self.docInfoResp.get("result", {}).get("data", {})
        for key, attrVal in data.items():
            id_ = int(key)
            v = self.verticesInfo.get(id_)
            if v is None:
                continue

            cit = int(attrVal.get("cit", "0"))
            v.extraProperties["citations"] = cit

            aff = int(attrVal.get("affiliated", "0"))
            v.extraProperties["NCN-affiliated"] = aff
            v.extraProperties["NCN"] = 1 if aff == 1 else 0
            v.propertyStatus["NCN"] = "layout"

            refType = "[" + str(attrVal.get("ref_type", "")) + "]"
            v.extraProperties["refType"] = refType
            v.extraProperties["refTypeResEdu"] = 0
            v.extraProperties["refTypeEdu"] = 0
            v.extraProperties["refTypeRes"] = 0
            v.extraProperties["refTypeAllRes"] = 0
            v.extraProperties["refTypeCyber"] = 0
            if "N" in refType:
                v.extraProperties["refTypeCode"] = 0
                v.extraProperties["refTypeResEdu"] = 1
                v.extraProperties["refTypeAllRes"] = 1
            elif "C" in refType:
                v.extraProperties["refTypeCode"] = 1
                v.extraProperties["refTypeCyber"] = 1
            elif "E" in refType:
                v.extraProperties["refTypeCode"] = 2
                v.extraProperties["refTypeEdu"] = 1
            elif "R" in refType:
                v.extraProperties["refTypeCode"] = 3
                v.extraProperties["refTypeRes"] = 1
                v.extraProperties["refTypeAllRes"] = 1
            else:
                v.extraProperties["refTypeCode"] = 4

            v.propertyStatus["refTypeResEdu"] = "layout"
            v.propertyStatus["refTypeEdu"] = "layout"
            v.propertyStatus["refTypeRes"] = "layout"
            v.propertyStatus["refTypeCyber"] = "layout"

            expList = int(attrVal.get("exp_list_exp_data", "0"))
            expData = int(attrVal.get("exp_data", "0"))
            v.extraProperties["expListData"] = expList
            v.extraProperties["expData"] = expData
            expListCode = 0
            if expList == 1:
                expListCode = 2
                v.extraProperties["expListDataBool"] = 1
            else:
                v.extraProperties["expListDataBool"] = 0
            v.propertyStatus["expListDataBool"] = "layout"

            if expData == 1:
                if expListCode == 0:
                    expListCode = 1
                v.extraProperties["expListDataCode"] = 1
            else:
                v.extraProperties["expDataBool"] = 0
            v.propertyStatus["expDataBool"] = "layout"
            v.extraProperties["expListDataCode"] = expListCode

            toolNameVal = attrVal.get("toolName")
            toolName = "NULL" if toolNameVal is None else str(toolNameVal)
            v.extraProperties["toolCited"] = toolName
            v.extraProperties["tool_workspace"] = 0
            v.extraProperties["tool_schred"] = 0
            v.extraProperties["tool_nanomos"] = 0
            v.extraProperties["tool_fettoy"] = 0
            if toolName.lower() == "workspace":
                v.extraProperties["tool_workspace"] = 1
                v.extraProperties["toolCode"] = 0
            elif toolName.lower() == "schred":
                v.extraProperties["tool_schred"] = 1
                v.extraProperties["toolCode"] = 1
            elif toolName.lower() == "nanomos":
                v.extraProperties["tool_nanomos"] = 1
                v.extraProperties["toolCode"] = 2
            elif toolName.lower() == "fettoy":
                v.extraProperties["tool_fettoy"] = 1
                v.extraProperties["toolCode"] = 3
            else:
                v.extraProperties["toolCode"] = 5

            v.propertyStatus["tool_workspace"] = "layout"
            v.propertyStatus["tool_schred"] = "layout"
            v.propertyStatus["tool_nanomos"] = "layout"
            v.propertyStatus["tool_fettoy"] = "layout"

            year = int(attrVal.get("year", "0"))
            v.extraProperties["year"] = year

    def setInYearRange(self, by: int, ey: int):
        for v in self.currentGraph.nodes():
            vi = self.verticesInfo[v]
            y = int(vi.extraProperties.get("year", 0))
            vi.extraProperties["inYearRange"] = 1 if by <= y <= ey else 0

    def config(self, params):
        pass

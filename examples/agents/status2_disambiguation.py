"""
Status 2 — Disambiguation Agent

Checks for potential duplicate citations (same title or same authors)
already present in the system.

Decision logic:
  • No strong duplicates found → advance to status 3
  • Likely duplicates detected → flag in notes, keep at status 2 for human review
"""

import json
from typing import Any, Dict, List

from .base_agent import BaseCitationAgent


class DisambiguationAgent(BaseCitationAgent):
    """Detect potential duplicate citations at status 2."""

    STAGE_NAME = "disambiguation"
    TARGET_STATUS = 2
    NEXT_STATUS = 3

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @property
    def _system_prompt(self) -> str:
        return (
            "You are a citation pipeline agent responsible for Step 2: Disambiguation.\n\n"
            "Your job is to detect whether a citation already exists in the system under a "
            "different ID (duplicate title, same authors, etc.).\n\n"
            "Workflow:\n"
            "1. Call `search_by_title` with a significant portion of the citation title.\n"
            "2. Call `search_by_author` for each of the first two authors (if available).\n"
            "3. Analyse the results. A match is suspicious if it shares the same title AND "
            "at least one overlapping author, OR if the DOI is identical.\n"
            "4. If NO suspicious duplicates are found, call `advance_to_status_3`.\n"
            "5. If suspicious duplicates ARE found, call `flag_potential_duplicates` listing "
            "the matching IDs and your reasoning. Do NOT advance; leave it for human review.\n"
            "6. Finish with a plain-text summary.\n\n"
            "Important: exclude the citation itself (same ID) from duplicate analysis. "
            "Do not over-flag — minor title differences (prepositions, punctuation) are not "
            "enough on their own."
        )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _tool_definitions(self) -> List[Dict]:
        return [
            {
                "name": "search_by_title",
                "description": (
                    "Search the citation database for citations whose title matches the query. "
                    "Returns up to 10 results with id, title, year, doi, and author names."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Title keywords to search for.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 10).",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_by_author",
                "description": (
                    "Search the citation database for citations authored by a given person. "
                    "Returns up to 10 results with id, title, year, and author list."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "author_name": {
                            "type": "string",
                            "description": "Author's full name or last name.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 10).",
                        },
                    },
                    "required": ["author_name"],
                },
            },
            {
                "name": "advance_to_status_3",
                "description": (
                    "Advance the citation to status 3 (classification). "
                    "Call only when no suspicious duplicates were found."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "note": {
                            "type": "string",
                            "description": "Brief note confirming no duplicates found.",
                        },
                    },
                    "required": ["citation_id"],
                },
            },
            {
                "name": "flag_potential_duplicates",
                "description": (
                    "Record potential duplicate IDs in the citation notes and keep it at "
                    "status 2 so a human can verify before proceeding."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "duplicate_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of citation IDs that may be duplicates.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why these are suspected duplicates.",
                        },
                    },
                    "required": ["citation_id", "duplicate_ids", "reasoning"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, input_data: Dict) -> Dict:

        if name == "search_by_title":
            query = input_data["query"]
            limit = input_data.get("limit", 10)
            try:
                docs = self.cit_client._api_call(
                    "CitationCRUD",
                    {"action": "list", "search": query, "limit": limit},
                ).get("documents", [])
                return {
                    "count": len(docs),
                    "results": [
                        {
                            "id": d.get("id"),
                            "title": d.get("title", ""),
                            "year": d.get("year"),
                            "doi": d.get("doi", ""),
                            "authors": [
                                f"{a.get('firstName','')} {a.get('lastName','')}".strip()
                                for a in (d.get("authors") or [])
                            ],
                        }
                        for d in docs
                    ],
                }
            except Exception as exc:
                return {"count": 0, "error": str(exc)}

        if name == "search_by_author":
            author_name = input_data["author_name"]
            limit = input_data.get("limit", 10)
            try:
                docs = self.cit_client._api_call(
                    "CitationCRUD",
                    {"action": "list", "search": author_name, "limit": limit},
                ).get("documents", [])
                return {
                    "count": len(docs),
                    "results": [
                        {
                            "id": d.get("id"),
                            "title": d.get("title", ""),
                            "year": d.get("year"),
                            "authors": [
                                f"{a.get('firstName','')} {a.get('lastName','')}".strip()
                                for a in (d.get("authors") or [])
                            ],
                        }
                        for d in docs
                    ],
                }
            except Exception as exc:
                return {"count": 0, "error": str(exc)}

        if name == "advance_to_status_3":
            cid = input_data["citation_id"]
            note = input_data.get("note", "No duplicates detected — advancing to classification.")
            return self._advance_status(cid, self.NEXT_STATUS, note)

        if name == "flag_potential_duplicates":
            cid = input_data["citation_id"]
            ids = input_data.get("duplicate_ids", [])
            reasoning = input_data.get("reasoning", "")
            note = f"[POSSIBLE DUPLICATES: {ids}] {reasoning}"
            return self._append_note(cid, note)

        return {"error": f"Unknown tool: {name}"}

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, citation) -> str:
        author_list = ", ".join(
            f"{a.get('firstName','')} {a.get('lastName','')}".strip()
            for a in (citation.authors or [])[:3]
        )
        return (
            f"Please check citation ID {citation.id} for potential duplicates in the system.\n\n"
            f"Citation details:\n"
            f"  Title   : {citation.title or '(none)'}\n"
            f"  Authors : {author_list or '(none)'}\n"
            f"  Year    : {citation.year or '(none)'}\n"
            f"  DOI     : {citation.doi or '(none)'}\n\n"
            "Search for similar titles and author names, then decide whether to advance "
            "or flag for human review."
        )

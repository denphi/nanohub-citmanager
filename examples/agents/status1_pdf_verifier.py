"""
Status 1 — PDF Verifier Agent

Checks whether a PDF file is associated with the citation.

Decision logic:
  • PDF found & accessible  → advance to status 2
  • PDF missing             → keep at status 1, add a flag note for manual upload
"""

import json
from typing import Any, Dict, List

from .base_agent import BaseCitationAgent


class PDFVerifierAgent(BaseCitationAgent):
    """Verify PDF existence for citations at status 1."""

    STAGE_NAME = "pdf_verifier"
    TARGET_STATUS = 1
    NEXT_STATUS = 2

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @property
    def _system_prompt(self) -> str:
        return (
            "You are a citation pipeline agent responsible for Step 1: PDF Verification.\n\n"
            "Your job is to determine whether a PDF file is attached to the citation and "
            "take the appropriate action:\n"
            "1. Call `get_pdf_info` to check if a PDF exists.\n"
            "2. If a PDF is found (filename is non-empty and size > 0), call `advance_to_status_2` "
            "to progress the citation. Include a brief note about what was found.\n"
            "3. If no PDF is found or the file is empty, call `flag_missing_pdf` with a helpful "
            "note explaining that a PDF needs to be uploaded before processing can continue.\n"
            "4. Always finish with a short plain-text summary of what you did and why.\n\n"
            "Be concise. Do not make more tool calls than necessary."
        )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _tool_definitions(self) -> List[Dict]:
        return [
            {
                "name": "get_pdf_info",
                "description": (
                    "Check whether a PDF file is attached to a citation. "
                    "Returns found=true and the filename if a PDF is attached, "
                    "or found=false if no PDF exists."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {
                            "type": "integer",
                            "description": "The citation ID to check.",
                        }
                    },
                    "required": ["citation_id"],
                },
            },
            {
                "name": "advance_to_status_2",
                "description": (
                    "Mark the citation as having a verified PDF and advance it to status 2 "
                    "(disambiguation). Call this when a valid PDF has been confirmed."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "note": {
                            "type": "string",
                            "description": "Optional note describing the PDF that was found.",
                        },
                    },
                    "required": ["citation_id"],
                },
            },
            {
                "name": "flag_missing_pdf",
                "description": (
                    "Record that no PDF was found and keep the citation at status 1 so a "
                    "human can upload the file before the pipeline continues."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "note": {
                            "type": "string",
                            "description": "Explanation of why the citation cannot proceed.",
                        },
                    },
                    "required": ["citation_id", "note"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, input_data: Dict) -> Dict:
        cid = input_data["citation_id"]

        if name == "get_pdf_info":
            try:
                # The PDFManager get action returns raw binary when a PDF exists,
                # so we check the full_text_path field on the citation itself —
                # it is populated when a PDF is attached.
                citation = self.cit_client.get(cid)
                filename = citation.full_text_path or ""
                found = bool(filename)
                return {
                    "found": found,
                    "filename": filename,
                    "source_url": citation.url or "",
                }
            except Exception as exc:
                return {"found": False, "error": str(exc)}

        if name == "advance_to_status_2":
            note = input_data.get("note", "PDF verified — advancing to disambiguation.")
            return self._advance_status(cid, self.NEXT_STATUS, note)

        if name == "flag_missing_pdf":
            note = input_data.get("note", "PDF missing — cannot proceed.")
            return self._append_note(cid, f"[MISSING PDF] {note}")

        return {"error": f"Unknown tool: {name}"}

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, citation) -> str:
        return (
            f"Please verify whether citation ID {citation.id} has an associated PDF.\n\n"
            f"Citation details:\n"
            f"  Title : {citation.title or '(none)'}\n"
            f"  Year  : {citation.year or '(none)'}\n"
            f"  DOI   : {citation.doi or '(none)'}\n"
            f"  full_text_path: {citation.full_text_path or '(none)'}\n\n"
            "Check for the PDF, then take the appropriate action."
        )

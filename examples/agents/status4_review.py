"""
Status 4 — Review Agent (Experimentalist & Experimental Data)

Runs two independent reviews on the citation and records the results:

  1. Experimentalist review   → is at least one author an experimentalist?
     Field: exp_list_exp_data  (1 = yes, 0 = no)

  2. Experimental Data review → does the paper contain experimental data?
     Field: exp_data           (1 = yes, 0 = no)

Both reviews are performed in the same agent pass.  After both flags are set
the citation advances to status 5 for final human review.
"""

import json
from typing import Any, Dict, List

from .base_agent import BaseCitationAgent


class ReviewAgent(BaseCitationAgent):
    """
    Dual-review agent for citations at status 4.

    Reviews performed:
      A) Experimentalist?    — author background analysis
      B) Experimental Data?  — content/abstract analysis
    """

    STAGE_NAME = "review"
    TARGET_STATUS = 4
    NEXT_STATUS = 5

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @property
    def _system_prompt(self) -> str:
        return (
            "You are a citation pipeline agent responsible for Step 4: Dual Review.\n\n"
            "You must perform two independent assessments and record the results:\n\n"
            "--- Review A: Experimentalist? ---\n"
            "Determine whether at least one of the authors is an experimentalist "
            "(i.e. someone who conducts laboratory or field experiments, as opposed to purely "
            "theoretical or computational researchers).\n"
            "Signals to look for: affiliation with experimental labs, paper keywords such as "
            "'experiment', 'measurement', 'fabrication', 'characterisation', 'synthesis', "
            "study design involving physical samples or devices, etc.\n"
            "Call `set_experimentalist_flag` with value 1 (yes) or 0 (no) and your reasoning.\n\n"
            "--- Review B: Experimental Data? ---\n"
            "Determine whether the paper reports original experimental data (measurements, "
            "observations, test results) as opposed to purely theoretical derivations, "
            "simulations, or review/survey papers.\n"
            "Call `set_experimental_data_flag` with value 1 (yes) or 0 (no) and your reasoning.\n\n"
            "Workflow:\n"
            "1. Call `get_citation_details` to read title, abstract, authors, and keywords.\n"
            "2. Perform Review A → call `set_experimentalist_flag`.\n"
            "3. Perform Review B → call `set_experimental_data_flag`.\n"
            "4. Call `advance_to_status_5` to hand off to the human reviewer.\n"
            "5. Finish with a plain-text summary of both decisions.\n\n"
            "Both flags MUST be set before advancing. Be explicit in your reasoning."
        )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _tool_definitions(self) -> List[Dict]:
        return [
            {
                "name": "get_citation_details",
                "description": "Fetch the full citation record (title, abstract, authors, keywords).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"}
                    },
                    "required": ["citation_id"],
                },
            },
            {
                "name": "set_experimentalist_flag",
                "description": (
                    "Record whether at least one author is an experimentalist. "
                    "Set value=1 for yes, value=0 for no. "
                    "This sets the exp_list_exp_data field."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "value": {
                            "type": "integer",
                            "enum": [0, 1],
                            "description": "1 = at least one experimentalist author, 0 = no.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of the decision.",
                        },
                    },
                    "required": ["citation_id", "value", "reasoning"],
                },
            },
            {
                "name": "set_experimental_data_flag",
                "description": (
                    "Record whether the paper contains original experimental data. "
                    "Set value=1 for yes, value=0 for no. "
                    "This sets the exp_data field."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "value": {
                            "type": "integer",
                            "enum": [0, 1],
                            "description": "1 = contains experimental data, 0 = does not.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of the decision.",
                        },
                    },
                    "required": ["citation_id", "value", "reasoning"],
                },
            },
            {
                "name": "advance_to_status_5",
                "description": (
                    "Advance the citation to status 5 (human review). "
                    "Call only after both review flags have been set."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "note": {
                            "type": "string",
                            "description": "Brief summary of both review decisions.",
                        },
                    },
                    "required": ["citation_id"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, input_data: Dict) -> Dict:

        if name == "get_citation_details":
            return self._get_citation_dict(input_data["citation_id"])

        if name == "set_experimentalist_flag":
            cid = input_data["citation_id"]
            value = int(input_data["value"])
            reasoning = input_data.get("reasoning", "")
            try:
                citation = self.cit_client.get(cid)
                citation.exp_list_exp_data = value
                note = f"[Review A — Experimentalist: {'YES' if value else 'NO'}] {reasoning}"
                existing = citation.notes or ""
                sep = "\n" if existing else ""
                citation.notes = f"{existing}{sep}[Agent/{self.STAGE_NAME}] {note}"
                self.cit_client.update(citation)
                return {"ok": True, "exp_list_exp_data": value}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        if name == "set_experimental_data_flag":
            cid = input_data["citation_id"]
            value = int(input_data["value"])
            reasoning = input_data.get("reasoning", "")
            try:
                citation = self.cit_client.get(cid)
                citation.exp_data = value
                note = f"[Review B — Experimental Data: {'YES' if value else 'NO'}] {reasoning}"
                existing = citation.notes or ""
                sep = "\n" if existing else ""
                citation.notes = f"{existing}{sep}[Agent/{self.STAGE_NAME}] {note}"
                self.cit_client.update(citation)
                return {"ok": True, "exp_data": value}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        if name == "advance_to_status_5":
            cid = input_data["citation_id"]
            note = input_data.get("note", "Both reviews completed — advancing to human review.")
            return self._advance_status(cid, self.NEXT_STATUS, note)

        return {"error": f"Unknown tool: {name}"}

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, citation) -> str:
        author_list = "\n".join(
            f"    • {a.get('firstName','')} {a.get('lastName','')}".strip()
            + (f" ({a.get('organization','')})" if a.get("organization") else "")
            for a in (citation.authors or [])
        ) or "    (no authors listed)"

        keywords = ", ".join(citation.keywords or []) or "(none)"

        return (
            f"Please perform the dual review for citation ID {citation.id}.\n\n"
            f"Title    : {citation.title or '(none)'}\n"
            f"Year     : {citation.year or '(none)'}\n"
            f"Keywords : {keywords}\n\n"
            f"Abstract :\n{citation.abstract or '(no abstract)'}\n\n"
            f"Authors  :\n{author_list}\n\n"
            "Assess whether any author is an experimentalist AND whether the paper reports "
            "experimental data, set both flags, then advance to status 5."
        )

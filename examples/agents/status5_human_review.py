"""
Status 5 — Human Review Agent

This is the final automated stage.  The agent compiles a structured review
package — collecting metadata, agent notes, and review flags from the previous
stages — and writes a concise summary into the citation notes so the human
reviewer has everything in one place.

The citation stays at status 5; the human is expected to make the final
decision (approve → PUBLISHED=100, or reject → negative status).
"""

import json
import re
from typing import Any, Dict, List

try:
    import requests as _requests
except ImportError:
    _requests = None

from .base_agent import BaseCitationAgent


def _crossref_date_parts_to_ymd(parts: List[int]) -> str:
    y = int(parts[0]) if len(parts) >= 1 else 1900
    m = int(parts[1]) if len(parts) >= 2 else 1
    d = int(parts[2]) if len(parts) >= 3 else 1
    m = 1 if m < 1 or m > 12 else m
    d = 1 if d < 1 or d > 31 else d
    return f"{y:04d}-{m:02d}-{d:02d}"


def _date_from_crossref_payload(payload: Dict[str, Any]) -> str:
    msg = payload.get("message", {}) if isinstance(payload, dict) else {}
    candidates = [
        msg.get("published-print", {}).get("date-parts", []),
        msg.get("published-online", {}).get("date-parts", []),
        msg.get("issued", {}).get("date-parts", []),
        msg.get("created", {}).get("date-parts", []),
    ]
    for c in candidates:
        if c and isinstance(c, list) and c[0]:
            return _crossref_date_parts_to_ymd(c[0])
    return ""


def _extract_date_from_html(html: str) -> str:
    patterns = [
        r'(?is)<meta[^>]+(?:name|property)=["\']citation_publication_date["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+(?:name|property)=["\']dc\.date["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+(?:name|property)=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for p in patterns:
        m = re.search(p, html)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        m2 = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", raw)
        if m2:
            return f"{int(m2.group(1)):04d}-{int(m2.group(2)):02d}-{int(m2.group(3)):02d}"
        m3 = re.search(r"\b(19|20)\d{2}\b", raw)
        if m3:
            return f"{int(m3.group(0)):04d}-01-01"
    return ""


class HumanReviewAgent(BaseCitationAgent):
    """
    Prepare a human-readable review package for citations at status 5.

    Unlike earlier agents, this one does NOT advance the citation automatically.
    It leaves the citation at status 5 with an enriched notes field.
    """

    STAGE_NAME = "human_review"
    TARGET_STATUS = 5
    # No automatic advancement — human decides
    NEXT_STATUS = 5

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @property
    def _system_prompt(self) -> str:
        return (
            "You are a citation pipeline agent responsible for Step 5: Human Review Preparation.\n\n"
            "Your job is to compile a clear, structured summary that helps a human reviewer "
            "make the final decision about whether to publish or reject this citation.\n\n"
            "Workflow:\n"
            "1. Call `get_citation_details` to retrieve all metadata.\n"
            "2. Call `get_pipeline_history` to retrieve the agent notes recorded in previous stages.\n"
            "3. Synthesize the information and call `write_review_summary` with a structured "
            "summary that includes ALL of the following sections:\n\n"
            "   [1] BIBLIOGRAPHIC OVERVIEW\n"
            "       Title, authors, year, publication name, DOI, ref_type, dates (received/accepted)\n\n"
            "   [2] PDF STATUS\n"
            "       Whether a PDF was found and verified (step 1)\n\n"
            "   [3] DISAMBIGUATION\n"
            "       Whether any potential duplicates were found (step 2)\n\n"
            "   [4] COMPLETENESS\n"
            "       Fields that were missing and how they were filled (step 3).\n"
            "       Note any fields that could not be filled.\n\n"
            "   [5] REVIEW FLAGS\n"
            "       Experimentalist (exp_list_exp_data): YES / NO\n"
            "       Experimental Data (exp_data)       : YES / NO\n"
            "       NCN Affiliated (affiliated)        : YES / NO\n\n"
            "   [6] NANOHUB CONNECTION ASSESSMENT\n"
            "       Based on ALL available evidence — affiliated flag, ref_type, keywords, abstract, "
            "publication name, author organizations — assess how likely it is that this citation is "
            "connected to nanoHUB or NCN.\n"
            "       Write a one-paragraph justification, then give a confidence percentage (0-100%).\n"
            "       Example: 'nanoHUB connection confidence: 85%'\n\n"
            "   [7] RECOMMENDATION\n"
            "       One of: APPROVE / REJECT / NEEDS ATTENTION\n"
            "       - APPROVE       : citation is clearly relevant to nanoHUB/NCN and all fields complete\n"
            "       - REJECT        : citation has no apparent nanoHUB/NCN connection\n"
            "       - NEEDS ATTENTION: incomplete fields, duplicate concerns, or ambiguous connection\n"
            "       Follow the recommendation with a single sentence explaining the reason.\n\n"
            "4. End with a plain-text confirmation that the summary has been written.\n\n"
            "The human reviewer will read only the notes field — make the summary self-contained "
            "and easy to scan. Use clear section headers and bullet points."
        )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _tool_definitions(self) -> List[Dict]:
        return [
            {
                "name": "get_citation_details",
                "description": "Fetch the full citation record including all metadata and flags.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"}
                    },
                    "required": ["citation_id"],
                },
            },
            {
                "name": "get_pipeline_history",
                "description": (
                    "Retrieve the notes field of the citation, which contains the log of "
                    "actions taken by each preceding pipeline agent."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"}
                    },
                    "required": ["citation_id"],
                },
            },
            {
                "name": "write_review_summary",
                "description": (
                    "Write a structured human-readable review summary into the citation notes. "
                    "This replaces only the summary section — previous agent logs are preserved."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "summary": {
                            "type": "string",
                            "description": (
                                "Full structured review summary for the human reviewer. "
                                "Must include: bibliographic overview, PDF status, disambiguation result, "
                                "completeness assessment, review flags (experimentalist/experimental data/affiliated), "
                                "nanoHUB connection assessment with confidence percentage (0-100%), "
                                "and a final recommendation (APPROVE / REJECT / NEEDS ATTENTION)."
                            ),
                        },
                    },
                    "required": ["citation_id", "summary"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, input_data: Dict) -> Dict:
        cid = input_data["citation_id"]

        if name == "get_citation_details":
            return self._get_citation_dict(cid)

        if name == "get_pipeline_history":
            try:
                citation = self.cit_client.get(cid)
                return {
                    "notes": citation.notes or "(no notes recorded)",
                    "exp_list_exp_data": citation.exp_list_exp_data,
                    "exp_data": citation.exp_data,
                    "affiliated": citation.affiliated,
                    "ref_type": citation.ref_type,
                    "status": citation.status,
                }
            except Exception as exc:
                return {"error": str(exc)}

        if name == "write_review_summary":
            summary = input_data.get("summary", "")
            try:
                citation = self.cit_client.get(cid)
                # Prepend the human review summary at the top of notes,
                # keeping previous agent logs below a separator.
                existing = citation.notes or ""
                separator = "\n" + "─" * 50 + "\n[Previous pipeline logs]\n"
                if "[Agent/" in existing:
                    citation.notes = (
                        f"[HUMAN REVIEW SUMMARY — Agent/{self.STAGE_NAME}]\n"
                        f"{summary}"
                        f"{separator}{existing}"
                    )
                else:
                    citation.notes = (
                        f"[HUMAN REVIEW SUMMARY — Agent/{self.STAGE_NAME}]\n"
                        f"{summary}"
                    )
                self.cit_client.update(citation)
                return {"ok": True, "summary_length": len(summary)}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        return {"error": f"Unknown tool: {name}"}

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, citation) -> str:
        exp_flag = {1: "YES", 0: "NO"}.get(citation.exp_list_exp_data, "not set")
        data_flag = {1: "YES", 0: "NO"}.get(citation.exp_data, "not set")
        return (
            f"Please prepare the human review package for citation ID {citation.id}.\n\n"
            f"Current state:\n"
            f"  Title              : {citation.title or '(none)'}\n"
            f"  Year               : {citation.year or '(none)'}\n"
            f"  Authors            : {len(citation.authors or [])} listed\n"
            f"  Experimentalist    : {exp_flag}\n"
            f"  Experimental Data  : {data_flag}\n"
            f"  Existing notes     : {(citation.notes or '')[:200]}{'...' if len(citation.notes or '') > 200 else ''}\n\n"
            "Gather all details, synthesise the pipeline results, and write a structured "
            "review summary so the human reviewer can make a final publish/reject decision."
        )

    # ------------------------------------------------------------------
    # Override run() — success = summary written (not status advance)
    # ------------------------------------------------------------------

    def _ensure_publish_date(self, citation_id: int) -> Dict[str, Any]:
        """
        Ensure date_publish is present for status 5 records.
        Strategy:
          1) keep existing date_publish
          2) DOI -> Crossref
          3) citation URL meta tags
          4) fallback date_accept/date_submit/year
        """
        citation = self.cit_client.get(citation_id)
        if citation.date_publish:
            return {"ok": True, "source": "existing", "date_publish": citation.date_publish}

        selected = ""
        source = ""

        if _requests is not None and citation.doi:
            try:
                doi = citation.doi.strip()
                crossref_url = f"https://api.crossref.org/works/{doi}"
                resp = _requests.get(crossref_url, timeout=12)
                if resp.status_code == 200:
                    selected = _date_from_crossref_payload(resp.json())
                    if selected:
                        source = "crossref"
            except Exception:
                pass

        if not selected and _requests is not None and citation.url:
            try:
                resp = _requests.get(
                    citation.url.strip(),
                    timeout=12,
                    allow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0 (citation-pipeline/1.0)"},
                )
                if resp.status_code < 400:
                    selected = _extract_date_from_html(resp.text[:400000])
                    if selected:
                        source = "citation_url"
            except Exception:
                pass

        if not selected and citation.date_accept:
            selected = citation.date_accept
            source = "date_accept"
        if not selected and citation.date_submit:
            selected = citation.date_submit
            source = "date_submit"
        if not selected and citation.year:
            selected = f"{int(citation.year):04d}-01-01"
            source = "year_fallback"

        if not selected:
            return {"ok": False, "error": "Unable to derive date_publish"}

        citation.date_publish = selected
        existing = citation.notes or ""
        sep = "\n" if existing else ""
        citation.notes = (
            f"{existing}{sep}[Agent/{self.STAGE_NAME}] "
            f"Auto-set date_publish={selected} (source={source})."
        )
        self.cit_client.update(citation)
        return {"ok": True, "source": source, "date_publish": selected}

    def run(self, citation_id: int):
        from .base_agent import AgentResult
        citation = self.cit_client.get(citation_id)
        status_before = citation.status

        print(f"\n{'─'*60}")
        print(f"  [{self.STAGE_NAME.upper()}] citation {citation_id}  (current status={status_before})")
        print(f"{'─'*60}")

        publish_date_result = self._ensure_publish_date(citation_id)
        if publish_date_result.get("ok"):
            print(
                "  [human_review] date_publish ensured: "
                f"{publish_date_result.get('date_publish')} "
                f"(source={publish_date_result.get('source')})"
            )
        else:
            print(f"  [human_review] warning: {publish_date_result.get('error')}")

        # refresh with any date_publish updates before prompt rendering
        citation = self.cit_client.get(citation_id)
        user_message = self._build_prompt(citation)
        result_text = self._run_agentic_loop(user_message)

        updated = self.cit_client.get(citation_id)
        # Success = notes now contain the review summary header
        success = "[HUMAN REVIEW SUMMARY" in (updated.notes or "")
        return AgentResult(
            success=success,
            citation_id=citation_id,
            status_before=status_before,
            status_after=updated.status,
            message=result_text[:300] if result_text else "(no output)",
            details={"ready_for_human_review": True},
        )

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
import os
import tempfile
import unicodedata
from urllib.parse import unquote_plus
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
        raw = unquote_plus((m.group(1) or "").strip()).replace("+", " ")
        m2 = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", raw)
        if m2:
            return f"{int(m2.group(1)):04d}-{int(m2.group(2)):02d}-{int(m2.group(3)):02d}"
        m_day_month = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", raw)
        if m_day_month:
            month_map = {
                "january": "01", "february": "02", "march": "03", "april": "04",
                "may": "05", "june": "06", "july": "07", "august": "08",
                "september": "09", "october": "10", "november": "11", "december": "12",
            }
            mm = month_map.get((m_day_month.group(2) or "").lower())
            if mm:
                return f"{int(m_day_month.group(3)):04d}-{mm}-{int(m_day_month.group(1)):02d}"
        m3 = re.search(r"\b(19|20)\d{2}\b", raw)
        if m3:
            return f"{int(m3.group(0)):04d}-01-01"

    raw_patterns = [
        r'(?is)\bpublicationDate=([^&"\']+)',
        r'(?is)"displayPublicationDate"\s*:\s*"([^"]+)"',
        r'(?is)"dateOfInsertion"\s*:\s*"([^"]+)"',
    ]
    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    for p in raw_patterns:
        m = re.search(p, html)
        if not m:
            continue
        raw = unquote_plus((m.group(1) or "").strip()).replace("+", " ")
        m2 = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", raw)
        if m2:
            return f"{int(m2.group(1)):04d}-{int(m2.group(2)):02d}-{int(m2.group(3)):02d}"
        m3 = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", raw)
        if m3:
            mm = month_map.get((m3.group(2) or "").lower())
            if mm:
                return f"{int(m3.group(3)):04d}-{mm}-{int(m3.group(1)):02d}"

    # Visible-text fallback for pages where date is rendered, not in meta tags
    text = re.sub(r"(?is)<script\b.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style\b.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    visible_patterns = [
        r"(?:Date\s+of\s+Publication|Publication\s+Date)\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"(?:Date\s+of\s+Publication|Publication\s+Date)\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"(?:Date\s+of\s+Publication|Publication\s+Date)\s*[:\-]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        r"First\s+Online\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"First\s+Online\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"First\s+Online\s*[:\-]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        r"Published(?:\s+online)?\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"Published(?:\s+online)?\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"Published(?:\s+online)?\s*[:\-]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    ]
    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    for p in visible_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        m2 = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", raw)
        if m2:
            return f"{int(m2.group(1)):04d}-{int(m2.group(2)):02d}-{int(m2.group(3)):02d}"
        m3 = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", raw)
        if m3:
            mm = month_map.get((m3.group(2) or "").lower())
            if mm:
                return f"{int(m3.group(3)):04d}-{mm}-{int(m3.group(1)):02d}"
        m4 = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\b", raw)
        if m4:
            mm = month_map.get((m4.group(1) or "").lower())
            if mm:
                return f"{int(m4.group(3)):04d}-{mm}-{int(m4.group(2)):02d}"
        m5 = re.search(r"\b(19|20)\d{2}\b", raw)
        if m5:
            return f"{int(m5.group(0)):04d}-01-01"
    return ""


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _safe_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return 0


_BAD_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize_for_llm(text: str, max_chars: int = 4000) -> str:
    """Normalize notes text into transport-safe compact content."""
    if not text:
        return ""
    t = str(text)
    t = _BAD_CONTROL_CHARS_RE.sub("", t)
    t = _SURROGATE_RE.sub("", t)
    t = t.encode("utf-8", errors="replace").decode("utf-8")
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) > max_chars:
        t = t[:max_chars] + f"\n...[truncated {len(t) - max_chars} chars]"
    return t


def _compact_pipeline_notes(notes: str, max_lines: int = 60, max_chars: int = 4000) -> str:
    """
    Return only the most relevant pipeline log lines to keep LLM context small/stable.
    """
    if not notes:
        return ""
    lines = [ln.strip() for ln in str(notes).splitlines() if ln.strip()]
    important = [
        ln for ln in lines
        if ("[Agent/" in ln)
        or ("[INCOMPLETE]" in ln)
        or ("[WARNING]" in ln)
        or ("HUMAN REVIEW SUMMARY" in ln)
    ]
    selected = important if important else lines
    compact = "\n".join(selected[-max_lines:])
    return _sanitize_for_llm(compact, max_chars=max_chars)


def _fix_mojibake(text: str) -> str:
    """Repeatedly decode latin1-as-utf8 mojibake until stable."""
    t = text or ""
    for _ in range(6):
        try:
            fixed = t.encode("latin-1").decode("utf-8")
        except Exception:
            break
        if fixed == t:
            break
        t = fixed
    return t


def _clean_field_text(text: str, max_chars: int = 12000) -> str:
    """
    Clean corrupted field text (title/abstract), removing repeated mojibake
    noise while preserving valid prose.
    """
    if not text:
        return ""
    t = _fix_mojibake(str(text))
    t = _BAD_CONTROL_CHARS_RE.sub("", t)
    t = _SURROGATE_RE.sub("", t)

    # Remove long repeated mojibake blocks like "Ã?Â??Ã?Â???..."
    t = re.sub(r"(?:Ã\?Â\?+|Ã\?Â|Ã\?|Ã|Â|\?){20,}", " ", t)
    t = re.sub(r"�{2,}", " ", t)

    # Normalize unicode and spacing
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip() + " ..."
    return t


def _detect_nanohub_evidence(citation) -> Dict[str, Any]:
    """
    Detect explicit nanoHUB/NCN evidence from citation text fields.
    Returns {'found': bool, 'matches': [..]}.
    """
    blobs = [
        ("title", citation.title or ""),
        ("abstract", citation.abstract or ""),
        ("notes", citation.notes or ""),
        ("url", citation.url or ""),
        ("ref_type", citation.ref_type or ""),
    ]
    patterns = [
        r"\bnanohub\b",
        r"\bnano[-\s]?hub\b",
        r"\bnetwork for computational nanotechnology\b",
        r"\bncn\b",
    ]
    matches: List[str] = []
    for field, text in blobs:
        t = (text or "").lower()
        for p in patterns:
            if re.search(p, t):
                matches.append(f"{field}:{p}")
                break
    return {"found": len(matches) > 0, "matches": matches}


def _detect_nanohub_evidence_in_pdf(cit_client, citation_id: int) -> Dict[str, Any]:
    """
    Scan PDF text for nanoHUB/NCN evidence.
    """
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return {"found": False, "matches": [], "error": "PyPDF2 not available"}

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        cit_client.download_pdf(citation_id, path)
        reader = PdfReader(path)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        low = text.lower()

        patterns = [
            r"\bnanohub\b",
            r"\bnano[-\s]?hub\b",
            r"\bnetwork for computational nanotechnology\b",
            r"\bncn\b",
            r"\bnanohub\.org\b",
        ]
        matches: List[str] = []
        for p in patterns:
            if re.search(p, low):
                matches.append(f"pdf:{p}")
        return {"found": len(matches) > 0, "matches": matches}
    except Exception as exc:
        return {"found": False, "matches": [], "error": str(exc)}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s<>\"]+", text, flags=re.IGNORECASE)
    out: List[str] = []
    seen = set()
    for u in urls:
        u = u.strip().rstrip(").,;")
        k = u.lower()
        if not u or k in seen:
            continue
        seen.add(k)
        out.append(u)
    return out


def _build_association_context(citation) -> Dict[str, Any]:
    """
    Build best-effort association context from citation fields/notes.
    """
    links: List[str] = []
    if citation.url:
        links.append(citation.url.strip())
    links.extend(_extract_urls(citation.notes or ""))
    links.extend(_extract_urls(citation.abstract or ""))
    # de-dup
    uniq_links: List[str] = []
    seen = set()
    for u in links:
        k = u.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq_links.append(u)

    parsed = []
    for u in uniq_links:
        lu = u.lower()
        if "nanohub.org/resources/" in lu or "nanohub.org/tools/" in lu:
            m = re.search(r"nanohub\.org/(?:resources?|tools)/([A-Za-z0-9_\-]+)", u, re.IGNORECASE)
            parsed.append({
                "type": "resource",
                "id_or_alias": m.group(1) if m else "",
                "url": u,
                "explanation": "nanoHUB tool/resource reference detected in citation context.",
            })
        elif "nanohub.org/publications/" in lu:
            m = re.search(r"nanohub\.org/publications/(\d+)", u, re.IGNORECASE)
            parsed.append({
                "type": "publication",
                "id_or_alias": m.group(1) if m else "",
                "url": u,
                "explanation": "nanoHUB publication reference detected in citation context.",
            })
        elif "nanohub.org/" in lu:
            parsed.append({
                "type": "link",
                "id_or_alias": "1",
                "url": u,
                "explanation": "Generic nanoHUB page citation (mapped as link:1).",
            })
        elif "/doi/" in lu or "doi.org/" in lu:
            parsed.append({
                "type": "evidence_url",
                "id_or_alias": "",
                "url": u,
                "explanation": "Publisher DOI page used as bibliographic evidence source.",
            })

    return {
        "count": len(parsed),
        "items": parsed[:30],
    }


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
            "3. Call `get_association_context` to retrieve related associations/links.\n"
            "4. Synthesize the information and call `write_review_summary` with a structured "
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
            "   [8] ASSOCIATIONS\n"
            "       List any related associations and explain each briefly: "
            "resource/publication/link/evidence URLs.\n\n"
            "5. End with a plain-text confirmation that the summary has been written.\n\n"
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
                "name": "get_association_context",
                "description": (
                    "Return best-effort association context derived from citation URL/notes/abstract, "
                    "including resource/publication/link references and evidence URLs."
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
                raw_notes = citation.notes or ""
                compact_notes = _compact_pipeline_notes(raw_notes)
                return {
                    "notes": compact_notes or "(no notes recorded)",
                    "notes_full_length": len(raw_notes),
                    "notes_compacted": bool(raw_notes and compact_notes != raw_notes),
                    "exp_list_exp_data": citation.exp_list_exp_data,
                    "exp_data": citation.exp_data,
                    "affiliated": citation.affiliated,
                    "ref_type": citation.ref_type,
                    "status": citation.status,
                }
            except Exception as exc:
                return {"error": str(exc)}

        if name == "get_association_context":
            try:
                citation = self.cit_client.get(cid)
                return _build_association_context(citation)
            except Exception as exc:
                return {"error": str(exc)}

        if name == "write_review_summary":
            summary = input_data.get("summary", "")
            try:
                citation = self.cit_client.get(cid)
                field_evidence = _detect_nanohub_evidence(citation)
                pdf_evidence = _detect_nanohub_evidence_in_pdf(self.cit_client, cid)
                evidence_found = bool(field_evidence.get("found") or pdf_evidence.get("found"))
                assoc_ctx = _build_association_context(citation)

                # Guardrail: don't allow REJECT if explicit nanoHUB/NCN evidence exists.
                if evidence_found:
                    if re.search(r"\bREJECT\b", summary, re.IGNORECASE):
                        summary = re.sub(
                            r"\bREJECT\b",
                            "NEEDS ATTENTION",
                            summary,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                        summary += (
                            "\n\n[Auto-guard] Recommendation adjusted from REJECT to NEEDS ATTENTION "
                            "because explicit nanoHUB/NCN evidence was detected "
                            "in citation fields and/or PDF text."
                        )

                # Ensure ASSOCIATIONS section exists in final summary.
                if not re.search(r"\bASSOCIATIONS\b", summary, re.IGNORECASE):
                    lines = ["\n[8] ASSOCIATIONS"]
                    if assoc_ctx.get("count", 0) == 0:
                        lines.append("- No explicit association URLs found in citation fields/notes.")
                    else:
                        for item in assoc_ctx.get("items", [])[:10]:
                            atype = item.get("type", "link")
                            ident = item.get("id_or_alias", "")
                            ident_txt = f": {ident}" if ident else ""
                            lines.append(
                                f"- {atype}{ident_txt} -> {item.get('url', '')} "
                                f"({item.get('explanation', '')})"
                            )
                    summary = summary.rstrip() + "\n" + "\n".join(lines)

                # Prepend the human review summary at the top of notes,
                # keeping previous agent logs below a separator.
                existing = citation.notes or ""
                separator = "\n" + "-" * 50 + "\n[Previous pipeline logs]\n"
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

    def _extract_member_candidates(self, payload: Any) -> List[Dict[str, Any]]:
        """Normalize different members/list payload shapes into candidate dicts."""
        out: List[Dict[str, Any]] = []

        def _walk(node: Any):
            if isinstance(node, dict):
                keys = {k.lower() for k in node.keys()}
                if (
                    "uidnumber" in keys
                    or "username" in keys
                    or "email" in keys
                    or "name" in keys
                ):
                    out.append(node)
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)
        return out

    def _find_best_member_match(self, author: Dict[str, Any], candidates: List[Dict[str, Any]]) -> int:
        """Score candidates by email/name and return best uidNumber (or 0)."""
        if not candidates:
            return 0

        fn = _norm(author.get("firstname") or author.get("firstName"))
        ln = _norm(author.get("lastname") or author.get("lastName"))
        fullname = _norm(f"{fn} {ln}")
        email = _norm(author.get("email"))

        best_uid = 0
        best_score = -1

        for c in candidates:
            uid = _safe_int(c.get("uidNumber") or c.get("uidnumber") or c.get("id"))
            if uid <= 0:
                continue

            c_name = _norm(c.get("name"))
            c_username = _norm(c.get("username"))
            c_email = _norm(c.get("email"))
            c_fullname = _norm(
                f"{c.get('firstname') or c.get('firstName') or ''} {c.get('lastname') or c.get('lastName') or ''}"
            )

            score = 0
            if email and c_email and email == c_email:
                score += 10
            if fullname and c_name and fullname == c_name:
                score += 8
            if fullname and c_fullname and fullname == c_fullname:
                score += 8
            if ln and c_name and ln in c_name:
                score += 2
            if fn and c_name and fn in c_name:
                score += 1
            if fn and ln and c_username and (fn in c_username or ln in c_username):
                score += 1

            if score > best_score:
                best_score = score
                best_uid = uid

        # Require at least some credible match signal.
        return best_uid if best_score >= 3 else 0

    def _enrich_nanohub_ids(self, citation_id: int) -> Dict[str, Any]:
        """
        Resolve nanoHUB IDs (cid) for authors using members/list endpoint.
        """
        if _requests is None:
            return {"ok": False, "error": "requests module is not available"}

        citation = self.cit_client.get(citation_id)
        authors = list(citation.authors or [])
        if not authors:
            return {"ok": True, "updated": 0, "checked": 0}

        endpoint = "https://nanohub.org/api/members/list"
        checked = 0
        updated = 0

        for idx, author in enumerate(authors):
            if not isinstance(author, dict):
                continue

            existing_cid = _safe_int(author.get("cid"))
            if existing_cid > 0:
                continue

            checked += 1
            email = (author.get("email") or "").strip()
            firstname = (author.get("firstname") or author.get("firstName") or "").strip()
            lastname = (author.get("lastname") or author.get("lastName") or "").strip()
            name_query = f"{firstname} {lastname}".strip()

            search_terms = []
            if email:
                search_terms.append(email)
            if name_query:
                search_terms.append(name_query)

            best_uid = 0
            for term in search_terms:
                try:
                    resp = _requests.get(
                        endpoint,
                        params={"search": term},
                        timeout=12,
                        headers={"accept": "application/json"},
                    )
                    if resp.status_code >= 400:
                        continue
                    payload = resp.json()
                    candidates = self._extract_member_candidates(payload)
                    best_uid = self._find_best_member_match(author, candidates)
                    if best_uid > 0:
                        break
                except Exception:
                    continue

            if best_uid > 0:
                authors[idx]["cid"] = best_uid
                updated += 1

        if updated == 0:
            return {"ok": True, "updated": 0, "checked": checked}

        # Avoid sending personId/id so backend applies mapped fields (including cid).
        sanitized_authors: List[Dict[str, Any]] = []
        for a in authors:
            if not isinstance(a, dict):
                continue
            b = dict(a)
            b.pop("id", None)
            b.pop("personId", None)
            b.pop("personid", None)
            sanitized_authors.append(b)

        citation.authors = sanitized_authors
        existing = citation.notes or ""
        sep = "\n" if existing else ""
        citation.notes = f"{existing}{sep}[Agent/{self.STAGE_NAME}] Auto-enriched nanoHUB IDs for {updated} author(s)."
        self.cit_client.update(citation)
        return {"ok": True, "updated": updated, "checked": checked}

    def _sanitize_text_fields(self, citation_id: int) -> Dict[str, Any]:
        """
        Validate/clean title and abstract for encoding-noise corruption.
        """
        try:
            citation = self.cit_client.get(citation_id)
            changed_fields: List[str] = []
            old_title = citation.title or ""
            old_abstract = citation.abstract or ""

            new_title = _clean_field_text(old_title, max_chars=600)
            new_abstract = _clean_field_text(old_abstract, max_chars=12000)

            # Only persist meaningful, non-destructive changes.
            if new_title and new_title != old_title and len(new_title) >= max(8, int(len(old_title) * 0.35)):
                citation.title = new_title
                changed_fields.append("title")
            if new_abstract and new_abstract != old_abstract and len(new_abstract) >= max(50, int(len(old_abstract) * 0.25)):
                citation.abstract = new_abstract
                changed_fields.append("abstract")

            if not changed_fields:
                return {"ok": True, "changed": []}

            existing = citation.notes or ""
            sep = "\n" if existing else ""
            citation.notes = (
                f"{existing}{sep}[Agent/{self.STAGE_NAME}] "
                f"Cleaned encoding noise in fields: {', '.join(changed_fields)}."
            )
            self.cit_client.update(citation)
            return {"ok": True, "changed": changed_fields}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

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

        text_fix_result = self._sanitize_text_fields(citation_id)
        if text_fix_result.get("ok"):
            changed = text_fix_result.get("changed") or []
            if changed:
                print(f"  [human_review] cleaned text fields: {', '.join(changed)}")
        else:
            print(f"  [human_review] warning: {text_fix_result.get('error')}")

        cid_result = self._enrich_nanohub_ids(citation_id)
        if cid_result.get("ok"):
            print(
                "  [human_review] nanoHUB ID enrichment: "
                f"updated={cid_result.get('updated', 0)} checked={cid_result.get('checked', 0)}"
            )
        else:
            print(f"  [human_review] warning: {cid_result.get('error')}")

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

"""
Status 3 — Classification Agent

Checks that all mandatory metadata fields are present and fills gaps from the PDF.

Fields checked / filled:
  Required  : title, authors (with org/dept/email/orcid), year, abstract,
              document_genre_name, ref_type, affiliated
  Recommended (filled but non-blocking): doi, publication_name, volume,
              begin_page, end_page, keywords

ref_type codes (comma-separated):
  R = Research paper
  C = Cyberinfrastructure / software tool
  E = Education / teaching material
  N = Nanotechnology topic

affiliated:
  1 = paper uses or mentions nanoHUB tools / NCN (Network for Computational
      Nanotechnology) in text, acknowledgements, or references
  0 = no nanoHUB / NCN connection found

Author enrichment:
  Each author should have organizationname, organizationdept, organizationtype,
  email, orcid, and countryresident filled from the PDF header/affiliations.
"""

import os
import re
import tempfile
import unicodedata
from typing import Any, Dict, List, Optional

try:
    import requests as _requests
except ImportError:
    _requests = None

from .base_agent import BaseCitationAgent


def _resolve_nanohub_resource(url_or_alias: str) -> Optional[int]:
    """
    Extract or resolve a nanoHUB resource to a numeric ID.
    Handles:
      - https://nanohub.org/resources/1308        → 1308
      - https://nanohub.org/resources/ucb_compnano → fetch ID from API
      - https://nanohub.org/tools/ucb_compnano     → fetch ID from API
      - plain alias string like 'ucb_compnano'     → fetch ID from API
    """
    # Try numeric ID from URL first
    m = re.search(r"nanohub\.org/(?:resources?|tools)/(\d+)", url_or_alias, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Extract alias from URL or use the string directly
    m = re.search(r"nanohub\.org/(?:resources?|tools)/([A-Za-z0-9_\-]+)", url_or_alias, re.IGNORECASE)
    alias = m.group(1) if m else url_or_alias.strip()

    if not alias or alias.isdigit():
        return int(alias) if alias.isdigit() else None

    if _requests is None:
        return None

    # Use the search API to resolve alias → numeric ID
    # Use a fresh Session per call to avoid exhausting the nanohubremote connection pool
    try:
        full_url = "https://nanohub.org/api/search/list"
        print(f"    [resolve_resource] GET {full_url}?terms={alias}&type=resource&section=alias")
        with _requests.Session() as s:
            resp = s.get(
                full_url,
                params={"terms": alias, "type": "resource", "section": "alias", "limit": 10},
                timeout=10,
            )
            print(f"    [resolve_resource] status={resp.status_code} body={resp.text[:300]}")
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                rid = None
                # First pass: exact alias match in URL + type is Tools
                for item in results:
                    item_id = item.get("id", "")
                    item_url = item.get("url", "")
                    item_type = item.get("type", "")
                    is_exact = alias.lower() in item_url.lower()
                    is_tool = item_type.lower() == "tools"
                    m = re.search(r"(\d+)$", item_id)
                    if m and is_exact and is_tool:
                        rid = int(m.group(1))
                        break
                if rid is None:
                    # Second pass: exact alias match in URL (any type)
                    for item in results:
                        item_id = item.get("id", "")
                        item_url = item.get("url", "")
                        is_exact = alias.lower() in item_url.lower()
                        m = re.search(r"(\d+)$", item_id)
                        if m and is_exact:
                            rid = int(m.group(1))
                            break
                if rid is None:
                    # Last resort: first Tools item
                    for item in results:
                        item_id = item.get("id", "")
                        item_type = item.get("type", "")
                        if item_type.lower() == "tools":
                            m = re.search(r"(\d+)$", item_id)
                            if m:
                                rid = int(m.group(1))
                                break
                return rid
    except Exception as exc:
        print(f"    [resolve_resource] error: {exc}")

    return None


REQUIRED_FIELDS = [
    "title",
    "authors",            # at least one author with organization filled
    "year",
    "abstract",
    "document_genre_name",
    "ref_type",           # R/C/E/N flags
    "affiliated",         # nanoHUB / NCN affiliation check (0 or 1)
]

RECOMMENDED_FIELDS = [
    "doi",
    "publication_name",
    "volume",
    "begin_page",
    "keywords",
]

# NCN / nanoHUB keywords to detect affiliation
_NCN_KEYWORDS = [
    "nanohub", "nano-hub", "ncn", "network for computational nanotechnology",
]


_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

_DATE_RE = re.compile(
    r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
)


_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)


def _parse_date(text: str) -> Optional[str]:
    """Parse 'DD Month YYYY' into 'YYYY-MM-DD', or return None."""
    m = _DATE_RE.search(text)
    if not m:
        return None
    month = _MONTHS.get(m.group(2).lower())
    if not month:
        return None
    return f"{m.group(3)}-{month}-{int(m.group(1)):02d}"


def _parse_ymd_date(text: str) -> Optional[str]:
    """
    Parse common date formats into YYYY-MM-DD.
    Supports: YYYY-MM-DD, YYYY/MM/DD, DD Month YYYY, Month DD, YYYY.
    """
    if not text:
        return None
    t = text.strip()

    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    ddmmyyyy = _parse_date(t)
    if ddmmyyyy:
        return ddmmyyyy

    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\b", t)
    if m:
        month = _MONTHS.get(m.group(1).lower())
        if month:
            return f"{int(m.group(3)):04d}-{month}-{int(m.group(2)):02d}"

    m = re.search(r"\b(19|20)\d{2}\b", t)
    if m:
        y = int(m.group(0))
        return f"{y:04d}-01-01"
    return None


def _html_to_text(html: str) -> str:
    """Best-effort HTML to plain text without extra dependencies."""
    if not html:
        return ""
    # Drop scripts/styles first
    html = re.sub(r"(?is)<script\b.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style\b.*?>.*?</style>", " ", html)
    # Convert tags to line breaks/spaces
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p\s*>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_publish_date_from_html_or_text(html: str, text: str) -> Optional[str]:
    """Extract publication date from common meta tags or visible text."""
    # Common metadata tags
    meta_patterns = [
        r'(?is)<meta[^>]+(?:name|property)=["\']citation_publication_date["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+(?:name|property)=["\']dc\.date["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+(?:name|property)=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+(?:name|property)=["\']prism\.publicationdate["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for pattern in meta_patterns:
        m = re.search(pattern, html)
        if m:
            parsed = _parse_ymd_date(m.group(1))
            if parsed:
                return parsed

    # Visible text fallbacks
    text_patterns = [
        r"Published(?:\s+online)?\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"Publication\s+date\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"Published(?:\s+online)?\s*[:\-]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    ]
    for pattern in text_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            parsed = _parse_ymd_date(m.group(1))
            if parsed:
                return parsed
    return None


def _extract_affiliation_candidates(text: str) -> List[str]:
    """Extract likely affiliation lines from webpage text."""
    if not text:
        return []
    candidates: List[str] = []
    keywords = (
        "university", "institute", "school", "department", "college",
        "laboratory", "lab", "center", "centre", "hospital", "technology",
    )
    for raw_line in text.splitlines():
        line = raw_line.strip(" \t,;:-")
        if len(line) < 8 or len(line) > 180:
            continue
        low = line.lower()
        if any(k in low for k in keywords):
            # Keep lines that look like proper affiliation phrases
            if re.search(r"[A-Za-z]{3,}", line):
                candidates.append(_clean_text(line))
    # Deduplicate preserving order
    seen = set()
    out = []
    for c in candidates:
        if not c or c.lower() in seen:
            continue
        seen.add(c.lower())
        out.append(c)
    return out[:30]


def _extract_submission_dates(text: str) -> Dict[str, Optional[str]]:
    """
    Scan PDF text for Received / Revised / Accepted date lines.
    Handles both single-line ('Received: DD Month YYYY / Revised: ...')
    and multi-line formats.
    Returns dict with keys: date_submit, date_revised, date_accept.
    """
    dates: Dict[str, Optional[str]] = {
        "date_submit": None,
        "date_revised": None,
        "date_accept": None,
    }
    # Normalise line endings and collapse whitespace for easier matching
    flat = re.sub(r"\s+", " ", text)
    patterns = [
        ("date_submit",  r"Received\s*[:\-]\s*([^/\n;]{5,40?})"),
        ("date_revised", r"Revised\s*[:\-]\s*([^/\n;]{5,40?})"),
        ("date_accept",  r"Accepted\s*[:\-]\s*([^/\n;]{5,40?})"),
    ]
    for key, pattern in patterns:
        m = re.search(pattern, flat, re.IGNORECASE)
        if m:
            parsed = _parse_date(m.group(1))
            if parsed:
                dates[key] = parsed
    return dates


def _fix_mojibake(text: str) -> str:
    """Repeatedly decode latin-1-as-UTF-8 mojibake until the text stabilises."""
    for _ in range(6):  # up to 6 layers of mis-encoding
        try:
            fixed = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if fixed == text:
            break
        text = fixed
    return text


def _ascii(text: str) -> str:
    """Normalize unicode to ASCII for safe API storage."""
    if not text:
        return ""
    text = _fix_mojibake(text)
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii").strip()


def _clean_text(text: str) -> str:
    """Clean PDF-extracted text: fix mojibake, normalize unicode, drop non-ASCII."""
    if not text:
        return ""
    text = _fix_mojibake(text)
    # NFKD decomposes accented characters (é → e + combining accent)
    normalized = unicodedata.normalize("NFKD", text)
    # Drop combining/non-ASCII characters
    cleaned = normalized.encode("ascii", "ignore").decode("ascii")
    # Collapse extra whitespace left by dropped characters
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class ClassificationAgent(BaseCitationAgent):
    """Verify and complete citation metadata at status 3."""

    STAGE_NAME = "classification"
    TARGET_STATUS = 3
    NEXT_STATUS = 4

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @property
    def _system_prompt(self) -> str:
        return (
            "You are a citation pipeline agent responsible for Step 3: Classification.\n\n"
            "Your job is to make sure a citation has COMPLETE metadata before it moves to review.\n\n"
            "=== REQUIRED FIELDS ===\n"
            "title, authors (with organisation filled), year, abstract, document_genre_name,\n"
            "ref_type, affiliated\n\n"
            "=== FIELD RULES ===\n\n"
            "abstract:\n"
            "  If missing, call extract_pdf_text, then write a concise academic abstract\n"
            "  (150-250 words) based on the actual PDF content. Save with update_citation_fields.\n\n"
            "ref_type  (comma-separated flags — pick all that apply):\n"
            "  R = research paper (reports new findings or analysis)\n"
            "  C = cyberinfrastructure / software tool / simulation\n"
            "  E = educational material / teaching resource\n"
            "  N = nanotechnology topic\n"
            "  Examples: 'R', 'R,N', 'R,C,N', 'E,N'\n"
            "  Determine from the PDF title, abstract, and keywords.\n"
            "  Save with update_citation_fields({'ref_type': '...'})\n\n"
            "affiliated  (integer):\n"
            "  1 ONLY if at least one author's last name matches one of these NCN researchers:\n"
            "    Klimeck, Strachan, Mejia, Verduzco, Faltens\n"
            "  (case-insensitive, partial match is fine, e.g. 'Verduzc' matches 'Verduzco')\n"
            "  0 otherwise.\n"
            "  Check the author list already in the citation record — no PDF scan needed.\n"
            "  Save with update_citation_fields({'affiliated': 1}) or 0.\n\n"
            "dates (Received / Revised / Accepted):\n"
            "  Look in the PDF text for lines like:\n"
            "    Received: DD Month YYYY  /  Revised: DD Month YYYY  /  Accepted: DD Month YYYY\n"
            "  Map them as follows:\n"
            "    Received  → date_submit  (format YYYY-MM-DD)\n"
            "    Accepted  → date_accept  (format YYYY-MM-DD)\n"
            "    Revised   → store in notes as 'Revised: YYYY-MM-DD' if found\n"
            "  Save date_submit / date_accept with update_citation_fields if found in the PDF.\n"
            "  If not found, leave these fields empty — do not invent dates.\n\n"
            "authors:\n"
            "  If any author has organization_name = 'N/A' or empty email/orcid, call\n"
            "  extract_pdf_text to read the first page author block, then call\n"
            "  update_authors with the enriched list.\n"
            "  If affiliations are unreadable in the PDF (e.g., affiliation block is an image),\n"
            "  call extract_web_context and use citation webpage/reference URLs to recover\n"
            "  organizationname/organizationdept.\n"
            "  For each author extract: organizationname, organizationdept,\n"
            "  organizationtype (Education / Industry / Government / Research),\n"
            "  countryresident (2-letter ISO code if determinable), email, orcid.\n"
            "  Keep firstname/lastname exactly as they already appear in the record.\n\n"
            "date_publish:\n"
            "  If date_publish is empty, try to obtain publication date from web metadata\n"
            "  by calling extract_web_context. Save date_publish via update_citation_fields.\n\n"
            "nanoHUB resource associations:\n"
            "  While reading the PDF, scan for any nanoHUB resource URLs in the text,\n"
            "  references, or acknowledgements. Patterns to look for:\n"
            "    https://nanohub.org/resources/<id>\n"
            "    https://nanohub.org/tools/<id>\n"
            "    nanohub.org/resources/<id>\n"
            "  For each unique resource URL or ID found, call add_nanohub_resource.\n"
            "  Do this regardless of whether affiliated is 0 or 1.\n\n"
            "=== WORKFLOW ===\n"
            "1. Call get_citation_details.\n"
            "2. Call extract_pdf_text (needed for abstract, ref_type, affiliated, authors, resource URLs).\n"
            "3. If author affiliations remain incomplete or date_publish is missing, call extract_web_context.\n"
            "4. Fill every missing required field using the appropriate tool.\n"
            "5. Also fill recommended fields if you can determine them confidently.\n"
            "6. Call add_nanohub_resource for every nanoHUB resource URL found in the PDF.\n"
            "7. Once all required fields are filled, call advance_to_status_4.\n"
            "8. If a required field cannot be determined from the PDF/web context, call flag_incomplete.\n"
            "9. End with a plain-text summary of what was done.\n\n"
            "Be precise. Never invent data that is not present in the PDF or citation record."
        )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _tool_definitions(self) -> List[Dict]:
        return [
            {
                "name": "get_citation_details",
                "description": "Fetch the full citation record including all metadata and author fields.",
                "input_schema": {
                    "type": "object",
                    "properties": {"citation_id": {"type": "integer"}},
                    "required": ["citation_id"],
                },
            },
            {
                "name": "extract_pdf_text",
                "description": (
                    "Download the citation PDF and return its full text (all pages). "
                    "Use for writing the abstract, detecting nanoHUB/NCN affiliation, "
                    "determining ref_type, and extracting author affiliations."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"citation_id": {"type": "integer"}},
                    "required": ["citation_id"],
                },
            },
            {
                "name": "extract_web_context",
                "description": (
                    "Fetch citation URL and any URLs present in notes/references, then return "
                    "plain webpage text for metadata recovery (author affiliations and publish date). "
                    "Also auto-saves date_publish if confidently detected and currently empty."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"citation_id": {"type": "integer"}},
                    "required": ["citation_id"],
                },
            },
            {
                "name": "update_citation_fields",
                "description": (
                    "Update scalar metadata fields on the citation. "
                    "Accepted keys: title, abstract, year, doi, publication_name, "
                    "document_genre_name, volume, issue, begin_page, end_page, publisher, "
                    "ref_type (string), affiliated (integer 0 or 1), "
                    "keywords (list of strings), "
                    "date_submit (Received date, YYYY-MM-DD), "
                    "date_accept (Accepted date, YYYY-MM-DD), "
                    "date_publish (publication date, YYYY-MM-DD)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "fields": {
                            "type": "object",
                            "description": "Dict of field_name → new_value pairs.",
                        },
                    },
                    "required": ["citation_id", "fields"],
                },
            },
            {
                "name": "update_authors",
                "description": (
                    "Replace the citation's author list with enriched author data extracted "
                    "from the PDF. Each entry must include firstname and lastname (copied from "
                    "the existing record). Optional enrichment fields: organizationname, "
                    "organizationdept, organizationtype, countryresident, email, orcid, "
                    "scopusid, researcherid, gsid, researchgateid."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "authors": {
                            "type": "array",
                            "description": "List of author dicts with firstname, lastname, and optional fields.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "firstname":        {"type": "string"},
                                    "lastname":         {"type": "string"},
                                    "email":            {"type": "string"},
                                    "orcid":            {"type": "string"},
                                    "organizationname": {"type": "string"},
                                    "organizationdept": {"type": "string"},
                                    "organizationtype": {"type": "string"},
                                    "countryresident":  {"type": "string"},
                                    "scopusid":         {"type": "string"},
                                    "researcherid":     {"type": "string"},
                                    "gsid":             {"type": "string"},
                                    "researchgateid":   {"type": "string"},
                                },
                                "required": ["firstname", "lastname"],
                            },
                        },
                    },
                    "required": ["citation_id", "authors"],
                },
            },
            {
                "name": "advance_to_status_4",
                "description": "Advance the citation to status 4 (review). Call only when all required fields are filled.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "note": {"type": "string"},
                    },
                    "required": ["citation_id"],
                },
            },
            {
                "name": "add_nanohub_resource",
                "description": (
                    "Register a nanoHUB resource URL found in the PDF as an association on this citation. "
                    "Pass the full URL (e.g. https://nanohub.org/resources/1308) or just the numeric ID. "
                    "Call once per unique resource URL found in the PDF text."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "resource_url": {
                            "type": "string",
                            "description": "Full nanoHUB resource URL or numeric resource ID.",
                        },
                    },
                    "required": ["citation_id", "resource_url"],
                },
            },
            {
                "name": "flag_incomplete",
                "description": "Record which required fields could not be filled and keep the citation at status 3.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "missing_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "note": {"type": "string"},
                    },
                    "required": ["citation_id", "missing_fields"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, input_data: Dict) -> Dict:

        if name == "get_citation_details":
            return self._get_citation_dict(input_data["citation_id"])

        if name == "extract_pdf_text":
            cid = input_data["citation_id"]
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.close()
            try:
                self.cit_client.download_pdf(cid, tmp.name)
                from PyPDF2 import PdfReader
                reader = PdfReader(tmp.name)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                clean = _clean_text(text)

                # Auto-register any nanoHUB resource URLs found in the PDF
                # Match both numeric IDs and named aliases
                raw_urls = set(re.findall(
                    r"nanohub\.org/(?:resources?|tools)/([A-Za-z0-9_\-]+)",
                    text, re.IGNORECASE,
                ))
                registered = []
                for raw in raw_urls:
                    rid = _resolve_nanohub_resource(f"nanohub.org/resources/{raw}")
                    if rid is None:
                        print(f"    [DocumentAssociation] could not resolve resource: {raw}")
                        continue
                    try:
                        result = self.cit_client._api_call(
                            "DocumentAssociation",
                            {
                                "action": "add",
                                "idDocument": cid,
                                "assocName": "resource",
                                "assocID": rid,
                            },
                        )
                        print(f"    [DocumentAssociation] resource={raw}({rid}) result={result}")
                        registered.append(rid)
                    except Exception as exc:
                        if "already exists" in str(exc).lower():
                            registered.append(rid)
                        else:
                            print(f"    [DocumentAssociation] resource={raw}({rid}) error={exc}")

                # Auto-save Received / Revised / Accepted dates from raw text
                dates = _extract_submission_dates(text)
                dates_saved = {}
                try:
                    citation = self.cit_client.get(cid)
                    changed = False
                    for field, value in [
                        ("date_submit", dates["date_submit"]),
                        ("date_accept", dates["date_accept"]),
                    ]:
                        if value and not getattr(citation, field, None):
                            setattr(citation, field, value)
                            dates_saved[field] = value
                            changed = True
                    if dates["date_revised"]:
                        dates_saved["date_revised"] = dates["date_revised"]
                    if changed:
                        self.cit_client.update(citation)
                        print(f"    [dates] saved {dates_saved}")
                except Exception as exc:
                    print(f"    [dates] error saving: {exc}")

                return {
                    "ok": True,
                    "text": clean,
                    "total_pages": len(reader.pages),
                    "nanohub_resources_registered": registered,
                    "dates_saved": dates_saved,
                    "note": "nanoHUB resource associations and dates have already been registered — do not call add_nanohub_resource for these IDs.",
                }
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        if name == "extract_web_context":
            cid = input_data["citation_id"]
            if _requests is None:
                return {"ok": False, "error": "requests module is not available"}
            try:
                citation = self.cit_client.get(cid)
                candidate_urls: List[str] = []
                if citation.url and citation.url.strip():
                    candidate_urls.append(citation.url.strip())

                for blob in [citation.notes or "", citation.abstract or ""]:
                    for u in _URL_RE.findall(blob):
                        candidate_urls.append(u.strip().rstrip(").,;"))

                # Deduplicate while preserving order
                seen = set()
                urls = []
                for u in candidate_urls:
                    key = u.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    urls.append(u)
                urls = urls[:5]

                pages = []
                all_affiliations: List[str] = []
                detected_date = None

                with _requests.Session() as s:
                    for url in urls:
                        try:
                            resp = s.get(
                                url,
                                timeout=12,
                                allow_redirects=True,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (citation-pipeline/1.0)"
                                },
                            )
                            if resp.status_code >= 400:
                                continue
                            html = resp.text[:400000]
                            text = _clean_text(_html_to_text(html))
                            if not text:
                                continue
                            pages.append({
                                "url": url,
                                "text_excerpt": text[:3500],
                            })

                            found_date = _extract_publish_date_from_html_or_text(html, text)
                            if found_date and not detected_date:
                                detected_date = found_date

                            all_affiliations.extend(_extract_affiliation_candidates(text))
                        except Exception:
                            continue

                date_saved = None
                if detected_date and not citation.date_publish:
                    citation.date_publish = detected_date
                    self.cit_client.update(citation)
                    date_saved = detected_date

                # Deduplicate affiliation list
                dedup_affs = []
                seen_affs = set()
                for a in all_affiliations:
                    k = a.lower()
                    if k in seen_affs:
                        continue
                    seen_affs.add(k)
                    dedup_affs.append(a)

                combined_text = "\n\n".join(
                    f"[URL] {p['url']}\n{p['text_excerpt']}" for p in pages
                )
                return {
                    "ok": True,
                    "urls_checked": urls,
                    "pages_loaded": len(pages),
                    "text": combined_text[:15000],
                    "candidate_affiliations": dedup_affs[:25],
                    "date_publish_detected": detected_date,
                    "date_publish_saved": date_saved,
                }
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        if name == "update_citation_fields":
            cid = input_data["citation_id"]
            fields: Dict = input_data.get("fields", {})
            try:
                citation = self.cit_client.get(cid)
                updated = []
                for key, value in fields.items():
                    if key == "keywords" and isinstance(value, list):
                        citation.keywords = [_clean_text(k) for k in value]
                        updated.append("keywords")
                    elif hasattr(citation, key):
                        setattr(citation, key, _clean_text(value) if isinstance(value, str) else value)
                        updated.append(key)
                self.cit_client.update(citation)
                return {"ok": True, "updated_fields": updated}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        if name == "update_authors":
            cid = input_data["citation_id"]
            raw_authors: List[Dict] = input_data.get("authors", [])
            try:
                citation = self.cit_client.get(cid)

                # ── Step 1: deduplicate existing DB author records ──────────
                # The PHP API appends rather than replaces, so prior failed runs
                # may leave stale N/A duplicate entries. Group by sig and keep
                # only the best record per author; remove the rest.
                by_sig: Dict[str, List[Dict]] = {}
                for existing in citation.authors:
                    fn = (existing.get("firstname") or "").lower()
                    ln = (existing.get("lastname") or "").lower()
                    sig = f"{ln}_{fn[:1]}"
                    by_sig.setdefault(sig, []).append(existing)

                def _has_real_org(a: Dict) -> bool:
                    org = (a.get("organization_name") or a.get("organizationname") or "")
                    return bool(org) and org.upper() != "N/A"

                existing_by_sig: Dict[str, Dict] = {}
                for sig, entries in by_sig.items():
                    if len(entries) == 1:
                        existing_by_sig[sig] = entries[0]
                    else:
                        # Keep the entry with real org data; remove the rest
                        real = [e for e in entries if _has_real_org(e)]
                        keep = real[0] if real else entries[-1]
                        existing_by_sig[sig] = keep
                        for stale in entries:
                            if stale is keep:
                                continue
                            stale_id = stale.get("id")
                            if stale_id:
                                try:
                                    self.cit_client._api_call(
                                        "PersonDocument",
                                        {"action": "remove", "idDocument": cid, "idPerson": stale_id},
                                    )
                                except Exception:
                                    pass  # best-effort removal

                # ── Step 2: build enriched author list ─────────────────────
                clean: List[Dict] = []
                seen: set = set()
                for a in raw_authors:
                    fn = _ascii(a.get("firstname", ""))
                    ln = _ascii(a.get("lastname", ""))
                    sig = f"{ln.lower()}_{fn[:1].lower()}"
                    if sig in seen:
                        continue
                    seen.add(sig)

                    # Start from the (now-deduplicated) existing record to
                    # preserve the DB id so PHP updates in-place.
                    base = dict(existing_by_sig.get(sig, {}))
                    # IMPORTANT: do not send person id fields to CitationCRUD saveAuthors.
                    # Backend branch `if personId > 0` loads person but does not apply
                    # new organization fields; omitting ids forces the update path.
                    for id_key in ("id", "personId", "personid"):
                        if id_key in base:
                            base.pop(id_key, None)
                    base["firstname"] = fn
                    base["lastname"]  = ln

                    for field, api_key in [
                        ("email",            "email"),
                        ("orcid",            "orcid"),
                        ("organizationname", "organizationname"),
                        ("organizationtype", "organizationtype"),
                        ("organizationdept", "organizationdept"),
                        ("countryresident",  "countryresident"),
                        ("scopusid",         "scopusid"),
                        ("researcherid",     "researcherid"),
                        ("gsid",             "gsid"),
                        ("researchgateid",   "researchgateid"),
                    ]:
                        val = a.get(field, "")
                        if val:
                            is_ascii_field = field not in ("orcid", "scopusid", "researcherid", "gsid", "researchgateid", "countryresident")
                            base[api_key] = _ascii(val) if is_ascii_field else str(val).strip()

                    clean.append(base)

                citation.authors = clean
                self.cit_client.update(citation)
                return {"ok": True, "authors_updated": len(clean)}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        if name == "advance_to_status_4":
            cid = input_data["citation_id"]
            note = input_data.get("note", "All required fields present — advancing to review.")
            return self._advance_status(cid, self.NEXT_STATUS, note)

        if name == "add_nanohub_resource":
            cid = input_data["citation_id"]
            resource_url = input_data.get("resource_url", "").strip()
            try:
                resource_id = _resolve_nanohub_resource(resource_url)
                if resource_id is None:
                    return {"ok": False, "error": f"Cannot resolve resource ID from: {resource_url}"}

                result = self.cit_client._api_call(
                    "DocumentAssociation",
                    {
                        "action": "add",
                        "idDocument": cid,
                        "assocName": "resource",
                        "assocID": resource_id,
                    },
                )
                # _api_call raises on error status, so reaching here means success
                print(f"    [DocumentAssociation] resource={resource_id} result={result}")
                return {"ok": True, "resource_id": resource_id, "result": result}
            except Exception as exc:
                if "already exists" in str(exc).lower():
                    return {"ok": True, "resource_id": resource_id, "message": "association already registered"}
                return {"ok": False, "error": str(exc)}

        if name == "flag_incomplete":
            cid = input_data["citation_id"]
            missing = input_data.get("missing_fields", [])
            note = input_data.get("note", "")
            return self._append_note(cid, f"[INCOMPLETE] Missing: {missing}. {note}".strip())

        return {"error": f"Unknown tool: {name}"}

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, citation) -> str:
        author_orgs = [
            a.get("organization_name", a.get("organizationname", "N/A"))
            for a in (citation.authors or [])
        ]
        orgs_summary = ", ".join(set(author_orgs)) or "(none)"
        return (
            f"Please classify citation ID {citation.id}.\n\n"
            f"Current state:\n"
            f"  Title        : {citation.title or '(missing)'}\n"
            f"  Year         : {citation.year or '(missing)'}\n"
            f"  Authors      : {len(citation.authors or [])} (orgs: {orgs_summary[:80]})\n"
            f"  Abstract     : {'present' if citation.abstract else '(MISSING)'}\n"
            f"  Genre        : {citation.document_genre_name or '(missing)'}\n"
            f"  ref_type     : {citation.ref_type or '(MISSING)'}\n"
            f"  affiliated   : {citation.affiliated} (-1 = not set)\n"
            f"  DOI          : {citation.doi or '(missing)'}\n\n"
            "Extract the PDF text, then fill ALL required fields "
            "(abstract, ref_type, affiliated, author details), and advance to status 4."
        )

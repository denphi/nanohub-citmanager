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
from urllib.parse import unquote_plus, urlparse
from typing import Any, Dict, List, Optional

try:
    import requests as _requests
except ImportError:
    _requests = None

from .base_agent import BaseCitationAgent


def _extract_nanohub_urls(text: str) -> List[str]:
    """
    Extract nanoHUB URLs from text, including schema-less forms.
    """
    if not text:
        return []
    raw = re.findall(r"(https?://)?(?:www\.)?nanohub\.org/[^\s<>\]\"')]+", text, re.IGNORECASE)
    # regex with optional group returns tuples/strings depending on engine path; normalize
    urls = []
    if raw and isinstance(raw[0], tuple):
        # fallback path if groups captured
        raw_iter = re.finditer(r"((?:https?://)?(?:www\.)?nanohub\.org/[^\s<>\]\"')]+)", text, re.IGNORECASE)
        urls = [m.group(1) for m in raw_iter]
    else:
        raw_iter = re.finditer(r"((?:https?://)?(?:www\.)?nanohub\.org/[^\s<>\]\"')]+)", text, re.IGNORECASE)
        urls = [m.group(1) for m in raw_iter]

    out: List[str] = []
    seen = set()
    for u in urls:
        u = u.strip().rstrip(".,;:)]")
        if not u:
            continue
        if not re.match(r"^https?://", u, re.IGNORECASE):
            u = "https://" + u
        k = u.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(u)
    return out


def _extract_real_url_from_scholar_url(scholar_url: str) -> Optional[str]:
    """
    Extract the actual paper URL embedded in a Google Scholar query URL.

    Scholar often wraps a direct link as:
      https://scholar.google.com/scholar?...&q=https://some-publisher.org/paper&...
    or:
      https://scholar.google.com/scholar?...&q=https%3A%2F%2Fsome-publisher.org%2F...

    Returns the decoded URL if it looks like a valid external http(s) URL, else None.
    """
    if not scholar_url:
        return None
    try:
        parsed = urlparse(scholar_url)
        # Parse query string manually to handle repeated keys
        qs = parsed.query
        for part in qs.split("&"):
            if part.startswith("q="):
                value = unquote_plus(part[2:])
                if re.match(r"^https?://", value, re.IGNORECASE):
                    candidate_host = urlparse(value).netloc.lower()
                    if "google" not in candidate_host:
                        return value
    except Exception:
        pass
    return None


def _is_google_scholar_url(url: str) -> bool:
    """Return True if the URL points to Google Scholar."""
    if not url:
        return False
    try:
        host = urlparse(url).netloc.lower()
        return "scholar.google" in host
    except Exception:
        return False


def _extract_real_url_from_scholar_html(html: str) -> Optional[str]:
    """
    Try to extract the actual paper URL linked from a Google Scholar result page.

    Scholar embeds the full-text link in a few patterns:
      • <div class="gs_or_ggsm"><a href="...">   (green "PDF" / publisher links)
      • <a class="gs_ggsd" href="...">
      • data-clk-atid / data-href attributes on anchor tags inside .gs_ggs

    Returns the first plausible non-Scholar, non-Google URL found, or None.
    """
    if not html:
        return None

    # Pattern 1: <div class="gs_or_ggsm"><a href="..."> or <div class="gs_ggs gs_scl"><a href="...">
    for pattern in [
        r'class="gs_or_ggsm"[^>]*>.*?<a\s+href="([^"]+)"',
        r'class="gs_ggsd"[^>]*href="([^"]+)"',
        r'class="gs_ggs[^"]*"[^>]*>.*?<a\s+href="([^"]+)"',
    ]:
        m = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            parsed = urlparse(candidate)
            # Skip Scholar/Google internal links
            if parsed.scheme in ("http", "https") and "google" not in parsed.netloc.lower():
                return candidate

    return None


def _is_scholar_captcha_page(html: str, status_code: int) -> bool:
    """
    Detect whether Google Scholar returned a CAPTCHA / bot-challenge page.

    Signals:
      • HTTP 429 or 503
      • Page contains typical CAPTCHA markers
    """
    if status_code in (429, 503):
        return True
    if not html:
        return False
    low = html[:8000].lower()
    captcha_markers = [
        "unusual traffic",
        "not a robot",
        "recaptcha",
        "captcha",
        "sorry, we can",
        "automated queries",
        "ipv4.google.com/sorry",
    ]
    return any(marker in low for marker in captcha_markers)


def _resolve_nanohub_association(url_or_token: str) -> Optional[Dict[str, Any]]:
    """
    Resolve a nanoHUB URL/token to a DocumentAssociation tuple.

    Mapping rules:
      - /resources/<id|alias>, /tools/<id|alias> -> assocName='resource'
      - /publications/<id>/...                   -> assocName='publication'
      - any other nanohub.org page              -> assocName='link', assocID=1
    """
    token = (url_or_token or "").strip()
    if not token:
        return None

    # Backward compatibility: bare numeric token is treated as resource ID
    if token.isdigit():
        rid = int(token)
        return {"assoc_name": "resource", "assoc_id": rid, "source": token}

    # publication URLs
    m_pub = re.search(r"nanohub\.org/publications/(\d+)", token, re.IGNORECASE)
    if m_pub:
        return {
            "assoc_name": "publication",
            "assoc_id": int(m_pub.group(1)),
            "source": token,
        }

    # resource/tool URLs or aliases
    rid = _resolve_nanohub_resource(token)
    if rid is not None:
        return {"assoc_name": "resource", "assoc_id": rid, "source": token}

    # generic nanoHUB page citation -> link:1
    if re.search(r"nanohub\.org/", token, re.IGNORECASE):
        return {"assoc_name": "link", "assoc_id": 1, "source": token}

    return None


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
    "authors",            # at least one author listed
    "year",
    "abstract",
    "publication_name",
    "document_genre_name",
    "ref_type",           # R/C/E/N flags
    "affiliated",         # nanoHUB / NCN affiliation check (0 or 1)
]

RECOMMENDED_FIELDS = [
    "doi",
    "volume",
    "begin_page",
    "keywords",
    "date_publish",
]

# NCN / nanoHUB keywords to detect affiliation
_NCN_KEYWORDS = [
    "nanohub", "nano-hub", "ncn", "network for computational nanotechnology",
]
_NCN_STRICT_PATTERNS = [
    r"\bnetwork for computational nanotechnology\b",
    r"\b(?:the\s+)?ncn\b",
    r"\bncn[-\s]*(?:purdue|center|centre|program|initiative)\b",
]


_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

_DATE_RE = re.compile(
    r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_MOJIBAKE_MARKERS_RE = re.compile(r"(Ã|Â|�|Ã\?Â|\?{6,})")


def _is_thesis_genre(genre_name: str) -> bool:
    g = (genre_name or "").strip().lower()
    if not g:
        return False
    return any(token in g for token in ("thesis", "dissertation"))


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
    t = unquote_plus(text.strip()).replace("+", " ")

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

    # Embedded JSON / query-string style fields often present on publisher pages.
    raw_patterns = [
        r'(?is)\bpublicationDate=([^&"\']+)',
        r'(?is)"displayPublicationDate"\s*:\s*"([^"]+)"',
        r'(?is)"dateOfInsertion"\s*:\s*"([^"]+)"',
    ]
    for pattern in raw_patterns:
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
        r"First\s+Online\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"First\s+Online\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"First\s+Online\s*[:\-]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        r"(?:Date\s+of\s+Publication|Publication\s+Date)\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"(?:Date\s+of\s+Publication|Publication\s+Date)\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"(?:Date\s+of\s+Publication|Publication\s+Date)\s*[:\-]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    ]
    for pattern in text_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            parsed = _parse_ymd_date(m.group(1))
            if parsed:
                return parsed

    # Raw HTML fallback: some pages keep publication-date strings in embedded
    # markup/JSON that may be dropped by html-to-text conversion.
    html_flat = re.sub(r"\s+", " ", html or "")
    for pattern in text_patterns:
        m = re.search(pattern, html_flat, re.IGNORECASE)
        if m:
            parsed = _parse_ymd_date(m.group(1))
            if parsed:
                return parsed

    # Conference date variants, e.g.:
    # "Date of Conference: 22-28 May 2021"
    # "Conference Date: 22-28 May 2021"
    m = re.search(
        r"(?:Date\s+of\s+Conference|Conference\s+Date)\s*[:\-]?\s*(\d{1,2})\s*[-–]\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        text,
        re.IGNORECASE,
    )
    if m:
        day_start = int(m.group(1))
        month_name = m.group(3)
        year = int(m.group(4))
        month = _MONTHS.get(month_name.lower())
        if month and 1 <= day_start <= 31:
            return f"{year:04d}-{month}-{day_start:02d}"

    # Single-day conference date
    m = re.search(
        r"(?:Date\s+of\s+Conference|Conference\s+Date)\s*[:\-]?\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
        text,
        re.IGNORECASE,
    )
    if m:
        month = _MONTHS.get(m.group(2).lower())
        if month:
            return f"{int(m.group(3)):04d}-{month}-{int(m.group(1)):02d}"

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


def _extract_author_affiliations_from_text(text: str) -> List[Dict[str, str]]:
    """
    Extract author-affiliation pairs from web text patterns like:
      'Jane Doe (Institution Name, University ...)'
    """
    if not text:
        return []

    # Normalize whitespace to improve regex matching.
    flat = re.sub(r"\s+", " ", text)
    pairs: List[Dict[str, str]] = []

    # Name: 2-4 words, starts with capitals and may contain initials/periods/hyphens.
    # Affiliation in parentheses, fairly long, includes academic/org keywords.
    pattern = re.compile(
        r"\b([A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){1,3})\s*\(([^()]{8,220})\)"
    )
    org_keywords = (
        "university", "institute", "center", "centre", "school",
        "department", "laboratory", "college", "computing", "hospital",
    )

    for m in pattern.finditer(flat):
        name = _clean_text(m.group(1))
        aff = _clean_text(m.group(2))
        low = aff.lower()
        if not any(k in low for k in org_keywords):
            continue
        pairs.append({"name": name, "affiliation": aff})

    # Deduplicate
    out: List[Dict[str, str]] = []
    seen = set()
    for p in pairs:
        k = (p["name"].lower(), p["affiliation"].lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out[:200]


def _apply_web_author_affiliations(citation, pairs: List[Dict[str, str]]) -> int:
    """
    Apply extracted web author affiliations to citation authors by name match.
    Returns count of authors updated.
    """
    if not pairs or not getattr(citation, "authors", None):
        return 0

    # Build lookup by normalized full name and last name.
    by_full: Dict[str, str] = {}
    by_last: Dict[str, str] = {}
    for p in pairs:
        name = p.get("name", "")
        aff = p.get("affiliation", "")
        if not name or not aff:
            continue
        full = re.sub(r"\s+", " ", name).strip().lower()
        by_full[full] = aff
        parts = full.split()
        if parts:
            by_last[parts[-1]] = aff

    updated = 0
    for a in citation.authors:
        fn = (a.get("firstname") or a.get("firstName") or "").strip()
        ln = (a.get("lastname") or a.get("lastName") or "").strip()
        if not ln:
            continue

        current_org = (a.get("organization_name") or a.get("organizationname") or "").strip()
        if current_org and current_org.upper() != "N/A":
            continue

        full = re.sub(r"\s+", " ", f"{fn} {ln}").strip().lower()
        aff = by_full.get(full) or by_last.get(ln.lower())
        if not aff:
            continue

        a["organizationname"] = aff
        # Best effort department extraction from comma-separated affiliation.
        if not a.get("organizationdept"):
            first_chunk = aff.split(",")[0].strip()
            if "center" in first_chunk.lower() or "department" in first_chunk.lower() or "school" in first_chunk.lower():
                a["organizationdept"] = first_chunk
        if not a.get("organizationtype"):
            a["organizationtype"] = "Education"
        updated += 1

    return updated


def _extract_doi(text: str) -> Optional[str]:
    """Extract first DOI-like token from text."""
    if not text:
        return None
    m = _DOI_RE.search(text)
    if not m:
        return None
    doi = m.group(0).strip().rstrip(".,;)")
    return doi


def _extract_doi_from_url(url: str) -> Optional[str]:
    """Extract DOI from URL patterns such as /doi/full/<doi>."""
    if not url:
        return None
    u = url.strip()
    m = re.search(r"/doi/(?:abs|full|pdf)/([^?#]+)", u, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().strip("/")
        candidate = candidate.replace("%2F", "/").replace("%2f", "/")
        parsed = _extract_doi(candidate)
        if parsed:
            return parsed
        if candidate.startswith("10."):
            return candidate
    m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", u, re.IGNORECASE)
    if m:
        return m.group(0).rstrip(".,;)")
    return None


def _extract_publication_name_from_text(text: str) -> Optional[str]:
    """
    Extract likely conference/journal publication name from PDF header text.
    Prioritizes IEEE-style conference names.
    """
    if not text:
        return None
    flat = re.sub(r"\s+", " ", text).strip()

    # Common IEEE conference format:
    # "2021 IEEE International Symposium on Circuits and Systems (ISCAS) | ..."
    m = re.search(
        r"\b(20\d{2}\s+IEEE\s+[^|]{8,180}?\([A-Za-z0-9\-]{2,20}\))\b",
        flat,
        re.IGNORECASE,
    )
    if m:
        return _clean_text(m.group(1)).strip(" -|")

    # Fallback: IEEE ... Conference/Symposium/Workshop ... (ACRONYM)
    m = re.search(
        r"\b(IEEE\s+[^|]{8,200}?(?:Conference|Symposium|Workshop)[^|]{0,80}(?:\([A-Za-z0-9\-]{2,20}\))?)\b",
        flat,
        re.IGNORECASE,
    )
    if m:
        return _clean_text(m.group(1)).strip(" -|")

    # IEEE journal header lines, often uppercase in first-page metadata.
    m = re.search(
        r"\b(IEEE\s+TRANSACTIONS\s+ON\s+[A-Z0-9\s:&,\-]{8,160})\b",
        flat,
        re.IGNORECASE,
    )
    if m:
        pub = _clean_text(m.group(1)).strip(" -|")
        # Keep canonical all-caps style if source is all-caps.
        if m.group(1).upper() == m.group(1):
            pub = pub.upper()
        return pub

    # Generic fallback for common journal names in uppercase headers.
    m = re.search(
        r"\b([A-Z][A-Z0-9&,\-:\s]{12,180}(?:JOURNAL|TRANSACTIONS|LETTERS|BRIEFS)[A-Z0-9&,\-:\s]{0,80})\b",
        flat,
    )
    if m:
        return _clean_text(m.group(1)).strip(" -|")

    return None


def _extract_doi_from_html_or_text(html: str, text: str) -> Optional[str]:
    """Extract DOI from common meta tags, then visible text."""
    meta_patterns = [
        r'(?is)<meta[^>]+(?:name|property)=["\']citation_doi["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+(?:name|property)=["\']dc\.identifier["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for p in meta_patterns:
        m = re.search(p, html)
        if not m:
            continue
        val = (m.group(1) or "").strip()
        parsed = _extract_doi(val)
        if parsed:
            return parsed
        if val.startswith("10."):
            return val
    return _extract_doi(text)


def _extract_publication_name_from_html_or_text(html: str, text: str) -> Optional[str]:
    """Extract publication name from webpage meta tags or visible text."""
    meta_patterns = [
        r'(?is)<meta[^>]+(?:name|property)=["\']citation_journal_title["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+(?:name|property)=["\']citation_conference_title["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+(?:name|property)=["\']dc\.source["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+(?:name|property)=["\']prism\.publicationname["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for p in meta_patterns:
        m = re.search(p, html)
        if m:
            val = _clean_text((m.group(1) or "").strip()).strip(" -|")
            if val and len(val) >= 6:
                return val

    # Fallback to existing text-based extractor on rendered/plain text.
    return _extract_publication_name_from_text(text)


def _detect_nanohub_in_text(text: str) -> Dict[str, Any]:
    """
    Detect nanoHUB/NCN evidence from full text and return matching snippets.
    """
    if not text:
        return {"found": False, "matches": []}
    low = text.lower()
    patterns = [
        r"\bnanohub\b",
        r"\bnano[-\s]?hub\b",
        r"\bnanohub\.org\b",
        r"\bncn\b",
        r"\bnetwork for computational nanotechnology\b",
    ]
    matches: List[str] = []
    for p in patterns:
        m = re.search(p, low)
        if m:
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 80)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            matches.append(snippet[:220])
    # de-dup
    uniq = []
    seen = set()
    for s in matches:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    return {"found": len(uniq) > 0, "matches": uniq[:5]}


def _detect_ncn_strict_in_text(text: str) -> Dict[str, Any]:
    """
    Strict NCN association detector.
    Does NOT treat plain 'nanohub' mentions alone as NCN affiliation.
    """
    if not text:
        return {"found": False, "matches": []}
    low = text.lower()
    matches: List[str] = []
    for p in _NCN_STRICT_PATTERNS:
        m = re.search(p, low)
        if m:
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 80)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            matches.append(snippet[:220])
    # de-dup
    uniq = []
    seen = set()
    for s in matches:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    return {"found": len(uniq) > 0, "matches": uniq[:5]}


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


def _clean_title_text(title: str) -> str:
    """
    Clean malformed title text while preserving valid leading content.
    Designed for mojibake-heavy tails like '...TeachingÃ?Â??...'.
    """
    if not title:
        return ""

    t = _fix_mojibake(title)
    t = t.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    t = re.sub(r"[ \t]{2,}", " ", t).strip()

    # If there is a clear mojibake tail, keep the valid prefix.
    m = _MOJIBAKE_MARKERS_RE.search(t)
    if m and m.start() >= 12:
        t = t[:m.start()].rstrip(" ,;:-")

    # Remove long runs of punctuation noise that often appear after decode failures.
    t = re.sub(r"[?�]{3,}", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip(" ,;:-")
    return _ascii(t)


class ClassificationAgent(BaseCitationAgent):
    """Verify and complete citation metadata at status 3."""

    STAGE_NAME = "classification"
    TARGET_STATUS = 3
    NEXT_STATUS = 4

    _NON_BLOCKING_AUTHOR_MISSING_PREFIXES = (
        "authors.organization",
        "authors.department",
        "authors.email",
        "authors.orcid",
        "authors.country",
    )

    def _prune_invalid_existing_authors(self, citation_id: int) -> Dict[str, Any]:
        """
        Remove already-linked authors that are missing first/last names.
        """
        try:
            citation = self.cit_client.get(citation_id)
            removed_ids: List[int] = []
            for a in (citation.authors or []):
                fn = (a.get("firstname") or a.get("firstName") or "").strip()
                ln = (a.get("lastname") or a.get("lastName") or "").strip()
                if fn and ln:
                    continue
                person_id = a.get("id") or a.get("personId") or a.get("personid")
                if not person_id:
                    continue
                try:
                    pid = int(str(person_id).strip())
                except Exception:
                    continue
                if pid <= 0:
                    continue
                try:
                    self.cit_client._api_call(
                        "PersonDocument",
                        {"action": "remove", "idDocument": citation_id, "idPerson": pid},
                    )
                    removed_ids.append(pid)
                except Exception:
                    continue

            return {"ok": True, "removed_ids": sorted(set(removed_ids))}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _validate_title(self, citation_id: int) -> Dict[str, Any]:
        """
        Auto-clean obvious mojibake/noise in title before LLM classification.
        """
        try:
            citation = self.cit_client.get(citation_id)
            original = (citation.title or "").strip()
            if not original:
                return {"ok": True, "changed": False, "reason": "empty_title"}

            cleaned = _clean_title_text(original)
            if not cleaned:
                return {"ok": True, "changed": False, "reason": "cleaner_produced_empty"}

            # Keep original if cleaning would be too destructive.
            if len(cleaned) < 12 and len(original) >= 20:
                return {"ok": True, "changed": False, "reason": "cleaned_too_short"}

            if cleaned != original:
                citation.title = cleaned
                existing = citation.notes or ""
                sep = "\n" if existing else ""
                citation.notes = (
                    f"{existing}{sep}[Agent/{self.STAGE_NAME}] "
                    "Title auto-cleaned for encoding noise."
                )
                self.cit_client.update(citation)
                return {"ok": True, "changed": True, "old": original[:160], "new": cleaned[:160]}

            return {"ok": True, "changed": False, "reason": "already_clean"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @property
    def _system_prompt(self) -> str:
        return (
            "You are a citation pipeline agent responsible for Step 3: Classification.\n\n"
            "Your job is to make sure a citation has COMPLETE metadata before it moves to review.\n\n"
            "=== REQUIRED FIELDS ===\n"
            "title, authors (at least one valid author), year, abstract, document_genre_name,\n"
            "publication_name, ref_type, affiliated\n\n"
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
            "thesis publication rule:\n"
            "  If document_genre_name indicates Thesis/Dissertation,\n"
            "  publication_name must be identical to the citation title.\n"
            "  Save with update_citation_fields({'publication_name': <title>}).\n\n"
            "affiliated  (integer):\n"
            "  STRICT NCN rule:\n"
            "  Set affiliated=1 ONLY when the citation has explicit NCN evidence in text,\n"
            "  such as 'Network for Computational Nanotechnology' or clear 'NCN' references.\n"
            "  A plain 'nanoHUB' mention by itself is NOT sufficient for affiliated=1.\n"
            "  Otherwise set affiliated=0.\n"
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
            "  IMPORTANT: Missing author detail fields (organization, department, email, ORCID, country)\n"
            "  are non-blocking if they cannot be determined confidently.\n"
            "  In that case, add a note and still advance to status 4.\n\n"
            "date_publish:\n"
            "  If date_publish is empty, try to obtain publication date from web metadata\n"
            "  by calling extract_web_context. Save date_publish via update_citation_fields.\n\n"
            "nanoHUB resource associations:\n"
            "  While reading the PDF, scan for any nanoHUB URLs in the text,\n"
            "  references, or acknowledgements. Patterns to look for:\n"
            "    https://nanohub.org/resources/<id>\n"
            "    https://nanohub.org/tools/<id>\n"
            "    https://nanohub.org/publications/<id>/...\n"
            "    nanohub.org/<any-page>\n"
            "  Association mapping:\n"
            "    resources/tools URL -> resource:<id>\n"
            "    publications URL    -> publication:<id>\n"
            "    generic nanohub page -> link:1\n"
            "  For each unique nanoHUB URL found, call add_nanohub_resource.\n"
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
                    "Register a nanoHUB association found in the PDF as an association on this citation. "
                    "Mapping rules: resources/tools URL -> resource:<id>, "
                    "publications URL -> publication:<id>, generic nanohub page -> link:1. "
                    "Pass full URL (preferred) or numeric resource ID."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "integer"},
                        "resource_url": {
                            "type": "string",
                            "description": "Full nanoHUB URL or numeric resource ID.",
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

                # Auto-register nanoHUB associations found in the PDF
                nh_urls = _extract_nanohub_urls(text)
                registered = []
                for url in nh_urls:
                    assoc = _resolve_nanohub_association(url)
                    if assoc is None:
                        print(f"    [DocumentAssociation] could not resolve association: {url}")
                        continue
                    try:
                        result = self.cit_client._api_call(
                            "DocumentAssociation",
                            {
                                "action": "add",
                                "idDocument": cid,
                                "assocName": assoc["assoc_name"],
                                "assocID": assoc["assoc_id"],
                            },
                        )
                        print(
                            "    [DocumentAssociation] "
                            f"{assoc['assoc_name']}={assoc['assoc_id']} source={assoc['source']} result={result}"
                        )
                        registered.append(assoc)
                    except Exception as exc:
                        if "already exists" in str(exc).lower():
                            registered.append(assoc)
                        else:
                            print(
                                "    [DocumentAssociation] "
                                f"{assoc['assoc_name']}={assoc['assoc_id']} source={assoc['source']} error={exc}"
                            )

                # Auto-save Received / Revised / Accepted dates from raw text
                dates = _extract_submission_dates(text)
                dates_saved = {}
                bib_saved = {}
                nanohub_evidence = _detect_nanohub_in_text(text)
                ncn_strict_evidence = _detect_ncn_strict_in_text(text)
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

                    # Deterministic DOI + publication extraction from PDF text
                    if not citation.doi:
                        extracted_doi = _extract_doi(text)
                        if extracted_doi:
                            citation.doi = extracted_doi
                            bib_saved["doi"] = extracted_doi
                            changed = True

                    if not citation.publication_name:
                        extracted_pub = _extract_publication_name_from_text(text)
                        if extracted_pub:
                            citation.publication_name = extracted_pub
                            bib_saved["publication_name"] = extracted_pub
                            changed = True
                    if (
                        not citation.publication_name
                        and _is_thesis_genre(citation.document_genre_name)
                        and citation.title
                    ):
                        citation.publication_name = _clean_text(citation.title)
                        bib_saved["publication_name"] = citation.publication_name
                        changed = True

                    # Strict NCN affiliation: require explicit NCN evidence.
                    if ncn_strict_evidence.get("found") and citation.affiliated != 1:
                        citation.affiliated = 1
                        changed = True
                        existing = citation.notes or ""
                        sep = "\n" if existing else ""
                        evidence_line = " | ".join(ncn_strict_evidence.get("matches", [])[:2])
                        citation.notes = (
                            f"{existing}{sep}[Agent/{self.STAGE_NAME}] "
                            f"[NCN_STRICT_EVIDENCE] Full PDF text contains explicit NCN evidence. "
                            f"Set affiliated=1. Evidence: {evidence_line}"
                        )

                    if changed:
                        self.cit_client.update(citation)
                        print(f"    [dates] saved {dates_saved}")
                        if bib_saved:
                            print(f"    [biblio] saved {bib_saved}")
                except Exception as exc:
                    print(f"    [dates] error saving: {exc}")

                return {
                    "ok": True,
                    "text": clean,
                    "total_pages": len(reader.pages),
                    "nanohub_associations_registered": registered,
                    "nanohub_evidence_found": bool(nanohub_evidence.get("found")),
                    "nanohub_evidence_matches": nanohub_evidence.get("matches", []),
                    "ncn_strict_evidence_found": bool(ncn_strict_evidence.get("found")),
                    "ncn_strict_evidence_matches": ncn_strict_evidence.get("matches", []),
                    "dates_saved": dates_saved,
                    "bibliographic_saved": bib_saved,
                    "note": "nanoHUB associations and dates have already been registered — do not call add_nanohub_resource again for these URLs.",
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
                author_aff_pairs: List[Dict[str, str]] = []
                detected_date = None
                detected_doi = None
                detected_publication_name = None

                # First pass: DOI directly from URL patterns (e.g., /doi/full/<doi>)
                for u in urls:
                    url_doi = _extract_doi_from_url(u)
                    if url_doi:
                        detected_doi = url_doi
                        break

                scholar_captcha_warning: Optional[str] = None
                scholar_url_replaced: Optional[str] = None
                changed = False

                with _requests.Session() as s:
                    for url in urls:
                        try:
                            # --- Google Scholar handling ---
                            if _is_google_scholar_url(url):
                                # Fast path: real URL may be embedded in Scholar's q= param
                                real_url = _extract_real_url_from_scholar_url(url)

                                if not real_url:
                                    # Slow path: fetch Scholar page and parse HTML links
                                    resp = s.get(
                                        url,
                                        timeout=12,
                                        allow_redirects=True,
                                        headers={
                                            "User-Agent": (
                                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                "Chrome/124.0.0.0 Safari/537.36"
                                            )
                                        },
                                    )
                                    raw_html = resp.text[:400000]
                                    if _is_scholar_captcha_page(raw_html, resp.status_code):
                                        scholar_captcha_warning = (
                                            f"Google Scholar returned a CAPTCHA/bot-challenge for {url}. "
                                            "Please open that URL in a browser, solve the CAPTCHA, "
                                            "then re-run this agent."
                                        )
                                        print(f"    [Scholar] CAPTCHA detected for {url}")
                                        continue
                                    real_url = _extract_real_url_from_scholar_html(raw_html)

                                if real_url:
                                    print(f"    [Scholar] Replacing Scholar URL with: {real_url}")
                                    citation.url = real_url
                                    scholar_url_replaced = real_url
                                    changed = True
                                    # Add real URL to the fetch queue so metadata is extracted
                                    if real_url not in urls:
                                        urls.insert(urls.index(url) + 1, real_url)
                                continue  # Scholar page itself is not useful for metadata

                            # --- Regular URL handling ---
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

                            found_doi = _extract_doi_from_html_or_text(html, text)
                            if found_doi and not detected_doi:
                                detected_doi = found_doi

                            found_pub = _extract_publication_name_from_html_or_text(html, text)
                            if found_pub and not detected_publication_name:
                                detected_publication_name = found_pub

                            all_affiliations.extend(_extract_affiliation_candidates(text))
                            author_aff_pairs.extend(_extract_author_affiliations_from_text(text))
                        except Exception:
                            continue

                date_saved = None
                doi_saved = None
                publication_saved = None
                authors_updated = 0
                if detected_date and not citation.date_publish:
                    citation.date_publish = detected_date
                    date_saved = detected_date
                    changed = True
                if detected_doi and not citation.doi:
                    citation.doi = detected_doi
                    doi_saved = detected_doi
                    changed = True
                if detected_publication_name and not citation.publication_name:
                    citation.publication_name = detected_publication_name
                    publication_saved = detected_publication_name
                    changed = True

                # Deterministic author affiliation enrichment from web text.
                authors_updated = _apply_web_author_affiliations(citation, author_aff_pairs)
                if authors_updated > 0:
                    changed = True
                if changed:
                    self.cit_client.update(citation)

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
                result: Dict[str, Any] = {
                    "ok": True,
                    "urls_checked": urls,
                    "pages_loaded": len(pages),
                    "text": combined_text[:15000],
                    "candidate_affiliations": dedup_affs[:25],
                    "date_publish_detected": detected_date,
                    "date_publish_saved": date_saved,
                    "doi_detected": detected_doi,
                    "doi_saved": doi_saved,
                    "publication_name_detected": detected_publication_name,
                    "publication_name_saved": publication_saved,
                    "author_affiliation_pairs_detected": len(author_aff_pairs),
                    "authors_updated_from_web": authors_updated,
                }
                if scholar_url_replaced:
                    result["scholar_url_replaced"] = scholar_url_replaced
                if scholar_captcha_warning:
                    result["warning"] = scholar_captcha_warning
                return result
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
                    elif key == "affiliated":
                        # Guardrail: do not downgrade affiliated to 0 if strict NCN evidence was recorded.
                        if int(value) == 0 and "[NCN_STRICT_EVIDENCE]" in (citation.notes or ""):
                            updated.append("affiliated_guarded")
                            continue
                        citation.affiliated = int(value)
                        updated.append("affiliated")
                    elif hasattr(citation, key):
                        setattr(citation, key, _clean_text(value) if isinstance(value, str) else value)
                        updated.append(key)

                # Thesis/Dissertation rule: publication_name must match title.
                effective_genre = str(fields.get("document_genre_name", citation.document_genre_name) or "")
                effective_title = _clean_text(str(fields.get("title", citation.title) or ""))
                effective_pub = _clean_text(str(fields.get("publication_name", citation.publication_name) or ""))
                if _is_thesis_genre(effective_genre) and effective_title and (not effective_pub or effective_pub != effective_title):
                    citation.publication_name = effective_title
                    updated.append("publication_name(thesis=title)")

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
                def _build_clean_authors(include_person_ids: bool) -> List[Dict]:
                    clean: List[Dict] = []
                    seen: set = set()
                    for a in raw_authors:
                        fn = _ascii(a.get("firstname", ""))
                        ln = _ascii(a.get("lastname", ""))
                        if not fn or not ln:
                            # Backend enforces both first/last names for Person records.
                            continue
                        sig = f"{ln.lower()}_{fn[:1].lower()}"
                        if sig in seen:
                            continue
                        seen.add(sig)

                        # Start from deduplicated existing author record.
                        base = dict(existing_by_sig.get(sig, {}))
                        if not include_person_ids:
                            # Preferred path: omit person IDs so backend writes
                            # person-document detail fields from this payload.
                            for id_key in ("id", "personId", "personid"):
                                base.pop(id_key, None)
                        else:
                            # Fallback path: preserve IDs to avoid duplicate-person
                            # insertion conflicts on backend unique keys.
                            for id_key in ("id", "personId", "personid"):
                                raw_id = a.get(id_key)
                                if raw_id and id_key not in base:
                                    base[id_key] = raw_id

                        base["firstname"] = fn
                        base["lastname"] = ln

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
                    return clean

                # First pass: no person IDs (best chance to persist org fields).
                clean = _build_clean_authors(include_person_ids=False)
                if not clean:
                    return {"ok": False, "error": "No valid authors after validation (firstname/lastname required)."}
                citation.authors = clean
                try:
                    self.cit_client.update(citation)
                    return {"ok": True, "authors_updated": len(clean)}
                except Exception as exc:
                    msg = str(exc)
                    duplicate_person = ("Duplicate entry" in msg and "person_unique" in msg)
                    if not duplicate_person:
                        raise

                    # Fallback: retry preserving person IDs to avoid duplicate inserts.
                    citation_retry = self.cit_client.get(cid)
                    clean_retry = _build_clean_authors(include_person_ids=True)
                    citation_retry.authors = clean_retry
                    self.cit_client.update(citation_retry)
                    return {
                        "ok": True,
                        "authors_updated": len(clean_retry),
                        "retry_mode": "with_person_ids_after_duplicate_person_unique",
                    }
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
                assoc = _resolve_nanohub_association(resource_url)
                if assoc is None:
                    return {"ok": False, "error": f"Cannot resolve nanoHUB association from: {resource_url}"}

                result = self.cit_client._api_call(
                    "DocumentAssociation",
                    {
                        "action": "add",
                        "idDocument": cid,
                        "assocName": assoc["assoc_name"],
                        "assocID": assoc["assoc_id"],
                    },
                )
                # _api_call raises on error status, so reaching here means success
                print(
                    "    [DocumentAssociation] "
                    f"{assoc['assoc_name']}={assoc['assoc_id']} source={assoc['source']} result={result}"
                )
                return {"ok": True, "association": assoc, "result": result}
            except Exception as exc:
                if "already exists" in str(exc).lower():
                    return {"ok": True, "association": assoc, "message": "association already registered"}
                return {"ok": False, "error": str(exc)}

        if name == "flag_incomplete":
            cid = input_data["citation_id"]
            missing = input_data.get("missing_fields", [])
            note = input_data.get("note", "")
            try:
                missing_list = [str(m).strip().lower() for m in missing if str(m).strip()]
                has_hard_blocker = False
                for field in missing_list:
                    if field == "authors":
                        has_hard_blocker = True
                        break
                    if any(field.startswith(p) for p in self._NON_BLOCKING_AUTHOR_MISSING_PREFIXES):
                        continue
                    has_hard_blocker = True
                    break

                if has_hard_blocker:
                    return self._append_note(cid, f"[INCOMPLETE] Missing: {missing}. {note}".strip())

                # Only non-blocking author detail gaps: record warning, then advance.
                warning = (
                    f"[WARNING] Non-blocking author details missing: {missing}. {note}. "
                    "Advancing to status 4."
                ).strip()
                append_result = self._append_note(cid, warning)
                advance_result = self._advance_status(
                    cid,
                    self.NEXT_STATUS,
                    "Advanced with non-blocking author-detail gaps.",
                )
                return {
                    "ok": bool(append_result.get("ok")) and bool(advance_result.get("ok")),
                    "non_blocking": True,
                    "append_result": append_result,
                    "advance_result": advance_result,
                }
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

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

    # ------------------------------------------------------------------
    # Override run() — enforce title validation before LLM loop
    # ------------------------------------------------------------------

    def run(self, citation_id: int):
        from .base_agent import AgentResult
        citation = self.cit_client.get(citation_id)
        status_before = citation.status

        print(f"\n{'─'*60}")
        print(f"  [{self.STAGE_NAME.upper()}] citation {citation_id}  (current status={status_before})")
        print(f"{'─'*60}")

        title_result = self._validate_title(citation_id)
        if title_result.get("ok") and title_result.get("changed"):
            print("  [classification] title validated and auto-cleaned.")
        elif not title_result.get("ok"):
            print(f"  [classification] title validation warning: {title_result.get('error')}")

        # Reload after potential title update
        citation = self.cit_client.get(citation_id)
        user_message = self._build_prompt(citation)
        result_text = self._run_agentic_loop(user_message)

        updated = self.cit_client.get(citation_id)
        status_after = updated.status
        success = (status_after == self.NEXT_STATUS)

        return AgentResult(
            success=success,
            citation_id=citation_id,
            status_before=status_before,
            status_after=status_after,
            message=result_text[:300] if result_text else "(no output)",
        )

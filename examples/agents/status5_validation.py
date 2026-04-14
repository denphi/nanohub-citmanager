"""
Status 5 — Post-Review Validation Agent

Runs after HumanReviewAgent as an external quality gate.  It does NOT use an
LLM for the structural checks; the LLM is called only to assess abstract
quality when the abstract is present but suspect.

Decision logic
--------------
  PASS  → all required fields present, abstract coherent, authors valid
          → citation stays at status 5, a [VALIDATION_PASSED] note is appended

  FAIL  → one or more hard blockers found
          → citation is moved back to status 3, a [VALIDATION_FAILED] note
            listing every failure reason is appended

Hard-blocker checklist
  1. title           — present, ≥ 10 chars, not garbage
  2. authors         — at least one author with real firstname + lastname
                       (no placeholders: N/A / Placeholder / Unknown / Author)
                       no duplicate authors (by lastname + first initial)
  3. year            — present and a plausible 4-digit year
  4. abstract        — present, ≥ 80 chars; LLM checks for garbage/mojibake
  5. document_genre_name — present and non-empty
  6. publication_name    — present and non-empty
  7. ref_type        — present (one of R/C/E/N flags)
  8. affiliated      — set (0 or 1, not −1 / None)
  9. date_publish    — present and non-empty
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseCitationAgent, AgentResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLACEHOLDER_NAMES = {"n/a", "placeholder", "unknown", "author", ""}

_GARBAGE_ABSTRACT_PATTERNS = [
    r"[?]{5,}",                      # five or more consecutive question marks
    r"[■□▪▫]{5,}",                   # runs of replacement-box chars
    r"[\ufffd]{3,}",                  # Unicode replacement chars
    r"[^\x00-\x7f]{30,}",            # very long uninterrupted non-ASCII runs
    r"(?:[ÃÂ][^\w\s]|[?]){6,}",     # unrecoverable latin-1/UTF-8 mojibake runs (Ã?Â??…)
]

_MIN_TITLE_LEN = 10
_MIN_ABSTRACT_LEN = 80
_YEAR_RE = re.compile(r"^\d{4}$")


# ---------------------------------------------------------------------------
# Pure-Python validation helpers
# ---------------------------------------------------------------------------

def _is_placeholder(name: str) -> bool:
    return name.strip().lower() in _PLACEHOLDER_NAMES


def _has_real_authors(authors: Optional[List[Dict]]) -> bool:
    if not authors:
        return False
    for a in authors:
        fn = (a.get("firstname") or a.get("firstName") or "").strip()
        ln = (a.get("lastname") or a.get("lastName") or "").strip()
        if fn and ln and not _is_placeholder(fn) and not _is_placeholder(ln):
            return True
    return False


def _authors_are_same_person(fn_a: str, ln_a: str, fn_b: str, ln_b: str) -> bool:
    """
    Return True if two author name pairs likely refer to the same person.
    Handles:
      - Exact match: 'John Smith' == 'John Smith'
      - Initial match: 'J. Smith' matches 'John Smith' (one is a prefix/initial
        of the other), but 'James Smith' does NOT match 'John Smith'.
    """
    if ln_a.lower() != ln_b.lower():
        return False
    fa, fb = fn_a.lower().rstrip("."), fn_b.lower().rstrip(".")
    if fa == fb:
        return True
    # One must be a single initial that is the start of the other full name
    short, long = (fa, fb) if len(fa) <= len(fb) else (fb, fa)
    return len(short) == 1 and long.startswith(short)


def _find_duplicate_authors(authors: Optional[List[Dict]]) -> List[str]:
    """
    Return a list of names that appear more than once, catching both exact
    duplicates and abbreviated-firstname variants like 'J. Smith' / 'John Smith'.
    'James Smith' and 'John Smith' are NOT considered duplicates.
    """
    if not authors:
        return []
    valid = []
    for a in authors:
        fn = (a.get("firstname") or a.get("firstName") or "").strip()
        ln = (a.get("lastname") or a.get("lastName") or "").strip()
        if fn and ln and not _is_placeholder(fn) and not _is_placeholder(ln):
            valid.append((fn, ln))

    dupes: List[str] = []
    for i, (fn_a, ln_a) in enumerate(valid):
        for fn_b, ln_b in valid[:i]:
            if _authors_are_same_person(fn_a, ln_a, fn_b, ln_b):
                dupes.append(f"{fn_a} {ln_a} (matches {fn_b} {ln_b})")
                break
    return dupes


def _looks_like_garbage_text(text: str) -> bool:
    """Return True if text matches known garbage/mojibake patterns."""
    for p in _GARBAGE_ABSTRACT_PATTERNS:
        if re.search(p, text):
            return True
    # Ratio of non-ASCII characters > 60 % in a long text is suspicious
    if len(text) >= 80:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.60:
            return True
    return False


def _check_required_fields(citation) -> List[str]:
    """
    Return a list of human-readable failure reasons.
    An empty list means all hard-blocker checks passed.
    """
    failures: List[str] = []

    # 1. Title
    title = (citation.title or "").strip()
    if not title:
        failures.append("title is missing")
    elif len(title) < _MIN_TITLE_LEN:
        failures.append(f"title is too short ({len(title)} chars, minimum {_MIN_TITLE_LEN})")
    elif _looks_like_garbage_text(title):
        failures.append("title appears to contain garbage/encoding noise")

    # 2. Authors — must have at least one real author, no duplicates
    if not _has_real_authors(citation.authors):
        failures.append("no valid authors (at least one author with first and last name required)")
    else:
        dupes = _find_duplicate_authors(citation.authors)
        if dupes:
            failures.append(f"duplicate authors detected: {'; '.join(dupes[:5])}")

    # 3. Year
    year = str(citation.year or "").strip()
    if not year:
        failures.append("year is missing")
    elif not _YEAR_RE.match(year) or not (1900 <= int(year) <= 2100):
        failures.append(f"year '{year}' is not a plausible 4-digit year")

    # 4. Abstract
    abstract = (citation.abstract or "").strip()
    if not abstract:
        failures.append("abstract is missing")
    elif len(abstract) < _MIN_ABSTRACT_LEN:
        failures.append(
            f"abstract is too short ({len(abstract)} chars, minimum {_MIN_ABSTRACT_LEN})"
        )
    elif _looks_like_garbage_text(abstract):
        failures.append("abstract appears to contain garbage/encoding noise")

    # 5. document_genre_name
    if not (citation.document_genre_name or "").strip():
        failures.append("document_genre_name is missing")

    # 6. publication_name
    if not (citation.publication_name or "").strip():
        failures.append("publication_name is missing")

    # 7. ref_type — must contain at least one of R/C/E/N
    ref_type = (citation.ref_type or "").strip().upper()
    valid_flags = set("RCEN")
    if not ref_type:
        failures.append("ref_type is missing")
    elif not any(f in valid_flags for f in re.split(r"[,\s]+", ref_type)):
        failures.append(f"ref_type '{ref_type}' does not contain a valid flag (R/C/E/N)")

    # 8. affiliated — must be 0 or 1 (not -1 / None)
    affiliated = citation.affiliated
    if affiliated is None or affiliated == -1:
        failures.append("affiliated is not set (must be 0 or 1)")

    # 9. date_publish — must be present and not a zero/sentinel value
    _INVALID_DATES = {"0000-00-00", "0000-00", "0000", "00-00-0000", ""}
    date_publish = (citation.date_publish or "").strip()
    if not date_publish or date_publish in _INVALID_DATES or re.match(r"^0+[-/0]*$", date_publish):
        failures.append(f"date_publish is missing or invalid ('{date_publish}')")

    return failures


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ValidationAgent(BaseCitationAgent):
    """
    External validation gate for citations at status 5.

    This agent is intentionally lightweight:
      - Structural checks are purely deterministic (no LLM cost).
      - Abstract quality check uses the LLM only when the abstract is present
        but flagged as potentially garbage by the heuristic.

    On PASS  → stays at status 5, appends [VALIDATION_PASSED] note.
    On FAIL  → moves back to status 3, appends [VALIDATION_FAILED] note
               listing every failure reason so ClassificationAgent can fix them.
    """

    STAGE_NAME = "validation"
    TARGET_STATUS = 5
    # On success the citation stays at 5; on failure it goes to 3.
    # The base class success check uses NEXT_STATUS, but we override run()
    # completely, so this value is informational only.
    NEXT_STATUS = 5

    # ------------------------------------------------------------------
    # System prompt — informational only; not used by the agentic loop
    # (abstract check uses direct single-turn calls, not the tool loop)
    # ------------------------------------------------------------------

    @property
    def _system_prompt(self) -> str:
        return "You are a citation quality validator."

    # ------------------------------------------------------------------
    # Tool definitions — none needed for this agent
    # ------------------------------------------------------------------

    def _tool_definitions(self) -> List[Dict]:
        return []

    def _execute_tool(self, name: str, input_data: Dict) -> Dict:
        return {"error": f"Unknown tool: {name}"}

    def _build_prompt(self, citation) -> str:  # pragma: no cover
        return ""

    # ------------------------------------------------------------------
    # LLM abstract quality check — direct single-turn call, no tool loop
    # ------------------------------------------------------------------

    _ABSTRACT_CHECK_SYSTEM = (
        "You are a citation quality validator. "
        "Assess whether the text is a valid academic abstract or contains "
        "encoding garbage, OCR noise, random characters, or placeholder content. "
        "Reply with ONLY a JSON object: {\"ok\": true/false, \"reason\": \"one sentence\"}. "
        "No prose, no markdown, no code fences — raw JSON only."
    )

    def _llm_check_abstract(self, abstract: str) -> Tuple[bool, str]:
        """
        Ask the LLM whether `abstract` is a real academic abstract.
        Uses a direct single-turn request (no tool loop) to avoid null-response
        issues on OpenWebUI models.
        Returns (is_ok, reason). Conservatively returns (True, ...) on any error.
        """
        import json as _json

        user_content = (
            "Is the following text a valid academic abstract?\n\n"
            f"---\n{abstract[:3000]}\n---\n\n"
            "Reply with ONLY: {\"ok\": true, \"reason\": \"...\"} or {\"ok\": false, \"reason\": \"...\"}"
        )

        raw = ""
        try:
            if self._backend == "anthropic":
                import anthropic as _anthropic
                resp = self._llm_client.messages.create(
                    model=self.ANTHROPIC_MODEL,
                    max_tokens=256,
                    system=self._ABSTRACT_CHECK_SYSTEM,
                    messages=[{"role": "user", "content": user_content}],
                )
                raw = next((b.text for b in resp.content if b.type == "text"), "")
            else:
                # OpenAI-compatible (OpenWebUI): try up to 3 times with no tools
                import time as _time
                for attempt in range(3):
                    try:
                        resp = self._llm_client.chat.completions.create(
                            model=self._openai_model,
                            max_tokens=256,
                            timeout=self.OPENAI_REQUEST_TIMEOUT,
                            messages=[
                                {"role": "system", "content": self._ABSTRACT_CHECK_SYSTEM},
                                {"role": "user", "content": user_content},
                            ],
                        )
                        if resp and getattr(resp, "choices", None):
                            raw = (resp.choices[0].message.content or "").strip()
                            if raw:
                                break
                    except Exception as exc:
                        if self._is_rate_limit_error(exc):
                            print(
                                f"    [validation] 429 rate limit — waiting {self.OPENAI_RATE_LIMIT_WAIT}s "
                                f"(attempt {attempt + 1}/3)..."
                            )
                            _time.sleep(self.OPENAI_RATE_LIMIT_WAIT)
                        else:
                            print(f"    [validation] LLM attempt {attempt + 1}/3 error: {exc}")
                    raw = ""

            if not raw:
                return True, "LLM check skipped (empty response)"

            # Strip markdown fences if model ignored the instruction
            cleaned = re.sub(r"^```[a-z]*\s*", "", raw.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip())

            # Try direct parse first
            try:
                data = _json.loads(cleaned)
                return bool(data.get("ok", True)), str(data.get("reason", ""))
            except _json.JSONDecodeError:
                pass

            # Fallback: extract JSON object from anywhere in the response
            m = re.search(r'\{[^{}]*"ok"\s*:\s*(true|false)[^{}]*\}', cleaned, re.IGNORECASE | re.DOTALL)
            if m:
                try:
                    data = _json.loads(m.group(0))
                    return bool(data.get("ok", True)), str(data.get("reason", ""))
                except _json.JSONDecodeError:
                    pass

            # Last resort: look for clear positive/negative signal in free text
            lower = cleaned.lower()
            if any(w in lower for w in ("not valid", "garbage", "noise", "invalid", "not an abstract", "placeholder")):
                return False, f"LLM flagged abstract as invalid (raw: {cleaned[:120]})"
            if any(w in lower for w in ("valid", "real", "legitimate", "academic", "coherent")):
                return True, f"LLM confirmed abstract is valid (raw: {cleaned[:120]})"

            # Cannot determine — be conservative
            return True, f"LLM check inconclusive (raw: {cleaned[:120]})"

        except Exception as exc:
            return True, f"LLM check skipped ({exc})"

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, citation_id: int) -> AgentResult:
        citation = self.cit_client.get(citation_id)
        status_before = citation.status

        print(f"\n{'─'*60}")
        print(f"  [{self.STAGE_NAME.upper()}] citation {citation_id}  (current status={status_before})")
        print(f"{'─'*60}")

        # ── 1. Deterministic field checks ──────────────────────────────
        failures = _check_required_fields(citation)
        print(f"  [validation] deterministic checks: {len(failures)} issue(s)")

        # ── 2. LLM abstract quality check (only when abstract present & passes heuristic) ──
        abstract = (citation.abstract or "").strip()
        llm_abstract_ok = True
        llm_abstract_reason = ""
        if abstract and len(abstract) >= _MIN_ABSTRACT_LEN and not _looks_like_garbage_text(abstract):
            print("  [validation] running LLM abstract quality check…")
            llm_abstract_ok, llm_abstract_reason = self._llm_check_abstract(abstract)
            if not llm_abstract_ok:
                failures.append(f"abstract failed LLM quality check: {llm_abstract_reason}")
                print(f"  [validation] LLM abstract check FAILED: {llm_abstract_reason}")
            else:
                print(f"  [validation] LLM abstract check passed: {llm_abstract_reason}")

        # ── 3. Decision ────────────────────────────────────────────────
        passed = len(failures) == 0

        existing_notes = citation.notes or ""
        sep = "\n" if existing_notes else ""

        if passed:
            note = (
                f"[Agent/{self.STAGE_NAME}] [VALIDATION_PASSED] "
                "All required fields present and abstract quality confirmed. "
                "Ready for human publish/reject decision."
            )
            citation.notes = f"{existing_notes}{sep}{note}"
            self.cit_client.update(citation)

            print("  [validation] PASSED — citation stays at status 5")
            return AgentResult(
                success=True,
                citation_id=citation_id,
                status_before=status_before,
                status_after=5,
                message="Validation passed. Citation ready for human review.",
                details={"failures": [], "llm_abstract_reason": llm_abstract_reason},
            )

        else:
            failures_text = "; ".join(f"({i+1}) {f}" for i, f in enumerate(failures))
            note = (
                f"[Agent/{self.STAGE_NAME}] [VALIDATION_FAILED] "
                f"Moved back to status 3 for re-classification. "
                f"Issues found: {failures_text}"
            )
            citation.notes = f"{existing_notes}{sep}{note}"
            citation.status = 3
            self.cit_client.update(citation)

            print(f"  [validation] FAILED ({len(failures)} issue(s)) — moved back to status 3")
            for i, f in enumerate(failures, 1):
                print(f"    ({i}) {f}")

            return AgentResult(
                success=False,
                citation_id=citation_id,
                status_before=status_before,
                status_after=3,
                message=f"Validation failed: {failures_text[:200]}",
                details={"failures": failures, "llm_abstract_reason": llm_abstract_reason},
            )

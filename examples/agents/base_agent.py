"""
Base Citation Agent

Shared base class providing the agentic loop, tool execution, and CitationManager
helpers used by all pipeline stage agents.

Supports two LLM backends (auto-detected from environment variables):
  • Anthropic   — set ANTHROPIC_API_KEY
  • OpenWebUI   — set OPENWEBUI_KEY (and optionally OPENWEBUI_URL)
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure project root is on the path when agents are imported directly
_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_AGENTS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from nanohubcitmanager import CitationManagerClient


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Return value from every agent stage."""
    success: bool
    citation_id: int
    status_before: int
    status_after: int
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        arrow = f"status {self.status_before} → {self.status_after}"
        mark = "✓" if self.success else "✗"
        return f"[{mark}] Citation {self.citation_id} ({arrow}): {self.message}"


# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    """Return 'anthropic' or 'openai' based on available environment variables."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENWEBUI_KEY"):
        return "openai"
    raise EnvironmentError(
        "No LLM backend configured. Set either:\n"
        "  ANTHROPIC_API_KEY  — for Anthropic (claude-opus-4-6)\n"
        "  OPENWEBUI_KEY      — for OpenWebUI / Purdue GenAI"
    )


def _make_anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def _make_openai_client():
    from openai import OpenAI
    url = os.getenv("OPENWEBUI_URL", "https://genai.rcac.purdue.edu/api/chat/completions")
    # The openai SDK appends /chat/completions automatically — strip it if present
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return OpenAI(base_url=url, api_key=os.getenv("OPENWEBUI_KEY"))


# ---------------------------------------------------------------------------
# Tool-format conversion helpers
# ---------------------------------------------------------------------------

def _to_openai_tools(anthropic_tools: List[Dict]) -> List[Dict]:
    """Convert Anthropic tool definitions to OpenAI format."""
    result = []
    for t in anthropic_tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseCitationAgent:
    """
    Base class for all citation pipeline agents.

    Backend is auto-detected from environment variables at init time.

    Subclasses must implement:
        STAGE_NAME     (str)   — human-readable stage label
        TARGET_STATUS  (int)   — status this agent handles
        NEXT_STATUS    (int)   — status to advance to on success
        _tool_definitions()    → List[Dict]  (Anthropic format)
        _execute_tool(name, input_data) → Dict
        _build_prompt(citation) → str
        _system_prompt (property → str)
    """

    STAGE_NAME: str = "base"
    TARGET_STATUS: int = 0
    NEXT_STATUS: int = 1

    # Model IDs per backend
    ANTHROPIC_MODEL: str = "claude-opus-4-6"
    OPENAI_MODEL: str = ""  # falls back to LLM_MODEL env var

    MAX_ITERATIONS: int = 15

    def __init__(self, cit_client: CitationManagerClient):
        self.cit_client = cit_client
        self._backend = _detect_backend()
        self._llm_client = (
            _make_anthropic_client()
            if self._backend == "anthropic"
            else _make_openai_client()
        )
        self._openai_model = (
            self.OPENAI_MODEL
            or os.getenv("LLM_MODEL", "gpt-oss:120b")
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, citation_id: int) -> AgentResult:
        """Run this agent on a single citation."""
        citation = self.cit_client.get(citation_id)
        status_before = citation.status

        print(f"\n{'─'*60}")
        print(f"  [{self.STAGE_NAME.upper()}] citation {citation_id}  (current status={status_before})")
        print(f"{'─'*60}")

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

    # ------------------------------------------------------------------
    # Agentic loop dispatcher
    # ------------------------------------------------------------------

    def _run_agentic_loop(self, user_message: str) -> str:
        if self._backend == "anthropic":
            return self._run_anthropic_loop(user_message)
        return self._run_openai_loop(user_message)

    # ------------------------------------------------------------------
    # Anthropic loop
    # ------------------------------------------------------------------

    def _run_anthropic_loop(self, user_message: str) -> str:
        import anthropic as _anthropic

        messages: List[Dict] = [{"role": "user", "content": user_message}]
        tools = self._tool_definitions()

        for _ in range(self.MAX_ITERATIONS):
            with self._llm_client.messages.stream(
                model=self.ANTHROPIC_MODEL,
                max_tokens=8096,
                thinking={"type": "adaptive"},
                system=self._system_prompt,
                tools=tools,
                messages=messages,
            ) as stream:
                response = stream.get_final_message()

            if response.stop_reason == "end_turn":
                return next(
                    (b.text for b in response.content if b.type == "text"), ""
                )

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for tb in tool_blocks:
                    print(f"    → {tb.name}({json.dumps(tb.input)[:120]})")
                    try:
                        outcome = self._execute_tool(tb.name, tb.input)
                    except KeyError as exc:
                        outcome = {"error": f"Missing required parameter: {exc}"}
                    except Exception as exc:
                        outcome = {"error": str(exc)}
                    print(f"    ← {json.dumps(outcome)[:120]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tb.id,
                        "content": json.dumps(outcome),
                    })
                messages.append({"role": "user", "content": tool_results})

        return "Max iterations reached."

    # ------------------------------------------------------------------
    # OpenAI-compatible loop (OpenWebUI / Purdue GenAI)
    # ------------------------------------------------------------------

    def _run_openai_loop(self, user_message: str) -> str:
        messages: List[Dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]
        tools = _to_openai_tools(self._tool_definitions())

        for _ in range(self.MAX_ITERATIONS):
            response = None
            for attempt in range(3):
                try:
                    response = self._llm_client.chat.completions.create(
                        model=self._openai_model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                    if response is not None:
                        break
                    print(f"    [openai] null response, retrying (attempt {attempt + 1}/3)...")
                except Exception as exc:
                    print(f"    [openai] request error: {exc}, retrying (attempt {attempt + 1}/3)...")
            if response is None:
                return "Error: LLM returned no response after 3 attempts."

            choice = response.choices[0]
            msg = choice.message

            # Append assistant message (with tool_calls if present)
            assistant_entry: Dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_entry)

            if choice.finish_reason == "stop" or not msg.tool_calls:
                return msg.content or ""

            if choice.finish_reason == "tool_calls":
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        input_data = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        input_data = {}

                    print(f"    → {name}({json.dumps(input_data)[:120]})")
                    try:
                        outcome = self._execute_tool(name, input_data)
                    except KeyError as exc:
                        outcome = {"error": f"Missing required parameter: {exc}"}
                    except Exception as exc:
                        outcome = {"error": str(exc)}
                    print(f"    ← {json.dumps(outcome)[:120]}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(outcome),
                    })

        return "Max iterations reached."

    # ------------------------------------------------------------------
    # CitationManager helpers available to all subclasses
    # ------------------------------------------------------------------

    def _get_citation_dict(self, citation_id: int) -> Dict:
        c = self.cit_client.get(citation_id)
        return {
            "id": c.id,
            "title": c.title,
            "abstract": c.abstract,
            "year": c.year,
            "doi": c.doi,
            "publication_name": c.publication_name,
            "document_genre_name": c.document_genre_name,
            "volume": c.volume,
            "issue": c.issue,
            "begin_page": c.begin_page,
            "end_page": c.end_page,
            "publisher": c.publisher,
            "authors": c.authors,
            "keywords": c.keywords,
            "status": c.status,
            "notes": c.notes,
            "exp_list_exp_data": c.exp_list_exp_data,
            "exp_data": c.exp_data,
            "affiliated": c.affiliated,
            "ref_type": c.ref_type,
            "date_submit": c.date_submit,
            "date_accept": c.date_accept,
            "date_publish": c.date_publish,
            "full_text_path": c.full_text_path,
            "url": c.url,
        }

    def _advance_status(self, citation_id: int, new_status: int, note: str = "") -> Dict:
        try:
            citation = self.cit_client.get(citation_id)
            citation.status = new_status
            if note:
                existing = citation.notes or ""
                sep = "\n" if existing else ""
                citation.notes = f"{existing}{sep}[Agent/{self.STAGE_NAME}] {note}"
            self.cit_client.update(citation)
            return {"ok": True, "new_status": new_status}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _append_note(self, citation_id: int, note: str) -> Dict:
        try:
            citation = self.cit_client.get(citation_id)
            existing = citation.notes or ""
            sep = "\n" if existing else ""
            citation.notes = f"{existing}{sep}[Agent/{self.STAGE_NAME}] {note}"
            self.cit_client.update(citation)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Subclass interface (must override)
    # ------------------------------------------------------------------

    @property
    def _system_prompt(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def _tool_definitions(self) -> List[Dict]:  # pragma: no cover
        raise NotImplementedError

    def _execute_tool(self, name: str, input_data: Dict) -> Dict:  # pragma: no cover
        raise NotImplementedError

    def _build_prompt(self, citation) -> str:  # pragma: no cover
        raise NotImplementedError

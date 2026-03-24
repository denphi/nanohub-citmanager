"""
Citation Pipeline Orchestrator

Chains the five pipeline agents together and processes one citation at a time.
Determines which agent(s) to run based on the citation's current status.

Status map:
  1 → PDFVerifierAgent      → advances to 2
  2 → DisambiguationAgent   → advances to 3
  3 → ClassificationAgent   → advances to 4
  4 → ReviewAgent           → advances to 5
  5 → HumanReviewAgent      → stays at 5 (human decides publish/reject)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from nanohubcitmanager import CitationManagerClient

from .base_agent import AgentResult
from .status1_pdf_verifier import PDFVerifierAgent
from .status2_disambiguation import DisambiguationAgent
from .status3_classification import ClassificationAgent
from .status4_review import ReviewAgent
from .status5_human_review import HumanReviewAgent


# Status labels for display
STATUS_LABELS: Dict[int, str] = {
    1: "PDF Verification",
    2: "Disambiguation",
    3: "Classification",
    4: "Review (Experimentalist + Experimental Data)",
    5: "Human Review",
    100: "Published",
    -9: "Junk / Deleted",
}


@dataclass
class PipelineReport:
    """Summary of a full or partial pipeline run for one citation."""

    citation_id: int
    stages_run: List[AgentResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(r.success for r in self.stages_run)

    @property
    def final_status(self) -> int:
        if self.stages_run:
            return self.stages_run[-1].status_after
        return -1

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  PIPELINE REPORT — Citation {self.citation_id}")
        print(f"{'='*60}")
        for r in self.stages_run:
            mark = "✓" if r.success else "✗"
            label = STATUS_LABELS.get(r.status_before, f"status {r.status_before}")
            print(f"  [{mark}] {label}: {r.message[:100]}")
        final_label = STATUS_LABELS.get(self.final_status, f"status {self.final_status}")
        print(f"\n  Final status : {self.final_status} — {final_label}")
        overall = "SUCCESS" if self.success else "BLOCKED"
        print(f"  Overall      : {overall}")
        print(f"{'='*60}\n")


class CitationPipelineOrchestrator:
    """
    Orchestrates the citation processing pipeline.

    Usage examples::

        # Process a single citation from wherever it currently is:
        report = orchestrator.process_citation(citation_id=1234)

        # Process only the current stage (no auto-advance):
        report = orchestrator.process_citation(citation_id=1234, run_full_pipeline=False)

        # Advance a citation from a specific status:
        report = orchestrator.process_from_status(citation_id=1234, from_status=2)
    """

    def __init__(self, cit_client: CitationManagerClient):
        self.cit_client = cit_client
        self._agents = {
            1: PDFVerifierAgent(cit_client),
            2: DisambiguationAgent(cit_client),
            3: ClassificationAgent(cit_client),
            4: ReviewAgent(cit_client),
            5: HumanReviewAgent(cit_client),
        }
        # Show backend once after all agents are initialised
        backend = self._agents[1]._backend
        model = (
            self._agents[1].ANTHROPIC_MODEL
            if backend == "anthropic"
            else self._agents[1]._openai_model
        )
        print(f"  LLM backend : {backend}  |  model : {model}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_citation(
        self,
        citation_id: int,
        run_full_pipeline: bool = True,
        stop_on_failure: bool = True,
    ) -> PipelineReport:
        """
        Process a citation starting from its current status.

        Args:
            citation_id:        ID of the citation to process.
            run_full_pipeline:  If True, keep running subsequent stages after each
                                successful advance.  If False, run only the current stage.
            stop_on_failure:    If True (default), halt the pipeline when a stage fails
                                to advance (e.g. missing PDF, duplicate detected).
        Returns:
            PipelineReport with results from all stages that ran.
        """
        report = PipelineReport(citation_id=citation_id)

        while True:
            citation = self.cit_client.get(citation_id)
            current_status = citation.status

            agent = self._agents.get(current_status)
            if agent is None:
                label = STATUS_LABELS.get(current_status, str(current_status))
                print(f"  No agent defined for status {current_status} ({label}). Stopping.")
                break

            result = agent.run(citation_id)
            report.stages_run.append(result)

            # Status 5 never auto-advances; it's a terminal stage for agents
            if current_status == 5:
                break

            if not result.success and stop_on_failure:
                print(
                    f"  Stage blocked at status {current_status} "
                    f"({STATUS_LABELS.get(current_status, '')}). "
                    "Stopping pipeline."
                )
                break

            if not run_full_pipeline:
                break

            # Guard: if status didn't change despite no failure, stop to avoid loops
            if result.status_after == current_status:
                break

        report.print_summary()
        return report

    def process_from_status(
        self,
        citation_id: int,
        from_status: int,
        run_full_pipeline: bool = True,
        stop_on_failure: bool = True,
    ) -> PipelineReport:
        """
        Process a citation starting from a specific status, regardless of its
        current status in the database.  Useful for re-running a specific stage.
        """
        # Temporarily set the citation status (in-memory trick: we just run the agent)
        agent = self._agents.get(from_status)
        if agent is None:
            raise ValueError(f"No agent defined for status {from_status}")

        report = PipelineReport(citation_id=citation_id)
        result = agent.run(citation_id)
        report.stages_run.append(result)

        if run_full_pipeline and result.success and result.status_after != from_status:
            # Continue from the new status
            continuation = self.process_citation(
                citation_id,
                run_full_pipeline=True,
                stop_on_failure=stop_on_failure,
            )
            report.stages_run.extend(continuation.stages_run)

        report.print_summary()
        return report

    def get_status_label(self, status: int) -> str:
        return STATUS_LABELS.get(status, f"Unknown (status={status})")

    def list_citations(
        self,
        status: Optional[int] = None,
        year: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict]:
        """Return citation dicts filtered by status/year (either or both)."""
        params: Dict[str, object] = {
            "action": "list",
            "limit": limit,
            "offset": offset,
            "orderBy": "d.timestamp",
            "orderDir": "ASC",
        }
        if status is not None:
            params["status"] = status
        if year is not None:
            params["year"] = year
        return self.cit_client._api_call("CitationCRUD", params).get("documents", [])

    def list_citations_at_status(
        self,
        status: int,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict]:
        """Backward-compatible wrapper for status-only listing."""
        return self.list_citations(status=status, limit=limit, offset=offset)

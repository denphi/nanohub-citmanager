"""
Citation Manager Agents

A pipeline of LLM-powered agents that process citations one at a time
through a five-stage workflow:

  Status 1 → PDFVerifierAgent      — Verify PDF exists
  Status 2 → DisambiguationAgent   — Author/title disambiguation
  Status 3 → ClassificationAgent   — Field completeness check
  Status 4 → ReviewAgent           — Experimentalist & Experimental Data review
  Status 5 → HumanReviewAgent      — Prepare summary for human review
"""

from .base_agent import BaseCitationAgent, AgentResult
from .status1_pdf_verifier import PDFVerifierAgent
from .status2_disambiguation import DisambiguationAgent
from .status3_classification import ClassificationAgent
from .status4_review import ReviewAgent
from .status5_human_review import HumanReviewAgent
from .orchestrator import CitationPipelineOrchestrator, STATUS_LABELS

__all__ = [
    "BaseCitationAgent",
    "AgentResult",
    "PDFVerifierAgent",
    "DisambiguationAgent",
    "ClassificationAgent",
    "ReviewAgent",
    "HumanReviewAgent",
    "CitationPipelineOrchestrator",
    "STATUS_LABELS",
]

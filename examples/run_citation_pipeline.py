#!/usr/bin/env python3
"""
Citation Pipeline Runner

Process one citation (or a batch) through the five-stage automated pipeline:

  Status 1 — PDF Verification
  Status 2 — Author / Title Disambiguation
  Status 3 — Field Completeness Classification
  Status 4 — Dual Review (Experimentalist? / Experimental Data?)
  Status 5 — Human Review Package Preparation

Usage examples
--------------
# Process a single citation from wherever it currently is:
  python run_citation_pipeline.py --id 1234

# Process only the stage the citation is currently at (no auto-advance):
  python run_citation_pipeline.py --id 1234 --single-stage

# Re-run a specific stage regardless of current DB status:
  python run_citation_pipeline.py --id 1234 --from-status 2

# Process all citations currently at status 1 (limit 10):
  python run_citation_pipeline.py --status 1 --limit 10

# Process all citations from year 2025:
  python run_citation_pipeline.py --year 2025 --limit 20

# Process citations from year 2025 that are currently at status 3:
  python run_citation_pipeline.py --status 3 --year 2025 --limit 20

# Dry-run: list matching citations without processing:
  python run_citation_pipeline.py --status 3 --limit 20 --dry-run
"""

import os
import sys
import argparse

# Project root (parent of examples/) — needed for nanohubcitmanager
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EXAMPLES_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _EXAMPLES_DIR)  # needed for `from agents import`

from dotenv import load_dotenv
load_dotenv()

from nanohubremote import Session
from nanohubcitmanager import CitationManagerClient
from agents import CitationPipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_client() -> CitationManagerClient:
    hub_url = os.getenv("NANOHUB_URL", "https://nanohub.org/api")
    token = os.getenv("NANOHUB_TOKEN")
    if not token:
        print("Error: NANOHUB_TOKEN environment variable is not set.")
        print("  export NANOHUB_TOKEN='your-personal-token'")
        sys.exit(1)
    credentials = {"grant_type": "personal_token", "token": token}
    session = Session(credentials, url=hub_url, max_retries=1, timeout=60)
    return CitationManagerClient(session)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the citation processing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Target selection
    parser.add_argument(
        "--id",
        type=int,
        metavar="CITATION_ID",
        help="Process a single citation by ID.",
    )
    parser.add_argument(
        "--status",
        type=int,
        metavar="STATUS",
        choices=[1, 2, 3, 4, 5],
        help="Batch filter: citations currently at the given status.",
    )
    parser.add_argument(
        "--year",
        type=int,
        metavar="YEAR",
        help="Batch filter: citations from the given publication year.",
    )

    # Pipeline control
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Run only the current stage for each citation (no auto-advance).",
    )
    parser.add_argument(
        "--from-status",
        type=int,
        metavar="STATUS",
        choices=[1, 2, 3, 4, 5],
        help="Force a specific starting stage (only valid with --id).",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="When processing a batch, keep going even if one citation is blocked.",
    )

    # Batch options (used with --status)
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max citations to process in a batch run (default: 10).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N citations in a batch run (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching citations without processing them.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    if args.id is None and args.status is None and args.year is None:
        print("Error: specify either --id for single citation, or batch filters (--status and/or --year).")
        sys.exit(2)
    if args.id is not None and args.year is not None:
        print("Error: --year is a batch filter and cannot be used with --id.")
        sys.exit(2)
    if args.id is not None and args.status is not None and not args.from_status:
        print("Error: --status is a batch filter and cannot be used with --id (use --from-status instead).")
        sys.exit(2)

    print("=" * 60)
    print("  CITATION PIPELINE RUNNER")
    print("=" * 60)

    client = _build_client()
    orchestrator = CitationPipelineOrchestrator(client)

    # ── Single citation ──────────────────────────────────────────────
    if args.id is not None:
        citation_id = args.id

        if args.from_status:
            label = orchestrator.get_status_label(args.from_status)
            print(f"\n  Citation {citation_id}: forcing start at status {args.from_status} ({label})")
            orchestrator.process_from_status(
                citation_id=citation_id,
                from_status=args.from_status,
                run_full_pipeline=not args.single_stage,
            )
        else:
            citation = client.get(citation_id)
            label = orchestrator.get_status_label(citation.status)
            print(f"\n  Citation {citation_id}: current status {citation.status} ({label})")
            orchestrator.process_citation(
                citation_id=citation_id,
                run_full_pipeline=not args.single_stage,
            )
        return

    # ── Batch by filters (status/year) ────────────────────────────────
    status = args.status
    year = args.year
    status_label = orchestrator.get_status_label(status) if status is not None else None
    parts = []
    if status is not None:
        parts.append(f"status {status} ({status_label})")
    if year is not None:
        parts.append(f"year {year}")
    print(f"\n  Batch mode: {', '.join(parts)}, limit={args.limit}, offset={args.offset}")

    docs = orchestrator.list_citations(
        status=status,
        year=year,
        limit=args.limit,
        offset=args.offset,
    )

    if not docs:
        filter_msg = ", ".join(parts)
        print(f"\n  No citations found for {filter_msg}.")
        return

    print(f"\n  Found {len(docs)} citation(s):\n")
    for i, doc in enumerate(docs, 1):
        title = (doc.get("title") or "(no title)")[:65]
        print(f"  [{i:2}] ID={doc.get('id')}  {title}")

    if args.dry_run:
        print("\n  Dry-run mode — no processing performed.")
        return

    confirm = input(f"\n  Process these {len(docs)} citation(s)? (yes/no): ").strip().lower()
    if confirm not in ("yes", "y"):
        print("  Cancelled.")
        return

    success_count = 0
    fail_count = 0

    for i, doc in enumerate(docs, 1):
        cid = doc.get("id")
        print(f"\n[{i}/{len(docs)}] Processing citation {cid}...")
        try:
            report = orchestrator.process_citation(
                citation_id=cid,
                run_full_pipeline=not args.single_stage,
                stop_on_failure=True,
            )
            if report.success:
                success_count += 1
            else:
                fail_count += 1
                if not args.continue_on_failure:
                    print("  Stopping batch due to failure (use --continue-on-failure to skip).")
                    break
        except Exception as exc:
            print(f"  ERROR processing citation {cid}: {exc}")
            fail_count += 1
            if not args.continue_on_failure:
                break

    print("\n" + "=" * 60)
    print("  BATCH COMPLETE")
    print(f"  Success : {success_count}")
    print(f"  Failed  : {fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()

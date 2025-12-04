#!/usr/bin/env python3
"""
Batch Process Citations

This script loads documents with specific criteria from the Citation Manager
and processes them through the LLM metadata extraction pipeline.

Usage:
    python batch_process_citations.py [options]

Options:
    --status STATUS       Filter by status (default: 3)
    --year YEAR          Filter by specific year (default: 2025)
    --year-from YEAR     Filter by year from (inclusive)
    --year-to YEAR       Filter by year to (inclusive)
    --limit LIMIT        Number of documents to process (default: 5)
    --dry-run            Show documents without processing
    --skip-errors        Continue processing even if individual documents fail
"""

import os
import sys
import argparse
import subprocess
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath('..'))

from nanohubremote import Session
from nanohubcitmanager import CitationManagerClient


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch process citations through LLM metadata extraction'
    )

    parser.add_argument(
        '--status',
        type=int,
        default=3,
        help='Filter by document status (default: 3)'
    )

    parser.add_argument(
        '--year',
        type=int,
        help='Filter by specific year (e.g., 2025)'
    )

    parser.add_argument(
        '--year-from',
        type=int,
        help='Filter by year from (inclusive)'
    )

    parser.add_argument(
        '--year-to',
        type=int,
        help='Filter by year to (inclusive)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Maximum number of documents to process (default: 5)'
    )

    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Skip first N documents (default: 0)'
    )

    parser.add_argument(
        '--order-by',
        type=str,
        default='d.timestamp',
        help='Field to order by (default: d.timestamp)'
    )

    parser.add_argument(
        '--order-dir',
        type=str,
        default='ASC',
        choices=['ASC', 'DESC'],
        help='Sort direction (default: ASC)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show documents without processing them'
    )

    parser.add_argument(
        '--skip-errors',
        action='store_true',
        help='Continue processing even if individual documents fail'
    )

    parser.add_argument(
        '--full-details',
        action='store_true',
        help='Load full document details in list (slower but shows more info)'
    )

    return parser.parse_args()


def list_documents(
    client: CitationManagerClient,
    status: int = None,
    year: int = None,
    year_from: int = None,
    year_to: int = None,
    limit: int = 50,
    offset: int = 0,
    order_by: str = 'd.timestamp',
    order_dir: str = 'ASC',
    full_details: bool = False
) -> List[Dict[str, Any]]:
    """
    List documents with filtering and sorting.

    Args:
        client: CitationManagerClient instance
        status: Filter by status (optional)
        year: Filter by specific year (optional)
        year_from: Filter by year from (optional)
        year_to: Filter by year to (optional)
        limit: Maximum number of results
        offset: Skip first N results
        order_by: Field to sort by
        order_dir: Sort direction (ASC or DESC)
        full_details: Return full document details

    Returns:
        List of document dictionaries
    """
    params = {
        "action": "list",
        "limit": limit,
        "offset": offset,
        "orderBy": order_by,
        "orderDir": order_dir,
        "fullDetails": full_details
    }

    # Add optional filters
    if status is not None:
        params["status"] = status
    if year is not None:
        params["year"] = year
    if year_from is not None:
        params["yearFrom"] = year_from
    if year_to is not None:
        params["yearTo"] = year_to

    result = client._api_call("CitationCRUD", params)
    return result.get("documents", [])


def process_document(citation_id: int, script_path: str) -> bool:
    """
    Process a single document through the LLM extraction script.

    Args:
        citation_id: ID of the citation to process
        script_path: Path to llm_metadata_extraction.py

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*80}")
        print(f"PROCESSING CITATION ID: {citation_id}")
        print(f"{'='*80}\n")

        # Run the LLM extraction script
        result = subprocess.run(
            [sys.executable, script_path, str(citation_id)],
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=os.path.dirname(script_path)
        )

        if result.returncode == 0:
            print(f"\n✓ Successfully processed citation {citation_id}")
            return True
        else:
            print(f"\n✗ Failed to process citation {citation_id} (exit code: {result.returncode})")
            return False

    except Exception as e:
        print(f"\n✗ Error processing citation {citation_id}: {e}")
        return False


def main():
    """Main execution function."""
    args = parse_args()

    # Check environment variables
    hub_url = os.getenv("NANOHUB_URL", "https://nanohub.org/api")
    api_token = os.getenv("NANOHUB_TOKEN")

    if not api_token:
        print("Error: NANOHUB_TOKEN environment variable not set")
        print("Set it with: export NANOHUB_TOKEN='your-token-here'")
        sys.exit(1)

    print("=" * 80)
    print("BATCH CITATION PROCESSOR")
    print("=" * 80)
    print(f"Status filter: {args.status}")
    if args.year:
        print(f"Year filter: {args.year}")
    elif args.year_from or args.year_to:
        print(f"Year range: {args.year_from or 'any'} - {args.year_to or 'any'}")
    print(f"Limit: {args.limit}")
    print(f"Offset: {args.offset}")
    print(f"Order by: {args.order_by} {args.order_dir}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)

    # Create session and client
    print("\nConnecting to Citation Manager...")
    credentials = {
        "grant_type": "personal_token",
        "token": api_token
    }
    session = Session(credentials, url=hub_url, max_retries=1)
    client = CitationManagerClient(session)

    # List documents
    print("\nFetching documents...")
    documents = list_documents(
        client=client,
        status=args.status,
        year=args.year,
        year_from=args.year_from,
        year_to=args.year_to,
        limit=args.limit,
        offset=args.offset,
        order_by=args.order_by,
        order_dir=args.order_dir,
        full_details=args.full_details
    )

    if not documents:
        print("\nℹ No documents found matching criteria")
        sys.exit(0)

    print(f"\nFound {len(documents)} document(s):")
    print("=" * 80)
    for idx, doc in enumerate(documents, 1):
        doc_id = doc.get('id')
        title = doc.get('title', 'No title')
        year = doc.get('year', 'N/A')
        status = doc.get('status', 'N/A')
        timestamp = doc.get('timestamp', 'N/A')

        print(f"\n[{idx}] Citation ID: {doc_id}")
        print(f"    Title: {title[:70]}{'...' if len(title) > 70 else ''}")
        print(f"    Year: {year}")
        print(f"    Status: {status}")
        print(f"    Modified: {timestamp}")

    print("\n" + "=" * 80)

    if args.dry_run:
        print("\nDry run mode - skipping processing")
        sys.exit(0)

    # Confirm processing
    response = input(f"\nProcess these {len(documents)} document(s)? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Processing cancelled")
        sys.exit(0)

    # Get path to LLM extraction script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    llm_script = os.path.join(script_dir, "llm_metadata_extraction.py")

    if not os.path.exists(llm_script):
        print(f"\nError: LLM extraction script not found at: {llm_script}")
        sys.exit(1)

    # Process each document
    print("\n" + "=" * 80)
    print("STARTING BATCH PROCESSING")
    print("=" * 80)

    success_count = 0
    error_count = 0

    for idx, doc in enumerate(documents, 1):
        doc_id = doc.get('id')
        print(f"\n[{idx}/{len(documents)}] Processing citation {doc_id}...")

        success = process_document(doc_id, llm_script)

        if success:
            success_count += 1
        else:
            error_count += 1
            if not args.skip_errors:
                print("\nStopping due to error (use --skip-errors to continue)")
                break

    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {success_count}/{len(documents)}")
    print(f"Errors: {error_count}/{len(documents)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

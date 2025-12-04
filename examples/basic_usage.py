#!/usr/bin/env python3
"""
Basic usage example for nanohub-citmanager library.

This example shows the core functionality:
- Creating a session
- Creating a new citation
- Uploading a PDF
- Retrieving a citation
- Updating citation metadata
- Searching citations
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from nanohubremote import Session
from nanohubcitmanager import CitationManagerClient, Citation


def main():
    # Get credentials from environment
    api_token = os.getenv(
        "NANOHUB_TOKEN")
    hub_url = os.getenv("NANOHUB_URL")

    if not api_token:
        print("Error: NANOHUB_TOKEN environment variable not set")
        print("Set it with: export NANOHUB_TOKEN='your-token-here'")
        sys.exit(1)

    print("NanoHub Citation Manager - Basic Usage Example")
    print("=" * 50)
    print(f"Hub URL: {hub_url}\n")

    # Create session
    print("1. Creating session...")
    credentials = {
        "grant_type": "personal_token",
        "token": api_token
    }
    session = Session(credentials, url=hub_url, max_retries=1)
    client = CitationManagerClient(session)
    print("   ✓ Session created\n")

    # Create a new citation
    print("2. Creating new citation...")
    citation = Citation()
    citation.title = "Example Research Paper on Machine Learning"
    citation.abstract = "This paper presents a novel approach to machine learning using deep neural networks for scientific applications. We demonstrate significant improvements over existing methods."
    citation.year = 2024
    citation.doi = "10.1234/example.ml.2024"
    citation.url = "https://example.com/paper"
    citation.publication_name = "Journal of Machine Learning Research"  # Will auto-create if doesn't exist
    citation.document_genre_name = "article"  # Will auto-create if doesn't exist
    citation.publisher = "Academic Press"
    citation.volume = "42"
    citation.issue = "3"
    citation.begin_page = "123"
    citation.end_page = "145"


    # Add authors
    citation.add_author("Alice", "Johnson", email="alice.johnson@university.edu")
    citation.add_author("Bob", "Smith", orcid="0000-0001-2345-6789")
    citation.add_author("Carol", "Williams")

    # Add keywords
    citation.add_keyword("machine learning")
    citation.add_keyword("deep learning")
    citation.add_keyword("neural networks")
    citation.add_keyword("scientific computing")

    try:
        citation_id = client.create(citation)
        print(f"   ✓ Citation created with ID: {citation_id}\n")
    except Exception as e:
        print(f"   ✗ Error creating citation: {e}\n")
        sys.exit(1)

    # Retrieve the citation
    print(f"3. Retrieving citation {citation_id}...")
    try:
        retrieved = client.get(citation_id)
        print(f"   ✓ Citation retrieved:")
        print(f"     Title: {retrieved.title}")
        print(f"     Authors: {len(retrieved.authors)} author(s)")
        for i, author in enumerate(retrieved.authors, 1):
            print(f"       {i}. {author.get('firstname')} {author.get('lastname')}")
        print(f"     Year: {retrieved.year}")
        print(f"     DOI: {retrieved.doi}")
        print(f"     Keywords: {', '.join(retrieved.keywords)}\n")
    except Exception as e:
        print(f"   ✗ Error retrieving citation: {e}\n")

    # Update the citation
    print("4. Updating citation...")
    try:
        retrieved.abstract = retrieved.abstract + " This work was supported by NSF Grant #123456."
        retrieved.add_keyword("AI")
        retrieved.notes = "Added funding information and AI keyword"
        client.update(retrieved)
        print("   ✓ Citation updated\n")
    except Exception as e:
        print(f"   ✗ Error updating citation: {e}\n")

    # Search for citations
    print("5. Searching for citations containing 'machine learning'...")
    try:
        results = client.search("machine learning", limit=5)
        print(f"   ✓ Found {len(results)} result(s)")
        for i, cit in enumerate(results[:3], 1):
            print(f"     {i}. [{cit.id}] {cit.title[:60]}... ({cit.year})")
        print()
    except Exception as e:
        print(f"   ✗ Error searching: {e}\n")

    # List citations with filter
    print("6. Listing published citations...")
    try:
        documents = client.list(status=100, limit=5)  # 100 = PUBLISHED
        print(f"   ✓ Found {len(documents)} published citation(s)")
        for i, doc in enumerate(documents[:3], 1):
            print(f"     {i}. [{doc.get('ID')}] {doc.get('title', '')[:60]}...")
        print()
    except Exception as e:
        print(f"   ✗ Error listing: {e}\n")

    # PDF operations example (if PDF file exists)
    pdf_example = "example_paper.pdf"
    if os.path.exists(pdf_example):
        print(f"7. Uploading PDF file '{pdf_example}'...")
        try:
            client.upload_pdf(citation_id, pdf_example)
            print("   ✓ PDF uploaded\n")

            print("8. Getting PDF info...")
            info = client.get_pdf_info(citation_id)
            print(f"   ✓ PDF info:")
            print(f"     Filename: {info.get('filename')}")
            print(f"     Size: {info.get('size')} bytes")
            print(f"     Path: {info.get('path')}\n")

            print("9. Downloading PDF...")
            output_path = f"downloaded_{citation_id}.pdf"
            client.download_pdf(citation_id, output_path)
            print(f"   ✓ PDF downloaded to: {output_path}\n")

            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception as e:
            print(f"   ✗ Error with PDF operations: {e}\n")
    else:
        print(f"7. Skipping PDF operations ('{pdf_example}' not found)\n")

    # Summary
    print("=" * 50)
    print("Example completed successfully!")
    print(f"Created citation ID: {citation_id}")
    print("\nTo clean up, you can delete the citation with:")
    print(f"  client.delete({citation_id})")


if __name__ == "__main__":
    main()

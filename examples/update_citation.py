#!/usr/bin/env python3
"""
Example: Update an existing citation's volume field.

This example shows how to:
- Load an existing citation by ID
- Modify its volume field (increment it)
- Update the citation in the database
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from nanohubremote import Session
from nanohubcitmanager import CitationManagerClient


def main():
    # Get credentials from environment
    api_token = os.getenv("NANOHUB_TOKEN")
    hub_url = os.getenv("NANOHUB_URL")

    if not api_token:
        print("Error: NANOHUB_TOKEN environment variable not set")
        print("Set it with: export NANOHUB_TOKEN='your-token-here'")
        sys.exit(1)

    print("NanoHub Citation Manager - Update Citation Example")
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

    # Load the citation - get from command line argument or environment variable
    citation_id = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("CITATION_ID", "1"))
    print(f"2. Loading citation {citation_id}...")
    try:
        citation = client.get(citation_id)
        print(f"   ✓ Citation loaded:")
        print(f"     Title: {citation.title}")
        print(f"     Current Volume: '{citation.volume}'")
        print(f"     Year: {citation.year}")
        print()
    except Exception as e:
        print(f"   ✗ Error loading citation: {e}")
        sys.exit(1)

    # Increment the volume
    print("3. Incrementing volume...")
    original_volume = citation.volume
    
    # Handle volume increment - convert to string first
    if citation.volume is not None:
        # Convert to string and try to parse as integer
        try:
            volume_int = int(citation.volume)
            citation.volume = str(volume_int + 1)
        except (ValueError, TypeError):
            # If not a number, just append "+1"
            print(f"   ! Volume '{citation.volume}' is not numeric, appending '+1'")
            citation.volume = f"{citation.volume}+1"
    else:
        # If empty/None, set to "1"
        citation.volume = "1"
    
    print(f"   ✓ Volume updated: '{original_volume}' → '{citation.volume}'\n")


    # Update the citation
    print("4. Saving updated citation...")
    try:
        client.update(citation)
        print("   ✓ Citation updated successfully\n")
    except Exception as e:
        print(f"   ✗ Error updating citation: {e}\n")
        sys.exit(1)

    # Verify the update
    print("5. Verifying update...")
    try:
        updated_citation = client.get(citation_id)
        print(f"   ✓ Verification complete:")
        print(f"     New Volume: '{updated_citation.volume}'")
        print()
    except Exception as e:
        print(f"   ✗ Error verifying: {e}\n")

    # Summary
    print("=" * 50)
    print("Update completed successfully!")
    print(f"Citation {citation_id} volume changed from '{original_volume}' to '{citation.volume}'")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
from nanohubremote import Session
from nanohubcitmanager import CitationManagerClient
import json

# Environment variables
hub_url = os.getenv("NANOHUB_URL", "https://nanohub.org/api")
api_token = os.getenv("NANOHUB_TOKEN")

if not api_token:
    print("Error: NANOHUB_TOKEN environment variable not set")
    sys.exit(1)

# Get citation ID from command line or environment
citation_id = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("CITATION_ID", "1"))

# Create session and client
credentials = {
    "grant_type": "personal_token",
    "token": api_token
}
session = Session(credentials, url=hub_url, max_retries=1)
client = CitationManagerClient(session)

# Get citation
citation = client.get(citation_id)
print("Title:", citation.title)
print("\nAuthors:")
for i, author in enumerate(citation.authors, 1):
    print(f"\n  Author {i}:")
    print(f"    Name: {author.get('firstname', '')} {author.get('lastname', '')}")
    print(f"    Email: {author.get('email', '')}")
    print(f"    Organization: {author.get('organizationName', '')}")
    print(f"    Department: {author.get('departmentName', '')}")
    print(f"    ORCID: {author.get('orcid', '')}")
    print(f"    Scopus ID: {author.get('scopusId', '')}")
    print(f"    ResearcherID: {author.get('researcherId', '')}")
    print(f"    Google Scholar: {author.get('gsId', '')}")

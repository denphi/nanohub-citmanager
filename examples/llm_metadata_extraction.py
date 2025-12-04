#!/usr/bin/env python3
"""
Example: LLM-powered Citation Metadata Extraction

This example demonstrates how to:
1. Load a citation by ID from the Citation Manager
2. Download the associated PDF file
3. Use a local LLM (via Ollama) to extract and complete citation metadata
4. Update the citation with the extracted information

Requirements:
    - pip install nanohub-citmanager ollama pypdf2
    - Ollama running locally with a model installed (e.g., llama3, mistral)
"""

import os
import sys
import sys
import os
sys.path.insert(0,os.path.abspath('..'))

from typing import Dict, Any, Optional, List
from nanohubremote import Session
from nanohubcitmanager import CitationManagerClient, Citation
from PyPDF2 import PdfReader
import json
import requests


def search_publication(publication_name: str, api_url: str = "https://nanohub.org") -> Optional[Dict[str, Any]]:
    """
    Search for a publication using the Citation Manager API autocomplete.

    Args:
        publication_name: Name of the publication to search for
        api_url: Base API URL (default: https://nanohub.org)

    Returns:
        Dictionary with publication info including id, or None if not found
    """
    if not publication_name or not publication_name.strip():
        return None

    search_url = f"{api_url}/api/citmanager/search/SearchPublication"
    payload = {
        "params": {
            "search": publication_name.strip()
        }
    }

    try:
        response = requests.post(search_url, json=payload, timeout=10)
        response.raise_for_status()

        results = response.json()

        # Check if we got results
        if results and isinstance(results, list) and len(results) > 0:
            # Return the first/best match
            best_match = results[0]
            print(f"  ✓ Found publication match: {best_match.get('name', 'Unknown')} (ID: {best_match.get('id')})")
            return best_match
        else:
            print(f"  ℹ No publication match found for: {publication_name}")
            return None

    except Exception as e:
        print(f"  ✗ Error searching for publication: {e}")
        return None


def search_nanohub_resources(title: str, notes: str = None, api_url: str = "https://nanohub.org") -> List[Dict[str, Any]]:
    """
    Search for nanohub resources (publications or resources) related to a title.

    Args:
        title: Title of the publication to search for
        notes: Notes field that may contain nanoHUB resource/tool names
        api_url: Base API URL (default: https://nanohub.org)

    Returns:
        List of resource dictionaries with id, title, type, url
    """
    if not title or not title.strip():
        return []

    # Clean the title for search - remove special characters that cause Solr parse errors
    # Remove/escape characters: : ( ) [ ] { } " ' \ / + - && || ! ^ ~ * ?
    import re
    clean_title = re.sub(r'[:\(\)\[\]\{\}"\'\\/\+\-&|!^~*?]', ' ', title)
    # Collapse multiple spaces
    clean_title = re.sub(r'\s+', ' ', clean_title).strip()

    # If cleaned title is too short or empty, try with just first few significant words
    if len(clean_title) < 3:
        print(f"  ℹ Title too short after cleaning, skipping search")
        return []

    # Check notes for nanoHUB tool/resource names
    notes_terms = []
    if notes and isinstance(notes, str):
        # Look for patterns like "tool:", "resource:", or common nanoHUB references
        # Extract potential tool/resource names from notes
        notes_lower = notes.lower()
        if 'nanohub' in notes_lower or 'tool' in notes_lower or 'resource' in notes_lower:
            # Try to extract quoted strings or tool names
            tool_patterns = [
                r'tool[:\s]+([a-zA-Z0-9_\-]+)',
                r'resource[:\s]+([a-zA-Z0-9_\-]+)',
                r'https://nanohub\.org/(?:tools|resources)/([a-zA-Z0-9_\-]+)',
                r'"([^"]{3,30})"',  # Quoted strings
                r"'([^']{3,30})'"  # Single-quoted strings
            ]
            for pattern in tool_patterns:
                matches = re.findall(pattern, notes, re.IGNORECASE)
                notes_terms.extend(matches)

            if notes_terms:
                # Clean and limit to first few terms
                notes_terms = [re.sub(r'[^\w\s\-]', ' ', t).strip() for t in notes_terms if len(t) > 2]
                notes_terms = [t for t in notes_terms if t][:3]  # Limit to 3 terms
                if notes_terms:
                    print(f"  ℹ Found potential resource names in notes: {', '.join(notes_terms)}")

    # Use only first 100 chars to avoid overly long queries
    search_terms = clean_title[:100]

    # Collect all resources from different searches
    all_resources = []
    seen_ids = set()

    # Search both publications and resources
    search_url = f"{api_url}/api/search/list"

    # First, search with title
    params = {
        "terms": search_terms,
        "type": "publication OR resources",
        "limit": 10
    }

    try:
        response = requests.get(search_url, params=params, timeout=15)
        response.raise_for_status()

        results = response.json()

        if results and isinstance(results, dict) and 'results' in results:
            for item in results['results']:
                item_id = item.get('id')
                if item_id not in seen_ids:
                    resource = {
                        'id': item_id,
                        'title': item.get('title'),
                        'type': item.get('type'),
                        'url': item.get('url'),
                        'description': item.get('description', '')[:500]  # Limit description
                    }
                    all_resources.append(resource)
                    seen_ids.add(item_id)

            print(f"  ✓ Found {len(all_resources)} resources from title search")
        else:
            print(f"  ℹ No resources found from title search")

    except Exception as e:
        print(f"  ✗ Error searching with title: {e}")

    # Also search with terms from notes if available
    if notes_terms:
        for note_term in notes_terms:
            clean_note_term = re.sub(r'[:\(\)\[\]\{\}"\'\\/\+\-&|!^~*?]', ' ', note_term).strip()
            if len(clean_note_term) < 3:
                continue

            params = {
                "terms": clean_note_term,
                "type": "resources",  # Focus on resources for note terms
                "limit": 5
            }

            try:
                response = requests.get(search_url, params=params, timeout=15)
                response.raise_for_status()
                results = response.json()

                if results and isinstance(results, dict) and 'results' in results:
                    for item in results['results']:
                        item_id = item.get('id')
                        if item_id not in seen_ids:
                            resource = {
                                'id': item_id,
                                'title': item.get('title'),
                                'type': item.get('type'),
                                'url': item.get('url'),
                                'description': item.get('description', '')[:500]
                            }
                            all_resources.append(resource)
                            seen_ids.add(item_id)
                            print(f"  ✓ Found additional resource from notes term '{clean_note_term}': {item.get('title', 'Unknown')[:40]}")
            except Exception as e:
                print(f"  ℹ Error searching with note term '{clean_note_term}': {e}")

    if all_resources:
        print(f"  ✓ Total: {len(all_resources)} unique resources to verify")
        return all_resources
    else:
        print(f"  ℹ No resources found")
        return []


def find_duplicate_authors_in_citation(citation: Citation) -> List[tuple]:
    """
    Find duplicate authors within a citation based on name similarity.
    
    Args:
        citation: Citation object with authors
        
    Returns:
        List of tuples containing (author1_index, author2_index, similarity_score)
        for authors that appear to be duplicates
    """
    duplicates = []
    authors = citation.authors
    
    if not authors or len(authors) < 2:
        return duplicates
    
    # Check each pair of authors
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            author1 = authors[i]
            author2 = authors[j]
            
            # Get names (handle both dict and object attributes)
            if isinstance(author1, dict):
                lastname1 = author1.get('lastname', '').lower().strip()
                firstname1 = author1.get('firstname', '').lower().strip()
            else:
                lastname1 = getattr(author1, 'lastname', '').lower().strip()
                firstname1 = getattr(author1, 'firstname', '').lower().strip()
                
            if isinstance(author2, dict):
                lastname2 = author2.get('lastname', '').lower().strip()
                firstname2 = author2.get('firstname', '').lower().strip()
            else:
                lastname2 = getattr(author2, 'lastname', '').lower().strip()
                firstname2 = getattr(author2, 'firstname', '').lower().strip()
            
            # Check if they're duplicates
            # Same last name and same first initial = likely duplicate
            if lastname1 and lastname2 and lastname1 == lastname2:
                if firstname1 and firstname2:
                    # Same first initial
                    if firstname1[0] == firstname2[0]:
                        # Calculate similarity score (0-1)
                        # Exact match = 1.0, first initial match = 0.7
                        if firstname1 == firstname2:
                            similarity = 1.0
                        else:
                            similarity = 0.7
                        duplicates.append((i, j, similarity))
    
    return duplicates


def merge_duplicate_authors(
    client: 'CitationManagerClient',
    citation: Citation,
    author_index_primary: int,
    author_index_duplicate: int
) -> bool:
    """
    Merge duplicate authors by keeping the primary (oldest ID) and marking the other as duplicate.
    
    This function sets the aliasID in the person table to create the duplicate relationship,
    following the same pattern as workflow.php step3Task for author disambiguation.
    
    Args:
        client: CitationManagerClient instance
        citation: Citation object
        author_index_primary: Index of the author to keep (primary)
        author_index_duplicate: Index of the author to mark as duplicate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        authors = citation.authors
        if author_index_primary >= len(authors) or author_index_duplicate >= len(authors):
            print(f"  ✗ Invalid author indices: {author_index_primary}, {author_index_duplicate}")
            return False
        
        # Get author IDs (if they exist in the database)
        author_primary = authors[author_index_primary]
        author_duplicate = authors[author_index_duplicate]
        
        # Get IDs - handle both dict and object attributes
        if isinstance(author_primary, dict):
            id_primary = author_primary.get('id', 0)
            name_primary = f"{author_primary.get('firstname', '')} {author_primary.get('lastname', '')}"
        else:
            id_primary = getattr(author_primary, 'id', 0)
            name_primary = f"{getattr(author_primary, 'firstname', '')} {getattr(author_primary, 'lastname', '')}"
            
        if isinstance(author_duplicate, dict):
            id_duplicate = author_duplicate.get('id', 0)
            name_duplicate = f"{author_duplicate.get('firstname', '')} {author_duplicate.get('lastname', '')}"
        else:
            id_duplicate = getattr(author_duplicate, 'id', 0)
            name_duplicate = f"{getattr(author_duplicate, 'firstname', '')} {getattr(author_duplicate, 'lastname', '')}"
        
        # If authors don't have IDs yet, just remove the duplicate from the list
        if not id_primary or not id_duplicate:
            print(f"  ℹ Authors not yet in database, removing duplicate from list: {name_duplicate}")
            # Remove the duplicate author from the citation
            citation.authors.pop(author_index_duplicate)
            return True
        
        # Determine which has the older (smaller) ID - that should be the primary
        if id_duplicate < id_primary:
            # Swap so the older ID is always the primary
            id_primary, id_duplicate = id_duplicate, id_primary
            name_primary, name_duplicate = name_duplicate, name_primary
            # Also swap indices
            author_index_primary, author_index_duplicate = author_index_duplicate, author_index_primary
        
        print(f"  → Merging authors: keeping '{name_primary}' (ID: {id_primary}), marking '{name_duplicate}' (ID: {id_duplicate}) as duplicate")
        
        # Use the AuthorAlias endpoint to set the aliasID in the database
        # This follows the pattern from workflow.php lines 540-541:
        # SQL: UPDATE person SET aliasID='<primary_id>' WHERE ID='<duplicate_id>'
        
        params = {
            "action": "setAlias",
            "personId": id_duplicate,
            "aliasId": id_primary
        }
        
        try:
            # 1. First remove the duplicate author from the document using PersonDocument service
            # This ensures they don't show up in the author list anymore
            if citation.id:
                remove_params = {
                    "action": "remove",
                    "idDocument": citation.id,
                    "idPerson": id_duplicate
                }
                try:
                    # PersonDocument is in the default /citmanager/document path
                    client._api_call("PersonDocument", remove_params)
                    print(f"  ✓ Successfully removed author {id_duplicate} from document {citation.id}")
                except Exception as e:
                    print(f"  ✗ Failed to remove author from document: {e}")
            
            # 2. Then set the alias in the database
            # Use the correct base URL for the author alias endpoint
            # The default is /citmanager/document, but we need /citmanager/authoralias
            result = client._api_call("AuthorAlias", params, base_url="/citmanager/authoralias")
            print(f"  ✓ Successfully set author {id_duplicate} as alias of {id_primary} in database")
            
            # Also update the local citation object to reflect the change
            if isinstance(authors[author_index_duplicate], dict):
                authors[author_index_duplicate]['aliasID'] = id_primary
            else:
                setattr(authors[author_index_duplicate], 'aliasID', id_primary)
            
            return True
        except Exception as e:
            # If the endpoint doesn't exist, fall back to removing from list
            print(f"  ℹ AuthorAlias endpoint not available ({e})")
            print(f"  → Removing duplicate from citation author list as fallback")
            citation.authors.pop(author_index_duplicate)
            return True
            
    except Exception as e:
        print(f"  ✗ Error merging duplicate authors: {e}")
        return False


def check_and_merge_duplicate_authors(
    client: 'CitationManagerClient',
    citation: Citation
) -> int:
    """
    Check for duplicate authors in a citation and merge them automatically.
    
    Args:
        client: CitationManagerClient instance
        citation: Citation object
        
    Returns:
        Number of duplicates merged
    """
    print("\n" + "=" * 60)
    print("CHECKING FOR DUPLICATE AUTHORS")
    print("=" * 60)
    
    duplicates = find_duplicate_authors_in_citation(citation)
    
    if not duplicates:
        print("  ✓ No duplicate authors found")
        print("=" * 60 + "\n")
        return 0
    
    print(f"\n→ Found {len(duplicates)} potential duplicate(s)\n")
    
    merged_count = 0
    # Process duplicates in reverse order to avoid index shifting issues
    duplicates_sorted = sorted(duplicates, key=lambda x: x[1], reverse=True)
    
    for idx, (i, j, similarity) in enumerate(duplicates_sorted, 1):
        authors = citation.authors
        
        # Get author names for display
        if isinstance(authors[i], dict):
            name1 = f"{authors[i].get('firstname', '')} {authors[i].get('lastname', '')}"
            name2 = f"{authors[j].get('firstname', '')} {authors[j].get('lastname', '')}"
        else:
            name1 = f"{getattr(authors[i], 'firstname', '')} {getattr(authors[i], 'lastname', '')}"
            name2 = f"{getattr(authors[j], 'firstname', '')} {getattr(authors[j], 'lastname', '')}"
        
        print(f"  [{idx}/{len(duplicates)}] Duplicate detected (similarity: {similarity:.2f}):")
        print(f"    Author {i}: {name1}")
        print(f"    Author {j}: {name2}")
        
        # Merge: keep the one with the lower index (appears first in author list)
        # and mark the later one as duplicate
        if merge_duplicate_authors(client, citation, i, j):
            merged_count += 1
            print(f"    ✓ Merged successfully\n")
        else:
            print(f"    ✗ Merge failed\n")
    
    print(f"{'=' * 60}")
    print(f"DUPLICATE AUTHOR CHECK COMPLETE: {merged_count} author(s) merged")
    print(f"{'=' * 60}\n")
    
    return merged_count


def extract_metadata_from_doi(doi: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from a DOI using the CrossRef API.

    Args:
        doi: DOI string (e.g., "10.1234/example")

    Returns:
        Dictionary with metadata or None if failed
    """
    try:
        # Use CrossRef API for DOI metadata
        crossref_url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(crossref_url, timeout=10)
        response.raise_for_status()

        data = response.json()
        if 'message' not in data:
            return None

        message = data['message']
        metadata = {}

        # Extract title
        if 'title' in message and message['title']:
            metadata['title'] = message['title'][0]

        # Extract authors
        if 'author' in message:
            authors = []
            for author in message['author']:
                author_data = {}
                if 'given' in author:
                    author_data['firstname'] = author['given']
                if 'family' in author:
                    author_data['lastname'] = author['family']
                if 'ORCID' in author:
                    author_data['orcid'] = author['ORCID'].replace('http://orcid.org/', '')
                if author_data:
                    authors.append(author_data)
            if authors:
                metadata['authors'] = authors

        # Extract publication info
        if 'published-print' in message and 'date-parts' in message['published-print']:
            year = message['published-print']['date-parts'][0][0]
            metadata['year'] = year
        elif 'published-online' in message and 'date-parts' in message['published-online']:
            year = message['published-online']['date-parts'][0][0]
            metadata['year'] = year

        if 'DOI' in message:
            metadata['doi'] = message['DOI']

        if 'container-title' in message and message['container-title']:
            metadata['journal'] = message['container-title'][0]
            metadata['publication'] = message['container-title'][0]

        if 'volume' in message:
            metadata['volume'] = str(message['volume'])

        if 'issue' in message:
            metadata['issue'] = str(message['issue'])

        if 'page' in message:
            metadata['pages'] = message['page']

        if 'publisher' in message:
            metadata['publisher'] = message['publisher']

        if 'abstract' in message:
            metadata['abstract'] = message['abstract']

        # Determine publication type
        if 'type' in message:
            type_map = {
                'journal-article': 'journal',
                'proceedings-article': 'conference',
                'book-chapter': 'book',
                'book': 'book',
                'dissertation': 'thesis',
                'report': 'report',
                'posted-content': 'preprint'
            }
            metadata['publication_type'] = type_map.get(message['type'], 'article')

        print(f"  ✓ Extracted metadata from DOI via CrossRef API")
        return metadata

    except Exception as e:
        print(f"  ℹ CrossRef API failed: {e}")
        return None


def extract_metadata_from_url(name: str, model: str, api_key: str, api_url: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from a URL (e.g., DOI, arXiv, journal webpage).

    Args:
        url: URL to extract metadata from
        model: Model name for LLM
        api_key: API key
        api_url: API endpoint

    Returns:
        Dictionary with extracted metadata or None if failed
    """
        
    url = "https://api.crossref.org/works?rows=1&query=" + name

    try:
        # Fetch the URL content with realistic browser headers
        headers = {
            'accept': 'application/json'
        }

        response = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status()

        # Get the content (limit to first 10000 chars to avoid huge pages)
        content = response.text[:10000]

        # Create a prompt to extract metadata from the webpage
        prompt = f"""You are a metadata extraction assistant. Extract bibliographic information from this crossref json.

URL: {url}

Webpage content (first 10000 chars):
{content}

========================
EXTRACTION TASK
========================

Extract any available bibliographic metadata from this webpage. Common sources include:
- Meta tags (DC.title, citation_title, DC.creator, citation_author, etc.)
- Structured data (JSON-LD, Schema.org)
- Visible page content (title, author names, publication info)
- DOI metadata pages
- arXiv pages
- Journal article pages

Extract these fields if available:
- title (paper/article title)
- authors (array of author objects with firstname, lastname, and any other details)
- year (publication year)
- doi (if this is a DOI URL or DOI is mentioned)
- journal or publication (publication venue name)
- publication_type (journal, conference, preprint, etc.)
- volume, issue, pages (if available)
- abstract (if available and not too long)
- publisher (if mentioned)

IMPORTANT:
- Only extract what is clearly present in the content
- For authors, extract firstname and lastname at minimum
- Return empty object {{}} if no useful metadata found

========================
OUTPUT FORMAT
========================
Output ONLY a JSON object:
{{
  "title": "...",
  "authors": [{{"firstname": "...", "lastname": "...", "organizationname": "...", "organizationdept": "...", "countryresident": "...", "countrySHORT": "...", "countryLONG": "..."}}],
  "year": 2024,
  "doi": "...",
  "publicationName": "...",
  "documentGenreName": "...",
  "volume": "...",
  "issue": "...",
  "pages": "...",
  "abstract": "..."
}}

Output ONLY the JSON object, no explanation."""

        wrapper = OpenWebUIWrapper(model, 0.0, api_key, api_url)
        llm_response = wrapper.chat([{'role': 'user', 'content': prompt}])

        # Parse the response
        metadata = parse_llm_response(llm_response)

        if metadata and len(metadata) > 0:
            print(f"  ✓ Extracted {len(metadata)} fields from URL")
            return metadata
        else:
            print(f"  ℹ No metadata extracted from URL")
            return None

    except Exception as e:
        print(f"  ✗ Error extracting metadata from URL: {e}")
        return None


# Try to import ollama (optional)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def extract_text_from_pdf(pdf_path: str, max_pages: int = 5) -> str:
    """
    Extract text from the first few pages of a PDF.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to extract

    Returns:
        Extracted text
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages[:max_pages]):
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""


def extract_pdf_sections(pdf_path: str, max_pages: Optional[int] = 15) -> Dict[str, str]:
    """
    Extract text from PDF with intelligent section detection.

    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to extract (None for all pages)

    Returns:
        Dictionary with sections: 'intro', 'methods', 'acknowledgments', 'references', 'full'
    """
    import re

    try:
        reader = PdfReader(pdf_path)
        full_text = ""

        # Extract text from pages
        pages_to_process = reader.pages if max_pages is None else reader.pages[:max_pages]
        for i, page in enumerate(pages_to_process):
            full_text += f"\n--- PAGE {i+1} ---\n"
            full_text += page.extract_text() + "\n"

        sections = {
            'full': full_text,
            'intro': '',
            'methods': '',
            'acknowledgments': '',
            'references': ''
        }

        # Extract first 3 pages as intro/abstract
        intro_pages = min(3, len(reader.pages))
        intro_text = ""
        for i in range(intro_pages):
            intro_text += reader.pages[i].extract_text() + "\n"
        sections['intro'] = intro_text

        # Try to find methods/experimental section
        methods_patterns = [
            r'(?i)(^|\n)\s*(methods?|experimental|materials?\s+and\s+methods?|procedure)',
            r'(?i)(^|\n)\s*\d+\.?\s+(methods?|experimental|materials?\s+and\s+methods?)'
        ]

        for pattern in methods_patterns:
            match = re.search(pattern, full_text)
            if match:
                start_idx = match.start()
                # Extract next ~2000 characters after methods header
                sections['methods'] = full_text[start_idx:start_idx + 2000]
                break

        # Try to find acknowledgments section
        ack_patterns = [
            r'(?i)(^|\n)\s*acknowledgm?e?nts?\s*[:\n]',
            r'(?i)(^|\n)\s*author\s+contributions?\s*[:\n]'
        ]

        for pattern in ack_patterns:
            match = re.search(pattern, full_text)
            if match:
                start_idx = match.start()
                # Extract next ~1000 characters after acknowledgments header
                sections['acknowledgments'] = full_text[start_idx:start_idx + 1000]
                break

        # Try to find references section
        ref_patterns = [
            r'(?i)(^|\n)\s*references?\s*[:\n]',
            r'(?i)(^|\n)\s*bibliography\s*[:\n]',
            r'(?i)(^|\n)\s*\d+\.?\s*references?\s*[:\n]'
        ]

        for pattern in ref_patterns:
            match = re.search(pattern, full_text)
            if match:
                start_idx = match.start()
                # Extract next ~5000 characters after references header (increased from 2000)
                sections['references'] = full_text[start_idx:start_idx + 5000]
                break

        return sections

    except Exception as e:
        print(f"Error extracting PDF sections: {e}")
        return {
            'full': '',
            'intro': '',
            'methods': '',
            'acknowledgments': '',
            'references': ''
        }


def create_author_extraction_prompt(pdf_sections: Dict[str, str], existing_citation: Citation) -> str:
    """
    Create a focused prompt for extracting author information.

    Args:
        pdf_sections: Dictionary of extracted PDF sections
        existing_citation: Current citation data

    Returns:
        Prompt string for author extraction
    """
    # Combine intro and acknowledgments for author extraction
    relevant_text = pdf_sections['intro']
    if pdf_sections['acknowledgments']:
        relevant_text += "\n\n--- ACKNOWLEDGMENTS SECTION ---\n" + pdf_sections['acknowledgments']

    prompt = f"""You are a research paper metadata extraction assistant. Extract ONLY author information from the provided text.

Current citation data:
{json.dumps({'authors': [a for a in existing_citation.to_dict().get('authors', [])]}, indent=2)}

Paper text:
{relevant_text[:6000]}

========================
EXTRACTION RULES
========================

Extract ALL information for each author:
- firstname (required)
- lastname (required)
- email (look in: author list, footnotes, "corresponding author", contact notes, acknowledgments)
- orcid (look for: ORCID icon, orcid.org link, or 0000-0000-0000-0000 format)
- scopusId (if explicitly mentioned)
- researcherid, gsid, researchgateid (only if present)
- organizationname (institution from affiliation superscripts/footnotes)
- departmentname (if listed with affiliation)
- countrySHORT (2-letter ISO country code from affiliation - e.g., US, UK, DE, CN, JP)
- countryLONG (full country name from affiliation - e.g., United States, United Kingdom, Germany)
- Always set `organization` and `department` to 0 (IDs are resolved elsewhere)

IMPORTANT:
- Extract ONLY what is explicitly stated in the text
- Never guess or infer missing information
- If an author has no email/ORCID/country visible, omit those fields
- Pay attention to affiliation superscripts (¹, ², *, †, etc.) linking authors to institutions
- For country: extract from institution affiliation (e.g., "MIT, USA" -> countrySHORT: "US", countryLONG: "United States")

========================
OUTPUT FORMAT
========================
Output ONLY a JSON object with this structure:
{{
  "authors": [
    {{
      "firstname": "...",
      "lastname": "...",
      "email": "...",
      "orcid": "...",
      "organizationname": "...",
      "departmentname": "...",
      "countrySHORT": "US",
      "countryLONG": "United States",
      "organization": 0,
      "department": 0
    }}
  ]
}}

Output ONLY the JSON object, no explanation."""
    return prompt


def create_paper_details_prompt(pdf_sections: Dict[str, str], existing_citation: Citation) -> str:
    """
    Create a focused prompt for extracting paper bibliographic details.

    Args:
        pdf_sections: Dictionary of extracted PDF sections
        existing_citation: Current citation data

    Returns:
        Prompt string for paper details extraction
    """
    # Build pages string outside to avoid f-string nesting issues
    pages_str = f"{existing_citation.begin_page}-{existing_citation.end_page}" if existing_citation.begin_page else None

    citation_data = {
        'title': existing_citation.title,
        'abstract': existing_citation.abstract,
        'year': existing_citation.year,
        'doi': existing_citation.doi,
        'journal': existing_citation.publication_name,
        'volume': existing_citation.volume,
        'issue': existing_citation.issue,
        'pages': pages_str,
        'publisher': existing_citation.publisher
    }

    prompt = f"""You are a research paper metadata extraction assistant. Extract ONLY bibliographic details from the provided text.

Current citation data:
{json.dumps(citation_data, indent=2)}

Paper text (first pages):
{pdf_sections['intro'][:5000]}

========================
EXTRACTION RULES
========================

Extract the following fields if they are present and can be improved:
- title (full paper title)
- abstract (COMPLETE abstract text - extract the ENTIRE abstract, not a summary)
- year (publication year)
- doi (Digital Object Identifier)
- journal (journal name if it's a journal article)
- publication (publication venue name - could be journal, conference, book title)
- publication_type (type of publication: "journal", "conference", "book", "thesis", "report", "preprint", etc.)
- volume (volume number)
- issue (issue number)
- pages (page range, e.g., "123-145")
- publisher (publisher name)
- isbn (if present, for books/proceedings)
- keywords (2-4 main keywords from Keywords section or recurring concepts)

IMPORTANT:
- Extract ONLY what is explicitly stated
- For abstract: extract the FULL text, do not truncate or summarize
- If existing data is already correct, omit that field
- Never guess or infer
- For publication_type: determine from context (journal article, conference paper, book chapter, etc.)

========================
OUTPUT FORMAT
========================
Output ONLY a JSON object with the fields you found:
{{
  "title": "...",
  "abstract": "FULL COMPLETE ABSTRACT TEXT HERE - DO NOT TRUNCATE",
  "year": 2024,
  "doi": "...",
  "journal": "...",
  "publication": "...",
  "publication_type": "journal",
  "volume": "...",
  "issue": "...",
  "pages": "123-145",
  "publisher": "...",
  "keywords": ["keyword1", "keyword2"]
}}

Output ONLY the JSON object, no explanation."""
    return prompt


def create_experimental_data_prompt(pdf_sections: Dict[str, str]) -> str:
    """
    Create a focused prompt for assessing experimental data flags.

    Args:
        pdf_sections: Dictionary of extracted PDF sections

    Returns:
        Prompt string for experimental assessment
    """
    # Combine methods and intro for experimental assessment
    relevant_text = pdf_sections['intro'][:3000]
    if pdf_sections['methods']:
        relevant_text += "\n\n--- METHODS/EXPERIMENTAL SECTION ---\n" + pdf_sections['methods']

    prompt = f"""You are a research paper classification assistant. Assess whether this paper contains experimental data.

Paper text:
{relevant_text[:5000]}

========================
ASSESSMENT RULES
========================

Determine TWO boolean flags:

1. expData (Experimental Data):
   Set to true ONLY if the paper clearly describes:
   - Experimental sections (Methods, Experimental Setup, Measurements, Fabrication)
   - Physical measurements, samples, materials, instrumentation
   - Laboratory procedures, equipment, protocols
   - Data collection from experiments (not just simulations)

   Set to false if:
   - Paper is purely theoretical, computational, or review
   - Only contains simulations without physical experiments
   - No clear experimental methodology described

2. expListExpData (Experimentalist Authors):
   Set to true ONLY if:
   - Author affiliations explicitly reference experimental labs, materials labs, physics/chemistry labs
   - Authors are described as performing hands-on experiments
   - Clear evidence authors work in experimental research groups

   Set to false if:
   - Authors are from purely computational/theoretical groups
   - No clear experimental affiliation
   - Insufficient information about author research focus

IMPORTANT:
- Be conservative: only set to true with strong evidence
- Theoretical papers analyzing experimental data from others = false for both
- Simulation-only papers = false for both

========================
OUTPUT FORMAT
========================
Output ONLY a JSON object:
{{
  "expData": true or false,
  "expListExpData": true or false,
  "reasoning": "Brief 1-2 sentence explanation"
}}

Output ONLY the JSON object, no additional text."""
    return prompt


def create_classification_prompt(pdf_sections: Dict[str, str]) -> str:
    """
    Create a focused prompt for classifying the paper type (R, C, E, N).

    Args:
        pdf_sections: Dictionary of extracted PDF sections

    Returns:
        Prompt string for classification
    """
    # Combine intro and abstract
    relevant_text = pdf_sections['intro'][:5000]

    prompt = f"""You are a research paper classification assistant. Classify this paper into one or more categories.

Paper text:
{relevant_text}

========================
CLASSIFICATION CATEGORIES
========================

Determine if the paper belongs to any of these categories (can be multiple):

1. Research (R):
   - Presents original research results, new methods, or scientific discoveries.
   - Most standard journal articles and conference papers fall here.
   - Set to true for almost all scientific papers.

2. Cyberinfrastructure (C):
   - Focuses on software tools, simulation platforms, databases, or computing infrastructure.
   - Describes development of code, web tools, or middleware.
   - Mentions "cyberinfrastructure", "hub", "platform", "gateway", "software tool".

3. Education (E):
   - Focuses on teaching, learning, curriculum, or educational tools.
   - Published in education journals (e.g., JEE, FIE).
   - Discusses student learning, pedagogy, classroom usage, or tutorials.

4. Nano Research (N):
   - Topic is related to nanotechnology, nanoscience, materials science at nanoscale.
   - Keywords: nano*, quantum, atomistic, molecular, materials, device physics.
   - Most papers in this domain will be true.

========================
OUTPUT FORMAT
========================
Output ONLY a JSON object:
{{
  "is_research": true or false,
  "is_cyberinfrastructure": true or false,
  "is_education": true or false,
  "is_nano": true or false
}}

Output ONLY the JSON object, no additional text."""
    return prompt


def create_resource_matching_prompt(citation_title: str, citation_abstract: str, resource: Dict[str, Any]) -> str:
    """
    Create a prompt for LLM to verify if a resource matches the citation.

    Args:
        citation_title: Title of the citation
        citation_abstract: Abstract of the citation
        resource: Resource dictionary with title, description, type

    Returns:
        Prompt string for resource matching
    """
    prompt = f"""You are a research paper matching assistant. Determine if this nanohub resource is related to the given citation.

CITATION:
Title: {citation_title}
Abstract: {citation_abstract[:500] if citation_abstract else "Not available"}

NANOHUB RESOURCE:
Title: {resource.get('title', 'Unknown')}
Type: {resource.get('type', 'Unknown')}
Description: {resource.get('description', 'No description')}
URL: {resource.get('url', '')}

========================
MATCHING RULES
========================

Determine if the resource is DIRECTLY related to the citation. A resource is related if:
- It is a simulation tool, dataset, or educational material mentioned in the paper
- It is developed by the same authors for the same research
- It is explicitly cited or referenced in the paper
- It implements the methods or algorithms described in the paper

DO NOT match if:
- Only general topic similarity (e.g., both about nanoscience)
- Different authors and no clear connection
- Resource is from a different time period with no citation
- Only tangentially related research area

Be CONSERVATIVE - only match when you are confident there is a direct relationship.

========================
OUTPUT FORMAT
========================
Output ONLY a JSON object:
{{
  "is_match": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief 1-2 sentence explanation"
}}

Output ONLY the JSON object, no additional text."""
    return prompt


def verify_resource_match_with_llm(
    citation: Citation,
    resource: Dict[str, Any],
    model: str,
    api_key: str,
    api_url: str
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to verify if a resource matches the citation.

    Args:
        citation: Citation object
        resource: Resource dictionary
        model: Model name
        api_key: API key
        api_url: API endpoint

    Returns:
        Dictionary with is_match, confidence, reasoning or None if failed
    """
    prompt = create_resource_matching_prompt(
        citation.title or "",
        citation.abstract or "",
        resource
    )

    try:
        wrapper = OpenWebUIWrapper(model, 0.1, api_key, api_url)
        content = wrapper.chat([{'role': 'user', 'content': prompt}])
        result = parse_llm_response(content)
        return result
    except Exception as e:
        print(f"  ✗ Error verifying resource match: {e}")
        return None


def add_resource_association(
    client: 'CitationManagerClient',
    citation_id: int,
    resource_id: str,
    resource_type: str
) -> bool:
    """
    Add a resource association to a citation using DocumentAssociationTask.

    Args:
        client: CitationManagerClient instance
        citation_id: ID of the citation
        resource_id: ID of the resource (as string, may include alias)
        resource_type: Type of resource (e.g., 'resources', 'publication')

    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract table name from resource_id prefix
        # IDs can be: numeric (123), prefixed ("resource-123", "publication-456"), or alias ("tools:atomistix")
        if "-" in resource_id and resource_id.split("-")[0] in ["resource", "publication"]:
            # Prefixed ID like "resource-43724" or "publication-123"
            # Use the prefix as the table name (already singular form)
            table_name = resource_id.split("-")[0]
            assoc_id = int(resource_id.split("-")[-1])
        elif resource_id.isdigit():
            # Pure numeric ID - fall back to resource_type parameter
            table_name = "resource" if resource_type == "resources" else "publication"
            assoc_id = int(resource_id)
        elif ":" in resource_id:
            # Alias like "tools:atomistix" - fall back to resource_type parameter
            table_name = "resource" if resource_type == "resources" else "publication"
            assoc_id = resource_id
        else:
            print(f"  ✗ Invalid resource ID format: {resource_id}")
            return False

        if not assoc_id:
            print(f"  ✗ Invalid resource ID: {resource_id}")
            return False

        params = {
            "action": "add",
            "idDocument": citation_id,
            "assocName": table_name,
            "assocID": assoc_id
        }

        result = client._api_call("DocumentAssociation", params)

        if result.get("status") == "OK":
            print(f"  ✓ Added association: {resource_id} ({table_name})")
            return True
        else:
            print(f"  ℹ Association result: {result.get('message', 'Unknown')}")
            return False

    except Exception as e:
        print(f"  ✗ Error adding resource association: {e}")
        return False


def find_and_add_nanohub_resources(
    client: 'CitationManagerClient',
    citation: Citation,
    model: str,
    api_key: str,
    api_url: str,
    hub_url: str = "https://nanohub.org"
) -> int:
    """
    Search for and add nanohub resources related to a citation.

    Args:
        client: CitationManagerClient instance
        citation: Citation object
        model: Model name for LLM verification
        api_key: API key for LLM
        api_url: API endpoint for LLM
        hub_url: NanoHub base URL

    Returns:
        Number of resources added
    """
    print("\n" + "=" * 60)
    print("SEARCHING FOR RELATED NANOHUB RESOURCES")
    print("=" * 60)

    if not citation.title:
        print("  ℹ No title available for resource search")
        return 0

    # Search for resources
    print(f"\n→ Searching nanohub for: {citation.title[:80]}...")
    # Pass notes field to help find resources mentioned in the citation
    citation_notes = getattr(citation, 'notes', None)
    resources = search_nanohub_resources(citation.title, notes=citation_notes, api_url=hub_url)

    if not resources:
        print("  ℹ No resources found")
        return 0

    print(f"\n→ Found {len(resources)} potential resources, verifying with LLM...")
    added_count = 0

    for idx, resource in enumerate(resources, 1):
        print(f"\n  [{idx}/{len(resources)}] Checking: {resource.get('title', 'Unknown')[:60]}...")

        # Verify match with LLM
        match_result = verify_resource_match_with_llm(
            citation, resource, model, api_key, api_url
        )

        if match_result and match_result.get('is_match'):
            confidence = match_result.get('confidence', 0.0)
            reasoning = match_result.get('reasoning', '')

            # Only add if confidence is high enough (>= 0.7)
            if confidence >= 0.7:
                print(f"    ✓ Match confirmed (confidence: {confidence:.2f})")
                print(f"    Reasoning: {reasoning}")

                # Add the association
                if add_resource_association(
                    client,
                    citation.id,
                    str(resource.get('id')),
                    resource.get('type', 'resources')
                ):
                    added_count += 1
            else:
                print(f"    ℹ Low confidence ({confidence:.2f}), skipping")
                print(f"    Reasoning: {reasoning}")
        else:
            print(f"    ✗ No match")
            if match_result:
                print(f"    Reasoning: {match_result.get('reasoning', 'Unknown')}")

    print(f"\n{'=' * 60}")
    print(f"RESOURCE ASSOCIATION COMPLETE: {added_count} resources added")
    print(f"{'=' * 60}\n")

    return added_count


def create_nanohub_relationship_prompt(text_chunk: str, chunk_num: int, total_chunks: int) -> str:
    """
    Create a focused prompt for assessing nanoHUB relationship in a text chunk.

    Args:
        text_chunk: A chunk of text from the paper
        chunk_num: Current chunk number (1-indexed)
        total_chunks: Total number of chunks

    Returns:
        Prompt string for nanoHUB assessment
    """
    prompt = f"""You are a research paper relationship analyzer. You are analyzing chunk {chunk_num} of {total_chunks} from a research paper.

Your task: Search this text chunk for ANY mention or reference to nanoHUB or NCN (Network for Computational Nanotechnology).

Paper text chunk {chunk_num}/{total_chunks}:
{text_chunk}

========================
WHAT IS NANOHUB/NCN?
========================

nanoHUB (https://nanohub.org) is an online platform for nanotechnology research.
NCN (Network for Computational Nanotechnology) is the organization that operates nanoHUB.

========================
SEARCH FOR ANY OF THESE
========================

Look for ANY of these variations (case-insensitive):
- "nanoHUB", "nanohub", "NanoHUB", "Nanohub", "nano-HUB", "nano HUB"
- "nanohub.org", "www.nanohub.org", "https://nanohub.org"
- "doi:10.xxxx/nanohub" or "doi.org/10.xxxx/nanohub"
- "NCN", "Network for Computational Nanotechnology"
- References/citations containing "nanohub" or "NCN"
- Acknowledgments mentioning nanoHUB or NCN or "Network for Computational Nanotechnology"
- Tool names with nanoHUB URLs
- Author affiliations with "NCN" or "Network for Computational Nanotechnology"

BE VERY THOROUGH - scan every line carefully.

========================
OUTPUT FORMAT
========================
Output ONLY a JSON object:
{{
  "foundInChunk": true or false,
  "evidence": "If found, quote the exact sentence/reference; otherwise empty string",
  "isAffiliated": true or false
}}

Set "isAffiliated" to true IF AND ONLY IF:
- The text explicitly states the work is supported by, funded by, or affiliated with NCN or nanoHUB.
- An author's affiliation explicitly mentions "NCN" or "Network for Computational Nanotechnology".
- The work is described as a "nanoHUB resource" or "NCN project".

Output ONLY the JSON object, no additional text."""
    return prompt


def create_extraction_prompt(pdf_text: str, existing_citation: Citation) -> str:
    """
    Create a prompt for the LLM to extract citation metadata.

    Args:
        pdf_text: Extracted text from PDF
        existing_citation: Current citation data

    Returns:
        Prompt string
    """
    prompt = f"""You are a research paper metadata extraction assistant. Your task is to analyze the provided research paper text and complete the citation metadata JSON object.

Current citation data:
{existing_citation.to_dict().__str__()}

Paper text (first {len(pdf_text)} characters):
{pdf_text[:4000]}

Your goal is to extract only verifiable metadata from the text and output a corrected and completed JSON object.

========================
EXTRACTION RULES
========================

GENERAL:
- Include ONLY fields for which explicit information appears in the text.
- If a field exists in the current citation and cannot be improved or corrected, omit it from the output.
- Never guess or infer content not present in the text.

AUTHORS (CRITICAL):
For each author, extract ALL available information:
- firstname (required)
- lastname (required)
- email (author list, footnotes, “corresponding author”, contact notes, acknowledgments)
- orcid (ORCID icon, orcid.org link, or 0000-0000-0000-0000 format)
- scopusId (explicitly referenced)
- other researcher IDs (ResearcherID, Google Scholar, ResearchGate) only if present
- organizationname (institution from affiliation superscripts)
- departmentname (if listed)
- Always set `organization` and `department` to 0 (IDs are resolved elsewhere)

EXPERIMENTAL FLAGS:
- `expData`: true only if the paper text clearly describes:
  - experimental sections (Methods, Experimental Setup, Measurement, etc.)
  - measurements, samples, fabrication, materials, instrumentation, etc.
- `expListExpData`: true only if:
  - author affiliations explicitly reference labs, experimental groups, materials/chemistry/physics labs, OR
  - authors describe performing experiments.
- Do NOT set these unless strong evidence is present.

NANOHUB:
- Always include a `notes` object in the JSON output.
- Include a `nanohub` field inside `notes`.
- If the paper explicitly references nanoHUB, any of its tools, datasets, notebooks, simulations, DOIs, or other resources:
    - Set `notes.nanohub` to a short explanation of the relationship.
    - Example: "nanohub is related because the paper uses nanoHUB simulation tools and cites nanoHUB resources."
- If there is no evidence of nanoHUB references:
    - Set `notes.nanohub` to a clear statement that it is not related.
    - Example: "nanohub is not related because the paper does not reference nanoHUB or its tools/resources."


KEYWORDS:
- Extract 2–3 keywords from:
  - a Keywords section, OR
  - recurring major concepts within the text.
- Do not invent speculative keywords.

========================
OUTPUT REQUIREMENTS
========================
- Output ONLY the final JSON object, with no explanation or commentary.
- Include only fields for which you found new or improved information."""
    return prompt


class OpenWebUIWrapper:
    """
    Wrapper for OpenWebUI API (compatible with Purdue's GenAI service).
    """
    def __init__(self, model: str, temperature: float, api_key: str, api_url: str):
        self.model = model
        self.temperature = temperature
        self.api_url = api_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def chat(self, messages: list) -> str:
        """Send chat request and return response."""
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature
            }
        )

        result = response.json()

        if 'error' in result or 'detail' in result:
            raise Exception(f"API Error: {result}")

        return result['choices'][0]['message']['content']


def sanitize_to_ascii(text: str) -> str:
    """
    Sanitize text to ASCII by removing or replacing non-ASCII characters.
    
    Args:
        text: Input text that may contain non-ASCII characters
        
    Returns:
        ASCII-safe string
    """
    if not text:
        return text
    
    # Try to encode as ASCII, replacing non-ASCII characters
    try:
        # First normalize unicode characters (e.g., é -> e)
        import unicodedata
        normalized = unicodedata.normalize('NFKD', text)
        # Encode to ASCII, ignoring characters that can't be represented
        ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
        return ascii_text
    except Exception:
        # Fallback: just remove non-ASCII characters
        return ''.join(char for char in text if ord(char) < 128)


def map_publication_type_to_genre_id(publication_type: str) -> int:
    """
    Map publication type string to documentgenreid.

    Args:
        publication_type: Publication type string (e.g., "journal", "conference", etc.)

    Returns:
        Genre ID (1-20)
    """
    # Normalize to lowercase
    pub_type = publication_type.lower().strip()

    # Mapping based on the valid IDs
    type_mapping = {
        'journal': 2,
        'proceedings': 3,
        'inbook': 4,
        'phdthesis': 5,
        'phd thesis': 5,
        'mastersthesis': 6,
        'masters thesis': 6,
        'conference': 7,
        'techreport': 8,
        'tech report': 8,
        'magazine': 9,
        'article': 10,
        'preprint': 11,
        'xarchive': 12,
        'patent': 13,
        'notes': 14,
        'letter': 15,
        'syllabus': 16,
        'tutorial': 17,
        'arxiv': 18,
        'inproceedings': 19,
        'misc': 20,
        'book': 4,  # Map to inbook
        'thesis': 5,  # Default to PhD thesis
        'report': 8,  # Map to techreport
    }

    return type_mapping.get(pub_type, 1)  # Default to 1 (N/A) if not found


def get_genre_name_by_id(genre_id: int) -> str:
    """
    Map genre ID to name.

    Args:
        genre_id: Genre ID

    Returns:
        Genre name string
    """
    id_mapping = {
        1: "N/A",
        2: "Journal",
        3: "Proceedings",
        4: "Book",
        5: "PhD Thesis",
        6: "Masters Thesis",
        7: "Conference",
        8: "Tech Report",
        9: "Magazine",
        10: "Article",
        11: "Preprint",
        12: "XArchive",
        13: "Patent",
        14: "Notes",
        15: "Letter",
        16: "Syllabus",
        17: "Tutorial",
        18: "Arxiv",
        19: "InProceedings",
        20: "Misc"
    }
    return id_mapping.get(genre_id, "N/A")


def update_citation_with_metadata(citation: Citation, metadata: Dict[str, Any]) -> Citation:
    """
    Update citation object with extracted metadata.

    Args:
        citation: Citation object to update
        metadata: Extracted metadata dictionary

    Returns:
        Updated citation object
    """
    # Update basic fields
    if 'title' in metadata and metadata['title']:
        citation.title = metadata['title']

    # Extract and save abstract (IMPORTANT: this is the complete abstract text)
    if 'abstract' in metadata and metadata['abstract']:
        abstract_text = str(metadata['abstract']).strip()
        if abstract_text and len(abstract_text) > 10:  # Make sure it's not empty or too short
            citation.abstract = abstract_text
            print(f"  → Abstract extracted: {len(abstract_text)} characters")

    if 'year' in metadata and metadata['year']:
        citation.year = int(metadata['year'])

    if 'doi' in metadata and metadata['doi']:
        citation.doi = metadata['doi']

    if 'publisher' in metadata and metadata['publisher']:
        citation.publisher = metadata['publisher']

    # Handle publication/journal - search for matching publication and set ID
    publication_to_search = None
    if 'publication' in metadata and metadata['publication']:
        publication_to_search = metadata['publication']
    elif 'journal' in metadata and metadata['journal']:
        publication_to_search = metadata['journal']

    if publication_to_search:
        # Store the name
        citation.publication_name = publication_to_search

        # Search for matching publication in the database
        print(f"  → Searching for publication: {publication_to_search}")
        pub_match = search_publication(publication_to_search)
        if pub_match and 'id' in pub_match:
            citation.publication_id = int(pub_match['id'])
            print(f"  ✓ Set publication ID: {pub_match['id']}")
        else:
            print(f"  ℹ Publication not found in database, saving name only")

    if 'publication_type' in metadata and metadata['publication_type']:
        # Map publication_type to documentgenreid
        genre_id = map_publication_type_to_genre_id(metadata['publication_type'])
        citation.documentgenreid = genre_id
        citation.document_genre_name = get_genre_name_by_id(genre_id)
        # Note: ref_type is now constructed from classification flags, not publication_type

    # Construct ref_type from classification flags
    ref_types = []
    if metadata.get('is_research'): ref_types.append('R')
    if metadata.get('is_cyberinfrastructure'): ref_types.append('C')
    if metadata.get('is_education'): ref_types.append('E')
    if metadata.get('is_nano'): ref_types.append('N')
    
    if ref_types:
        citation.ref_type = ",".join(ref_types)
        print(f"  ✓ Set ref_type: {citation.ref_type}")

    if 'volume' in metadata and metadata['volume']:
        citation.volume = str(metadata['volume'])

    if 'issue' in metadata and metadata['issue']:
        citation.issue = str(metadata['issue'])

    if 'pages' in metadata and metadata['pages']:
        # Split page range
        pages = str(metadata['pages']).split('-')
        if len(pages) == 2:
            citation.begin_page = pages[0].strip()
            citation.end_page = pages[1].strip()

    if 'isbn' in metadata and metadata['isbn']:
        citation.isbn = metadata['isbn']

    if 'institution' in metadata and metadata['institution']:
        citation.institution = metadata['institution']

    if 'school' in metadata and metadata['school']:
        citation.school = metadata['school']
    
    if 'exp_data' in metadata:
        citation.exp_data = 1 if metadata['exp_data'] else 0
    
    if 'exp_list_exp_data' in metadata:
        citation.exp_list_exp_data = 1 if metadata['exp_list_exp_data'] else 0

    # Update notes (for nanoHUB relationship)
    # Note: The 'notes' field in the database is a string (JSON), not an object
    if 'notes' in metadata and metadata['notes']:
        if isinstance(metadata['notes'], dict):
            # Convert dict to JSON string
            citation.notes = json.dumps(metadata['notes'])
        elif isinstance(metadata['notes'], str):
            citation.notes = metadata['notes']

    if 'affiliated' in metadata:
        citation.affiliated = 1 if metadata['affiliated'] else 0

    # Update authors
    if 'authors' in metadata and metadata['authors']:
        # Clear existing authors and add new ones
        citation.authors = []
        
        # Track added authors to prevent duplicates within the same batch
        added_signatures = set()
        
        # Add new authors
        for author in metadata['authors']:
            lastname = author.get('lastname', '').lower().strip()
            firstname = author.get('firstname', '').lower().strip()
            first_letter = firstname[0] if firstname else ''
            signature = f"{lastname}_{first_letter}"
            
            # Skip if this author already added in this batch
            if signature in added_signatures:
                print(f"  ℹ Skipping duplicate author in batch: {author.get('firstname', '')} {author.get('lastname', '')}")
                continue
            
            # Build author dict with only non-empty values
            # Sanitize names to ASCII to avoid collation issues
            author_data = {
                'firstname': sanitize_to_ascii(author.get('firstname', '')),
                'lastname': sanitize_to_ascii(author.get('lastname', '')),
            }

            # Add optional fields only if they have values
            if author.get('email') and str(author.get('email')).strip():
                author_data['email'] = str(author.get('email')).strip()
            if author.get('orcid') and str(author.get('orcid')).strip():
                author_data['orcid'] = str(author.get('orcid')).strip()
            if author.get('scopusid') and str(author.get('scopusid')).strip():
                author_data['scopusid'] = str(author.get('scopusid')).strip()
            if author.get('researcherid') and str(author.get('researcherid')).strip():
                author_data['researcherid'] = str(author.get('researcherid')).strip()
            if author.get('gsid') and str(author.get('gsid')).strip():
                author_data['gsid'] = str(author.get('gsid')).strip()
            if author.get('researchgateid') and str(author.get('researchgateid')).strip():
                author_data['researchgateid'] = str(author.get('researchgateid')).strip()

            # Always include organization/department fields (required by updated PHP code)
            # Sanitize to ASCII to avoid collation issues
            author_data['organizationname'] = sanitize_to_ascii(str(author.get('organizationname', '')).strip()) if author.get('organizationname') else ''
            author_data['organizationtype'] = sanitize_to_ascii(str(author.get('organizationtype', '')).strip()) if author.get('organizationtype') else ''
            author_data['organizationdept'] = sanitize_to_ascii(str(author.get('organizationdept', '')).strip()) if author.get('organizationdept') else ''
            
            # Always include country and email fields
            author_data['countryresident'] = str(author.get('countryresident', '')).strip() if author.get('countryresident') else ''
            author_data['email'] = str(author.get('email', '')).strip() if author.get('email') else ''
            author_data['countrySHORT'] = str(author.get('countrySHORT', '')).strip() if author.get('countrySHORT') else ''
            author_data['countryLONG'] = str(author.get('countryLONG', '')).strip() if author.get('countryLONG') else ''
            
            # Map countrySHORT to countryresident for the database (if not already set)
            if author.get('countrySHORT') and str(author.get('countrySHORT')).strip():
                author_data['countryresident'] = str(author.get('countrySHORT')).strip()
                author_data['countrySHORT'] = str(author.get('countrySHORT')).strip()
            if author.get('countryLONG') and str(author.get('countryLONG')).strip():
                author_data['countryLONG'] = str(author.get('countryLONG')).strip()
            if author.get('organization') and int(author.get('organization', 0)) > 0:
                author_data['organization'] = int(author.get('organization'))
            if author.get('department') and int(author.get('department', 0)) > 0:
                author_data['department'] = int(author.get('department'))

            citation.add_author(**author_data)
            added_signatures.add(signature)  # Add to set to prevent duplicates within the same batch

    # Update keywords
    if 'keywords' in metadata and metadata['keywords']:
        citation.keywords = []  # Clear existing
        for keyword in metadata['keywords']:
            citation.add_keyword(keyword)

    return citation


def extract_metadata_with_openwebui(
    pdf_text: str,
    existing_citation: Citation,
    model: str,
    api_key: str,
    api_url: str = "https://genai.rcac.purdue.edu/api/chat/completions"
) -> Dict[str, Any]:
    """
    Use OpenWebUI API to extract metadata from PDF text.

    Args:
        pdf_text: Extracted text from PDF
        existing_citation: Current citation object
        model: Model name (e.g., "llama3.1:70b")
        api_key: API key for authentication
        api_url: API endpoint URL

    Returns:
        Dictionary of extracted metadata
    """
    prompt = create_extraction_prompt(pdf_text, existing_citation)

    print(f"Calling OpenWebUI (model: {model})...")
    wrapper = OpenWebUIWrapper(model, 0.1, api_key, api_url)
    content = wrapper.chat([{
        'role': 'user',
        'content': prompt
    }])

    response = parse_llm_response(content)
    return response


def parse_llm_response(content: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response.

    Args:
        content: LLM response content

    Returns:
        Dictionary of extracted metadata
    """

    try:
        # Try to find JSON in the response
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:

            json_str = content[start_idx:end_idx]
            print(f"Response: {json_str}")
            metadata = json.loads(json_str)
            print(f"Response2: {metadata}")
            return metadata
        else:
            print("No JSON found in LLM response")
            return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response: {content}")
        return {}


def extract_authors_with_llm(
    pdf_sections: Dict[str, str],
    existing_citation: Citation,
    model: str,
    api_key: str,
    api_url: str
) -> Optional[Dict[str, Any]]:
    """
    Extract author information using focused LLM call.

    Args:
        pdf_sections: Extracted PDF sections
        existing_citation: Current citation object
        model: Model name
        api_key: API key
        api_url: API endpoint

    Returns:
        Dictionary with 'authors' key or None if failed
    """
    print("  → Extracting authors...")
    prompt = create_author_extraction_prompt(pdf_sections, existing_citation)

    wrapper = OpenWebUIWrapper(model, 0.1, api_key, api_url)
    content = wrapper.chat([{'role': 'user', 'content': prompt}])

    result = parse_llm_response(content)
    if result and 'authors' in result:
        print(f"  ✓ Found {len(result['authors'])} authors")
    return result


def extract_paper_details_with_llm(
    pdf_sections: Dict[str, str],
    existing_citation: Citation,
    model: str,
    api_key: str,
    api_url: str
) -> Optional[Dict[str, Any]]:
    """
    Extract paper bibliographic details using focused LLM call.

    Args:
        pdf_sections: Extracted PDF sections
        existing_citation: Current citation object
        model: Model name
        api_key: API key
        api_url: API endpoint

    Returns:
        Dictionary with paper details or None if failed
    """
    print("  → Extracting paper details...")
    prompt = create_paper_details_prompt(pdf_sections, existing_citation)

    wrapper = OpenWebUIWrapper(model, 0.0, api_key, api_url)
    content = wrapper.chat([{'role': 'user', 'content': prompt}])

    result = parse_llm_response(content)
    if result:
        print(f"  ✓ Extracted {len(result)} fields")
    return result


def extract_experimental_flags_with_llm(
    pdf_sections: Dict[str, str],
    model: str,
    api_key: str,
    api_url: str
) -> Optional[Dict[str, Any]]:
    """
    Assess experimental data flags using focused LLM call.

    Args:
        pdf_sections: Extracted PDF sections
        model: Model name
        api_key: API key
        api_url: API endpoint

    Returns:
        Dictionary with 'expData' and 'expListExpData' or None if failed
    """
    print("  → Assessing experimental data...")
    prompt = create_experimental_data_prompt(pdf_sections)

    wrapper = OpenWebUIWrapper(model, 0.2, api_key, api_url)
    content = wrapper.chat([{'role': 'user', 'content': prompt}])

    result = parse_llm_response(content)
    if result:
        print(f"  ✓ expData: {result.get('expData')}, expListExpData: {result.get('expListExpData')}")
    return result


def extract_classification_with_llm(
    pdf_sections: Dict[str, str],
    model: str,
    api_key: str,
    api_url: str
) -> Optional[Dict[str, Any]]:
    """
    Classify paper type (R, C, E, N) using focused LLM call.

    Args:
        pdf_sections: Extracted PDF sections
        model: Model name
        api_key: API key
        api_url: API endpoint

    Returns:
        Dictionary with classification flags or None if failed
    """
    print("  → Classifying paper type (R, C, E, N)...")
    prompt = create_classification_prompt(pdf_sections)

    wrapper = OpenWebUIWrapper(model, 0.1, api_key, api_url)
    content = wrapper.chat([{'role': 'user', 'content': prompt}])

    result = parse_llm_response(content)
    if result:
        flags = []
        if result.get('is_research'): flags.append('R')
        if result.get('is_cyberinfrastructure'): flags.append('C')
        if result.get('is_education'): flags.append('E')
        if result.get('is_nano'): flags.append('N')
        print(f"  ✓ Classification: {', '.join(flags)}")
    return result


def extract_nanohub_relationship_with_llm(
    pdf_sections: Dict[str, str],
    model: str,
    api_key: str,
    api_url: str
) -> Optional[Dict[str, Any]]:
    """
    Assess nanoHUB relationship using chunked LLM analysis.

    Args:
        pdf_sections: Extracted PDF sections
        model: Model name
        api_key: API key
        api_url: API endpoint

    Returns:
        Dictionary with 'nanohubRelated' and 'nanohubNotes' or None if failed
    """
    print("  → Analyzing nanoHUB relationship...")

    # Pre-check: search for "nanohub" in full text (case-insensitive)
    full_text = pdf_sections['full']
    full_lower = full_text.lower()
    
    # Check for nanohub or NCN
    nanohub_found = 'nanohub' in full_lower or 'nano-hub' in full_lower or 'nano hub' in full_lower
    ncn_found = 'network for computational nanotechnology' in full_lower or ' ncn ' in full_lower or '(ncn)' in full_lower
    
    if nanohub_found or ncn_found:
        print("  ℹ Pre-check: Found 'nanohub' or 'NCN' in text - analyzing with LLM")
    else:
        print("  ℹ Pre-check: No 'nanohub' or 'NCN' found in text")
        return {
            "nanohubRelated": False,
            "nanohubNotes": "No evidence of nanoHUB relationship found",
            "affiliated": False
        }

    # Split text into chunks of ~4000 chars to ensure thorough analysis
    chunk_size = 4000
    chunks = []
    for i in range(0, len(full_text), chunk_size):
        chunks.append(full_text[i:i + chunk_size + 200])  # +200 for overlap

    print(f"  ℹ Searching {len(chunks)} chunks for nanoHUB/NCN references...")

    wrapper = OpenWebUIWrapper(model, 0.1, api_key, api_url)  # Lower temp for factual search
    all_evidence = []
    is_affiliated = False

    # Analyze each chunk
    for idx, chunk in enumerate(chunks, 1):
        # Only analyze chunks that contain "nanohub" or "ncn" to save API calls (case-insensitive)
        chunk_lower = chunk.lower()
        has_nanohub = 'nanohub' in chunk_lower or 'nano-hub' in chunk_lower or 'nano hub' in chunk_lower
        has_ncn = 'network for computational nanotechnology' in chunk_lower or ' ncn ' in chunk_lower or '(ncn)' in chunk_lower
        
        if not has_nanohub and not has_ncn:
            continue

        prompt = create_nanohub_relationship_prompt(chunk, idx, len(chunks))
        content = wrapper.chat([{'role': 'user', 'content': prompt}])
        result = parse_llm_response(content)

        if result and result.get('foundInChunk') and result.get('evidence'):
            evidence = result.get('evidence', '').strip()
            if evidence:
                all_evidence.append(f"[Chunk {idx}] {evidence}")
                print(f"  ✓ Found in chunk {idx}: {evidence[:80]}...")
            
            if result.get('isAffiliated'):
                is_affiliated = True
                print(f"  ✓ Found affiliation in chunk {idx}")

    # Compile final result
    if all_evidence:
        combined_notes = "Paper references nanoHUB/NCN: " + " | ".join(all_evidence[:3])  # Limit to 3 examples
        return {
            "nanohubRelated": True,
            "nanohubNotes": combined_notes,
            "affiliated": is_affiliated
        }
    else:
        return {
            "nanohubRelated": False,
            "nanohubNotes": "Text contains 'nanohub'/'NCN' but no clear references found upon detailed analysis",
            "affiliated": False
        }


def extract_metadata_multi_call(
    pdf_sections: Dict[str, str],
    existing_citation: Citation,
    model: str,
    api_key: str,
    api_url: str
) -> Dict[str, Any]:
    """
    Orchestrate multiple specialized LLM calls to extract complete metadata.

    Args:
        pdf_sections: Extracted PDF sections
        existing_citation: Current citation object
        model: Model name
        api_key: API key
        api_url: API endpoint

    Returns:
        Merged dictionary of all extracted metadata
    """
    print(f"\nRunning multi-call metadata extraction (model: {model})...")
    print("=" * 60)

    merged_metadata = {}

    # Call 0: Extract metadata from URL if available
    if existing_citation.url and existing_citation.url.strip():
        try:
            print(f"\n→ Extracting metadata from URL...")

            if existing_citation.doi != "":
                print(f"  ℹ Detected DOI URL, using CrossRef API: {doi_match}")
                metadata = extract_metadata_from_doi(existing_citation.doi)
                if metadata:
                    merged_metadata.update(metadata)
                            
            url_metadata = extract_metadata_from_url(
                existing_citation.title, model, api_key, api_url
            )

            if url_metadata:
                # Merge URL metadata (PDF extraction will override if conflicts)
                merged_metadata.update(url_metadata)
                print(f"  ✓ URL metadata added to extraction pipeline")
        except Exception as e:
            print(f"  ✗ URL metadata extraction failed: {e}")

    # Call 1: Extract authors from PDF
    try:
        authors_data = extract_authors_with_llm(
            pdf_sections, existing_citation, model, api_key, api_url
        )
        if authors_data and 'authors' in authors_data:
            # PDF authors override URL authors (PDF is more detailed)
            merged_metadata['authors'] = authors_data['authors']
        elif 'authors' not in merged_metadata:
            # No PDF authors found, keep URL authors if available
            print(f"  ℹ No authors extracted from PDF, using URL authors if available")
    except Exception as e:
        print(f"  ✗ Author extraction failed: {e}")

    # Call 2: Extract paper details
    try:
        paper_data = extract_paper_details_with_llm(
            pdf_sections, existing_citation, model, api_key, api_url
        )
        if paper_data:
            merged_metadata.update(paper_data)
    except Exception as e:
        print(f"  ✗ Paper details extraction failed: {e}")

    # Call 3: Assess experimental flags
    try:
        exp_data = extract_experimental_flags_with_llm(
            pdf_sections, model, api_key, api_url
        )
        if exp_data:
            if 'expData' in exp_data:
                merged_metadata['exp_data'] = exp_data['expData']
            if 'expListExpData' in exp_data:
                merged_metadata['exp_list_exp_data'] = exp_data['expListExpData']
    except Exception as e:
        print(f"  ✗ Experimental assessment failed: {e}")

    # Call 4: Assess nanoHUB relationship
    try:
        nanohub_data = extract_nanohub_relationship_with_llm(
            pdf_sections, model, api_key, api_url
        )
        if nanohub_data:
            # Store in notes field
            if 'notes' not in merged_metadata:
                merged_metadata['notes'] = {}
            if 'nanohubNotes' in nanohub_data:
                merged_metadata['notes']['nanohub'] = nanohub_data['nanohubNotes']
            
            # Store affiliated status
            if 'affiliated' in nanohub_data:
                merged_metadata['affiliated'] = nanohub_data['affiliated']
    except Exception as e:
        print(f"  ✗ nanoHUB assessment failed: {e}")

    # Call 5: Classify paper type (R, C, E, N)
    try:
        class_data = extract_classification_with_llm(
            pdf_sections, model, api_key, api_url
        )
        if class_data:
            merged_metadata.update(class_data)
    except Exception as e:
        print(f"  ✗ Classification failed: {e}")

    print("=" * 60)
    print(f"✓ Multi-call extraction complete. Extracted {len(merged_metadata)} top-level fields.")

    return merged_metadata
    
def main():
    """Main execution function."""
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python llm_metadata_extraction.py <citation_id> [backend] [model_name]")
        print("\nBackends:")
        print("\nExamples:")
        print("\nEnvironment variables:")
        print("  NANOHUB_TOKEN    - Required: Your NanoHub API token")
        print("  NANOHUB_URL      - Optional: Hub URL (default: https://nanohub.org/api)")
        print("  OPENWEBUI_KEY    - Required for openwebui: API key")
        print("  OPENWEBUI_URL    - Optional: API URL (default: https://genai.rcac.purdue.edu/api/chat/completions)")
        sys.exit(1)

    citation_id = int(sys.argv[1])
    backend = "openwebui"
    model_name = os.getenv("LLM_MODEL", "gpt-oss:120b")

    # Check environment variables
    hub_url = os.getenv("NANOHUB_URL", "https://nanohub.org/api")
    api_token = os.getenv("NANOHUB_TOKEN")

    if not api_token:
        print("Error: NANOHUB_TOKEN environment variable not set")
        print("Set it with: export NANOHUB_TOKEN='your-token-here'")
        print("Or create a .env file in the project root (see .env.example)")
        sys.exit(1)

    # Check OpenWebUI requirements
    openwebui_key = None
    openwebui_url = None
    if backend == "openwebui":
        openwebui_key = os.getenv("OPENWEBUI_KEY")
        openwebui_url = os.getenv("OPENWEBUI_URL", "https://genai.rcac.purdue.edu/api/chat/completions")
        if not openwebui_key:
            print("Error: OPENWEBUI_KEY environment variable not set")
            print("Set it with: export OPENWEBUI_KEY='your-api-key-here'")
            print("Or create a .env file in the project root (see .env.example)")
            sys.exit(1)

    print(f"Citation Manager LLM Metadata Extraction")
    print(f"=========================================")
    print(f"Citation ID: {citation_id}")
    print()

    # Create session and client
    print("Connecting to Citation Manager...")
    credentials = {
        "grant_type": "personal_token",
        "token": api_token
    }
    session = Session(credentials, url=hub_url, max_retries=1)
    client = CitationManagerClient(session)

    # Get citation
    print(f"\nFetching citation {citation_id}...")
    citation = client.get(citation_id)
    print(f"Current title: {citation.title}")
    print(f"Current authors: {len(citation.authors)} author(s)")
    print(f"Current keywords: {len(citation.keywords)} keyword(s)")

    # Download PDF
    pdf_path = f"citation_{citation_id}.pdf"
    #print(f"\nDownloading PDF...")
    try:
        client.download_pdf(citation_id, pdf_path)
        print(f"PDF saved to: {pdf_path}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        sys.exit(1)

    # Extract text and sections from PDF
    print("\nExtracting text and sections from PDF...")
    pdf_sections = extract_pdf_sections(pdf_path, max_pages=None)  # Extract all pages to ensure we get references
    if not pdf_sections['full']:
        print("Failed to extract text from PDF")
        sys.exit(1)
    print(f"Extracted {len(pdf_sections['full'])} characters from PDF")
    print(f"  - Intro/Abstract: {len(pdf_sections['intro'])} chars")
    print(f"  - Methods: {len(pdf_sections['methods'])} chars")
    print(f"  - Acknowledgments: {len(pdf_sections['acknowledgments'])} chars")
    print(f"  - References: {len(pdf_sections['references'])} chars")

    # Debug: Show first 200 chars of references to verify extraction
    if pdf_sections['references']:
        print(f"\n  References preview (first 200 chars):")
        print(f"  {pdf_sections['references'][:200]}")
    else:
        print(f"\n  WARNING: No references section found!")

    # Always check for "nanohub" in full text
    full_lower = pdf_sections['full'].lower()
    if 'nanohub' in full_lower:
        idx = full_lower.index('nanohub')
        print(f"\n  ✓ Found 'nanohub' at character position {idx} in full text:")
        print(f"  Context: ...{pdf_sections['full'][max(0, idx-50):idx+150]}...")
    else:
        print(f"\n  ✗ 'nanohub' not found anywhere in the extracted PDF text.")
        # Save extracted text to file for manual inspection
        debug_file = f"citation_{citation_id}_extracted.txt"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(pdf_sections['full'])
        print(f"  Saved extracted text to {debug_file} for manual inspection.")

    # Extract metadata with multi-call LLM approach
    metadata = extract_metadata_multi_call(
        pdf_sections, citation, model=model_name,
        api_key=openwebui_key, api_url=openwebui_url
    )

    if not metadata:
        print("Failed to extract metadata from LLM")
        sys.exit(1)

    print("\nExtracted metadata:")
    #print(json.dumps(metadata, indent=2))

    # Update citation
    print("\nUpdating citation with extracted metadata...")
    #print(json.dumps(metadata, indent=2))

    updated_citation = update_citation_with_metadata(citation, metadata)

    print(json.dumps(updated_citation.to_dict(), indent=2))


    # Ask for confirmation
    #response = input("\nSave changes to Citation Manager? (yes/no): ")
    #if response.lower() in ['yes', 'y']:
    #    print("\nSaving changes...")
    client.update(updated_citation)
    #    print("Citation updated successfully!")
    #else:
    #    print("Changes discarded.")

    # Check for and merge duplicate authors
    print("\n" + "=" * 60)
    print("PHASE 2: CHECK AND MERGE DUPLICATE AUTHORS")
    print("=" * 60)
    
    try:
        duplicates_merged = check_and_merge_duplicate_authors(
            client=client,
            citation=updated_citation
        )
        
        if duplicates_merged > 0:
            print(f"\n✓ Merged {duplicates_merged} duplicate author(s)")
            # Update the citation again after merging duplicates
            print("\n→ Updating citation after merging duplicates...")
            client.update(updated_citation)
            print("  ✓ Citation updated successfully")
        else:
            print(f"\nℹ No duplicate authors to merge")
    except Exception as e:
        print(f"\n✗ Error during duplicate author check: {e}")

    # Search for and add related nanohub resources
    print("\n" + "=" * 60)
    print("PHASE 3: SEARCH AND LINK NANOHUB RESOURCES")
    print("=" * 60)

    try:
        resources_added = find_and_add_nanohub_resources(
            client=client,
            citation=updated_citation,
            model=model_name,
            api_key=openwebui_key,
            api_url=openwebui_url,
            hub_url=hub_url.replace('/api', '')  # Remove /api suffix for search
        )

        if resources_added > 0:
            print(f"\n✓ Successfully added {resources_added} nanohub resource(s) to citation")
        else:
            print(f"\nℹ No matching nanohub resources found")
    except Exception as e:
        print(f"\n✗ Error during resource search: {e}")

    # Clean up
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        print(f"\nTemporary PDF file removed: {pdf_path}")


if __name__ == "__main__":
    main()

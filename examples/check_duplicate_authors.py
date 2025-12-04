#!/usr/bin/env python3
"""
Example: Check and Merge Duplicate Authors in a Citation

This example demonstrates how to:
1. Load a citation by ID
2. Check for duplicate authors
3. Automatically merge duplicates (keeping the oldest ID)

Requirements:
    - pip install nanohub-citmanager
    - Set NANOHUB_TOKEN environment variable
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from nanohubremote import Session
from nanohubcitmanager import CitationManagerClient, Citation
from typing import List


def find_duplicate_authors_in_citation(citation: Citation) -> List[tuple]:
    """
    Find duplicate authors within a citation based on name similarity.
    
    Args:
        citation: Citation object with authors
        
    Returns:
        List of tuples containing (author1_index, author2_index, similarity_score)
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
                        if firstname1 == firstname2:
                            similarity = 1.0
                        else:
                            similarity = 0.7
                        duplicates.append((i, j, similarity))
    
    return duplicates


def merge_duplicate_authors(
    client: CitationManagerClient,
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
        author_index_primary: Index of the author to keep
        author_index_duplicate: Index of the author to mark as duplicate
        
    Returns:
        True if successful
    """
    try:
        authors = citation.authors
        if author_index_primary >= len(authors) or author_index_duplicate >= len(authors):
            print(f"  ✗ Invalid author indices")
            return False
        
        # Get author info
        author_primary = authors[author_index_primary]
        author_duplicate = authors[author_index_duplicate]
        
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
            print(f"  → Removing duplicate from list: {name_duplicate}")
            citation.authors.pop(author_index_duplicate)
            return True
        
        # Determine which has the older (smaller) ID - that should be the primary
        if id_duplicate < id_primary:
            id_primary, id_duplicate = id_duplicate, id_primary
            name_primary, name_duplicate = name_duplicate, name_primary
            # Also swap indices
            author_index_primary, author_index_duplicate = author_index_duplicate, author_index_primary
        
        print(f"  → Merging: keeping '{name_primary}' (ID: {id_primary}), marking '{name_duplicate}' (ID: {id_duplicate}) as duplicate")
        
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
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python check_duplicate_authors.py <citation_id>")
        print("\nEnvironment variables:")
        print("  NANOHUB_TOKEN    - Required: Your NanoHub API token")
        print("  NANOHUB_URL      - Optional: Hub URL (default: https://nanohub.org/api)")
        sys.exit(1)

    citation_id = int(sys.argv[1])

    # Check environment variables
    hub_url = os.getenv("NANOHUB_URL", "https://nanohub.org/api")
    api_token = os.getenv("NANOHUB_TOKEN")

    if not api_token:
        print("Error: NANOHUB_TOKEN environment variable not set")
        sys.exit(1)

    print(f"Citation Manager - Duplicate Author Check")
    print(f"==========================================")
    print(f"Citation ID: {citation_id}\n")

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
    print(f"Title: {citation.title}")
    print(f"Authors: {len(citation.authors)} author(s)\n")

    # List current authors
    print("Current authors:")
    for i, author in enumerate(citation.authors):
        if isinstance(author, dict):
            name = f"{author.get('firstname', '')} {author.get('lastname', '')}"
            author_id = author.get('id', 'N/A')
        else:
            name = f"{getattr(author, 'firstname', '')} {getattr(author, 'lastname', '')}"
            author_id = getattr(author, 'id', 'N/A')
        print(f"  [{i}] {name} (ID: {author_id})")

    # Check for duplicates
    print("\n" + "=" * 60)
    print("CHECKING FOR DUPLICATE AUTHORS")
    print("=" * 60)
    
    duplicates = find_duplicate_authors_in_citation(citation)
    
    if not duplicates:
        print("\n✓ No duplicate authors found")
        return
    
    print(f"\n→ Found {len(duplicates)} potential duplicate(s)\n")
    
    # Process duplicates
    merged_count = 0
    duplicates_sorted = sorted(duplicates, key=lambda x: x[1], reverse=True)
    
    for idx, (i, j, similarity) in enumerate(duplicates_sorted, 1):
        authors = citation.authors
        
        if isinstance(authors[i], dict):
            name1 = f"{authors[i].get('firstname', '')} {authors[i].get('lastname', '')}"
            name2 = f"{authors[j].get('firstname', '')} {authors[j].get('lastname', '')}"
        else:
            name1 = f"{getattr(authors[i], 'firstname', '')} {getattr(authors[i], 'lastname', '')}"
            name2 = f"{getattr(authors[j], 'firstname', '')} {getattr(authors[j], 'lastname', '')}"
        
        print(f"[{idx}/{len(duplicates)}] Duplicate detected (similarity: {similarity:.2f}):")
        print(f"  Author {i}: {name1}")
        print(f"  Author {j}: {name2}")
        
        if merge_duplicate_authors(client, citation, i, j):
            merged_count += 1
            print(f"  ✓ Merged successfully\n")
        else:
            print(f"  ✗ Merge failed\n")
    
    print("=" * 60)
    print(f"COMPLETE: {merged_count} author(s) merged")
    print("=" * 60)
    
    # Update citation if any duplicates were merged
    if merged_count > 0:
        print("\n→ Updating citation...")
        client.update(citation)
        print("✓ Citation updated successfully")
        
        # Show final author list
        print("\nFinal author list:")
        for i, author in enumerate(citation.authors):
            if isinstance(author, dict):
                name = f"{author.get('firstname', '')} {author.get('lastname', '')}"
            else:
                name = f"{getattr(author, 'firstname', '')} {getattr(author, 'lastname', '')}"
            print(f"  [{i}] {name}")


if __name__ == "__main__":
    main()

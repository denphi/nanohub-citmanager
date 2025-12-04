#  Copyright 2025 HUBzero Foundation, LLC.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""
Citation Manager API Client based on nanohubremote.
"""

from nanohubremote.session import Session
from .citation import Citation
from .pdf import PDFManager
from typing import List, Dict, Optional, Any
import json


class CitationManagerClient:
    """
    Client for HubZero Citation Manager API.

    This client provides methods for managing citations/documents including
    CRUD operations and PDF file management.

    Attributes:
        session: NanoHub session object
        api_base: Base URL for Citation Manager API endpoints
        pdf_manager: PDFManager instance for handling PDF operations
    """

    def __init__(self, session: Session, api_base: str = "/citmanager/document"):
        """
        Initialize the Citation Manager client.

        Args:
            session: Authenticated nanohubremote Session object
            api_base: Base API path (default: /api/citmanager/document)
        """
        self.session = session
        self.api_base = api_base
        self.pdf_manager = PDFManager(self)
        self._request_id = 0

    def _get_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_id += 1
        return f"request-{self._request_id}"

    def _api_call(self, endpoint: str, params: Dict[str, Any], base_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Make an API call to the Citation Manager.

        Args:
            endpoint: API endpoint (e.g., 'CitationCRUD', 'PDFManager')
            params: Request parameters
            base_url: Optional base URL to override default (e.g. for different controllers)

        Returns:
            API response dictionary

        Raises:
            Exception: If API returns an error
        """
        base = base_url if base_url else self.api_base
        url = f"{base}/{endpoint}"
        payload = {
            "id": self._get_request_id(),
            "params": params
        }

        response = self.session.requestPost(url, json=payload)

        payload_str = json.dumps(payload)
        headers = self.session.headers
        #print(f"curl -X POST {url} " + " ".join([f"-H '{k}: {v}'" for k, v in headers.items()]) + f" -H 'Content-Type: application/json' -d '{payload_str}'")

        #print(response.text)
        response.raise_for_status()

        result = response.json()

        if result.get("result", {}).get("status") == "ERROR":
            raise Exception(f"API Error: {result['result'].get('message')}")

        return result.get("result", {})

    def create(self, citation: Citation) -> int:
        """
        Create a new citation.

        Args:
            citation: Citation object with data to create

        Returns:
            ID of the created citation

        Example:
            >>> citation = Citation(title="My Paper", year=2024)
            >>> citation.add_author("John", "Doe")
            >>> citation_id = client.create(citation)
        """
        params = citation.to_dict()
        params["action"] = "create"

        result = self._api_call("CitationCRUD", params)
        citation.id = result.get("documentId")
        return citation.id

    def get(self, citation_id: int) -> Citation:
        """
        Retrieve a citation by ID.

        Args:
            citation_id: ID of the citation to retrieve

        Returns:
            Citation object populated with data from the API

        Example:
            >>> citation = client.get(123)
            >>> print(citation.title)
        """
        params = {
            "action": "read",
            "idDocument": citation_id
        }

        result = self._api_call("CitationCRUD", params)
        return Citation(result.get("document", {}))

    def update(self, citation: Citation) -> bool:
        """
        Update an existing citation.

        Args:
            citation: Citation object with updated data (must have id set)

        Returns:
            True if successful

        Raises:
            ValueError: If citation has no ID

        Example:
            >>> citation = client.get(123)
            >>> citation.title = "Updated Title"
            >>> client.update(citation)
        """
        if not citation.id:
            raise ValueError("Citation must have an ID to update")

        params = citation.to_dict()
        params["action"] = "update"
        params["idDocument"] = citation.id

        self._api_call("CitationCRUD", params)
        return True

    def delete(self, citation_id: int) -> bool:
        """
        Delete a citation (soft delete - sets status to JUNK).

        Args:
            citation_id: ID of the citation to delete

        Returns:
            True if successful

        Example:
            >>> client.delete(123)
        """
        params = {
            "action": "delete",
            "idDocument": citation_id
        }

        self._api_call("CitationCRUD", params)
        return True

    def list(self,
             search: Optional[str] = None,
             status: Optional[int] = None,
             limit: int = 50,
             offset: int = 0) -> List[Dict[str, Any]]:
        """
        List citations with optional filtering.

        Args:
            search: Search term for title/abstract
            status: Filter by workflow status
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of citation dictionaries

        Example:
            >>> citations = client.list(search="machine learning", limit=10)
            >>> for cit in citations:
            ...     print(cit['title'])
        """
        params = {
            "action": "list",
            "limit": limit,
            "offset": offset
        }

        if search:
            params["search"] = search
        if status is not None:
            params["status"] = status

        result = self._api_call("CitationCRUD", params)
        return result.get("documents", [])

    def search(self, query: str, limit: int = 50) -> List[Citation]:
        """
        Search for citations.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of Citation objects

        Example:
            >>> results = client.search("deep learning", limit=20)
        """
        documents = self.list(search=query, limit=limit)
        return [Citation(doc) for doc in documents]

    def download_pdf(self, citation_id: int, output_path: str) -> bool:
        """
        Download PDF file for a citation.

        Args:
            citation_id: ID of the citation
            output_path: Path to save the PDF file

        Returns:
            True if successful

        Example:
            >>> client.download_pdf(123, "paper.pdf")
        """
        return self.pdf_manager.download(citation_id, output_path)

    def upload_pdf(self, citation_id: int, pdf_path: str) -> bool:
        """
        Upload PDF file for a citation.

        Args:
            citation_id: ID of the citation
            pdf_path: Path to the PDF file to upload

        Returns:
            True if successful

        Example:
            >>> client.upload_pdf(123, "my_paper.pdf")
        """
        return self.pdf_manager.upload(citation_id, pdf_path)

    def get_pdf_info(self, citation_id: int) -> Dict[str, Any]:
        """
        Get information about a citation's PDF file.

        Args:
            citation_id: ID of the citation

        Returns:
            Dictionary with PDF metadata (filename, size, path, downloadUrl)

        Example:
            >>> info = client.get_pdf_info(123)
            >>> print(f"PDF: {info['filename']}, Size: {info['size']} bytes")
        """
        return self.pdf_manager.get_info(citation_id)

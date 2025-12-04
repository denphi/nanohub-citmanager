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
PDF file management for Citation Manager.
"""

import base64
import os
from typing import Dict, Any, Optional


class PDFManager:
    """
    Manager for PDF file operations (upload, download, delete).
    """

    def __init__(self, client):
        """
        Initialize PDF Manager.

        Args:
            client: CitationManagerClient instance
        """
        self.client = client

    def download(self, citation_id: int, output_path: str) -> bool:
        """
        Download PDF file for a citation using direct streaming.

        Args:
            citation_id: ID of the citation
            output_path: Path to save the PDF file

        Returns:
            True if successful

        Raises:
            Exception: If download fails
        """
        # Use base64=True to trigger file streaming (via Content Server)
        url = self.client.session.getUrl(f"{self.client.api_base}/PDFManager")
        payload = {
            "id": self.client._get_request_id(),
            "params": {
                "action": "get",
                "idDocument": citation_id,
                "base64": True  # Triggers Content Server streaming
            }
        }

        # Use the underlying session for streaming support
        response = self.client.session._session.post(url, json=payload, stream=True)
        response.raise_for_status()

        # Write streamed content to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return True

    def upload(self, citation_id: int, pdf_path: str, filename: Optional[str] = None) -> bool:
        """
        Upload PDF file for a citation.

        Args:
            citation_id: ID of the citation
            pdf_path: Path to the PDF file to upload
            filename: Optional custom filename

        Returns:
            True if successful

        Raises:
            Exception: If upload fails or file doesn't exist
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Read and encode PDF
        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")

        if not filename:
            filename = os.path.basename(pdf_path)

        params = {
            "action": "upload",
            "idDocument": citation_id,
            "filename": filename,
            "data": pdf_data
        }

        self.client._api_call("PDFManager", params)
        return True

    def get_info(self, citation_id: int) -> Dict[str, Any]:
        """
        Get information about a citation's PDF file.

        Args:
            citation_id: ID of the citation

        Returns:
            Dictionary with PDF metadata

        Raises:
            Exception: If retrieval fails
        """
        params = {
            "action": "get",
            "idDocument": citation_id,
            "base64": False
        }

        return self.client._api_call("PDFManager", params)

    def delete(self, citation_id: int) -> bool:
        """
        Delete PDF file associated with a citation.

        Args:
            citation_id: ID of the citation

        Returns:
            True if successful

        Raises:
            Exception: If deletion fails
        """
        params = {
            "action": "delete",
            "idDocument": citation_id
        }

        self.client._api_call("PDFManager", params)
        return True

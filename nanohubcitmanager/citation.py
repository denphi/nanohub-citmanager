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
Citation class representing a document/citation in the Citation Manager.
"""

from typing import List, Dict, Any, Optional


class Citation:
    """
    Represents a citation/document with all its metadata.

    This class supports all fields from the DocumentExp model including:
    - Core document fields (title, abstract, etc.)
    - BibTeX fields
    - Paper/journal fields
    - NanoHub-specific fields
    - Citation metrics
    - Authors and keywords
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """
        Initialize a Citation object.

        Args:
            data: Dictionary containing citation data from API response
        """
        # Core document fields
        self.id: Optional[int] = None
        self.title: str = ""
        self.abstract: str = ""
        self.publication_id: Optional[int] = None
        self.publication_name: str = ""
        self.document_genre_id: Optional[int] = None
        self.document_genre_name: str = ""
        self.publication_date: Optional[str] = None
        self.full_text_path: str = ""
        self.timestamp: Optional[str] = None

        # BibTeX fields
        self.address: str = ""
        self.booktitle: str = ""
        self.chapter: str = ""
        self.edition: str = ""
        self.editor: str = ""
        self.eprint: str = ""
        self.howpublished: str = ""
        self.institution: str = ""
        self.key: str = ""
        self.month: str = ""
        self.note: str = ""
        self.organization: str = ""
        self.publisher: str = ""
        self.series: str = ""
        self.school: str = ""
        self.type: str = ""

        # Paper/journal fields
        self.volume: str = ""
        self.issue: str = ""
        self.begin_page: str = ""
        self.end_page: str = ""

        # NanoHub fields
        self.url: str = ""
        self.year: Optional[int] = None
        self.isbn: str = ""
        self.cite: str = ""
        self.affiliated: int = -1
        self.fundedby: int = -1
        self.created: Optional[str] = None
        self.doi: str = ""
        self.ref_type: str = ""
        self.status: int = 0
        self.date_submit: Optional[str] = None
        self.date_accept: Optional[str] = None
        self.date_publish: Optional[str] = None
        self.software_use: int = -1
        self.res_edu: int = -1
        self.exp_list_exp_data: int = -1
        self.exp_data: str = ""
        self.notes: str = ""

        # Citation metrics
        self.url_citations: str = ""
        self.url_search: str = ""
        self.cnt_citations: int = 0
        self.date_citations: Optional[str] = None

        # Related data
        self.authors: List[Dict[str, Any]] = []
        self.keywords: List[str] = []

        # Load data if provided
        if data:
            self.from_dict(data)

    def from_dict(self, data: Dict[str, Any]) -> 'Citation':
        """
        Populate citation from dictionary (API response).

        Args:
            data: Dictionary containing citation data

        Returns:
            Self for method chaining
        """

        # Core fields
        self.id = data.get('id')
        self.title = data.get('title', '')
        self.abstract = data.get('abstract', '')
        self.publication_id = data.get('publication_id')
        self.publication_name = data.get('publication_name', '')
        self.document_genre_id = data.get('document_genre_id')
        self.document_genre_name = data.get('document_genre_name', '')
        self.publication_date = data.get('publication_date')
        self.full_text_path = data.get('full_text_path', '')
        self.timestamp = data.get('timestamp')

        # BibTeX fields
        self.address = data.get('address', '')
        self.booktitle = data.get('booktitle', '')
        self.chapter = data.get('chapter', '')
        self.edition = data.get('edition', '')
        self.editor = data.get('editor', '')
        self.eprint = data.get('eprint', '')
        self.howpublished = data.get('howpublished', '')
        self.institution = data.get('institution', '')
        self.key = data.get('key', '')
        self.month = data.get('month', '')
        self.note = data.get('note', '')
        self.organization = data.get('organization', '')
        self.publisher = data.get('publisher', '')
        self.series = data.get('series', '')
        self.school = data.get('school', '')
        self.type = data.get('type', '')

        # Paper fields
        self.volume = data.get('volume', '')
        self.issue = data.get('issue', '')
        self.begin_page = data.get('begin_page', '')
        self.end_page = data.get('end_page', '')

        # NanoHub fields
        self.url = data.get('url', '')
        self.year = data.get('year')
        self.isbn = data.get('isbn', '')
        self.cite = data.get('cite', '')
        self.affiliated = data.get('affiliated', -1)
        self.fundedby = data.get('fundedby', -1)
        self.created = data.get('created')
        self.doi = data.get('doi', '')
        self.ref_type = data.get('ref_type', '')
        self.status = data.get('status', 0)
        self.date_submit = data.get('date_submit')
        self.date_accept = data.get('date_accept')
        self.date_publish = data.get('date_publish')
        self.software_use = data.get('software_use', -1)
        self.res_edu = data.get('res_edu', -1)
        self.exp_list_exp_data = 1 if data.get('exp_list_exp_data', False) else 0
        self.exp_data = 1 if data.get('exp_data', False) else 0
        self.notes = data.get('notes', '')

        # Citation metrics
        self.url_citations = data.get('url_citations', '')
        self.url_search = data.get('url_search', '')
        self.cnt_citations = data.get('cnt_citations', 0)
        self.date_citations = data.get('date_citations')

        # Related data
        self.authors = data.get('authors', [])
        self.keywords = data.get('keywords', [])

        return self

    def _normalize_authors_for_api(self, authors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize author dictionaries to use PascalCase keys expected by PHP API.

        Args:
            authors: List of author dictionaries

        Returns:
            List of normalized author dictionaries
        """
        normalized = []
        for author in authors:
            normalized_author = {}
            for key, value in author.items():
                # Map all variations to camelCase expected by PersonExp::setByMap
                # PersonExp expects: firstName, lastName, middleInitial (camelCase)
                key_mapping = {
                    'firstname': 'firstName',  # Convert to camelCase
                    'firstName': 'firstName',  # Keep as-is
                    'middleinitial': 'middleInitial',
                    'middleInitial': 'middleInitial',
                    'lastname': 'lastName',  # Convert to camelCase
                    'lastName': 'lastName',  # Keep as-is
                    'photourl': 'photoURL',
                    'photoURL': 'photoURL',  # Keep as-is
                    'affiliationid': 'affiliationID',
                    'affiliationID': 'affiliationID',  # Keep as-is
                    'dateupdated': 'dateUpdated',
                    'dateUpdated': 'dateUpdated',  # Keep as-is
                    'aliasid': 'aliasID',
                    'aliasID': 'aliasID',  # Keep as-is
                    'orcid': 'ORCID',
                    'ORCID': 'ORCID',  # Keep as-is
                    'researcherid': 'WoSID',
                    'WoSID': 'WoSID',  # Keep as-is
                    'gsid': 'GSID',
                    'GSID': 'GSID',  # Keep as-is
                    'scopusid': 'SCOPUSID',
                    'scopusId': 'SCOPUSID',  # Handle camelCase variant
                    'SCOPUSID': 'SCOPUSID',  # Keep as-is
                    'researchgateid': 'RESEARCHGATEID',
                    'researchgateId': 'RESEARCHGATEID',  # Handle camelCase variant
                    'researchGateId': 'RESEARCHGATEID',  # Handle camelCase variant
                    'RESEARCHGATEID': 'RESEARCHGATEID',  # Keep as-is
                    'notes': 'NOTES',
                    'NOTES': 'NOTES',  # Keep as-is
                    # Keep these lowercase as they're used in PHP directly
                    'cid': 'cid',
                    'countryresident': 'countryresident',
                    'ip': 'ip',
                    'host': 'host',
                    'countrySHORT': 'countrySHORT',
                    'countryLONG': 'countryLONG',
                    'ipREGION': 'ipREGION',
                    'ipCITY': 'ipCITY',
                    'in_network': 'in_network',
                    'organization': 'organization',
                    'department': 'department',
                    'organizationname': 'organizationname',
                    'organizationName': 'organizationname',  # Handle PascalCase variant
                    'organization_name': 'organizationname',  # Handle snake_case from PHP serialization
                    'organizationtype': 'organizationtype',
                    'organizationType': 'organizationtype',  # Handle PascalCase variant
                    'organization_type': 'organizationtype',  # Handle snake_case from PHP serialization
                    'organizationdept': 'organizationdept',
                    'department_name': 'organizationdept',  # Handle snake_case from PHP serialization
                    # Common variations
                    'phone': 'phone',
                    'email': 'email',
                    'personId': 'personId',
                }

                # Use mapped key if exists, otherwise keep original
                mapped_key = key_mapping.get(key, key)
                normalized_author[mapped_key] = value

            normalized.append(normalized_author)

        return normalized

    def to_dict(self, include_none: bool = False) -> Dict[str, Any]:
        """
        Convert citation to dictionary for API requests.

        Args:
            include_none: Whether to include None/empty values

        Returns:
            Dictionary representation of citation
        """
        data = {
            'title': self.title,
            'abstract': self.abstract,
            # Match DocumentExp::setByMap() field names exactly
            'publicationID': self.publication_id,  # Line 602: publicationID (not publicationId)
            'publicationName': self.publication_name,  # Handled by CitationCRUD before setByMap
            'documentGenreID': self.document_genre_id,  # Line 603: documentGenreID
            'documentGenreName': self.document_genre_name,  # Handled by CitationCRUD
            'publicationDate': self.publication_date,  # Line 604: publicationDate
            'fulltextPath': self.full_text_path,  # Line 606: fulltextPath

            # BibTeX fields (lines 608-624: all lowercase)
            'address': self.address,
            'booktitle': self.booktitle,
            'chapter': self.chapter,
            'edition': self.edition,
            'editor': self.editor,
            'eprint': self.eprint,
            'howpublished': self.howpublished,
            'institution': self.institution,
            'key': self.key,
            'month': self.month,
            'note': self.note,
            'organization': self.organization,
            'publisher': self.publisher,
            'series': self.series,
            'school': self.school,
            'type': self.type,

            # Paper fields (lines 626-629: all lowercase)
            'volume': self.volume,
            'issue': self.issue,
            'beginpage': self.begin_page,  # Line 628: beginpage (not beginPage)
            'endpage': self.end_page,  # Line 629: endpage (not endPage)

            # NanoHub fields (lines 636-653: snake_case)
            'url': self.url,
            'year': self.year,
            'isbn': self.isbn,
            'cite': self.cite,
            'affiliated': self.affiliated,
            'fundedby': self.fundedby,
            'doi': self.doi,
            'ref_type': self.ref_type,
            'status': self.status,
            'date_submit': self.date_submit,
            'date_accept': self.date_accept,
            'date_publish': self.date_publish,
            'software_use': self.software_use,
            'res_edu': self.res_edu,
            'exp_list_exp_data': self.exp_list_exp_data,
            'exp_data': self.exp_data,
            'notes': self.notes,

            # Related data - normalize author keys to PascalCase for PHP API
            'authors': self._normalize_authors_for_api(self.authors),
            'keywords': self.keywords,
        }

        if not include_none:
            # Remove None and empty values
            # Also remove 0 for ID fields to prevent foreign key constraint violations
            # But keep 0 for boolean-like fields (exp_data, exp_list_exp_data, affiliated)
            id_fields = {'publicationID', 'documentGenreID'}  # Use correct field names
            boolean_fields = {'exp_data', 'exp_list_exp_data', 'affiliated'}  # Keep 0 values for these
            data = {k: v for k, v in data.items()
                   if (v is not None and v != '' and v != [] and v != -1)
                   or (k in boolean_fields and v == 0)  # Keep 0 for boolean fields
                   and not (k in id_fields and v == 0)}

        return data

    def __repr__(self) -> str:
        """String representation of citation."""
        return f"Citation(id={self.id}, title='{self.title[:50]}...', year={self.year})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        authors_str = ", ".join([f"{a.get('firstname', '')} {a.get('lastname', '')}"
                                for a in self.authors])
        return f"{authors_str} ({self.year}). {self.title}. {self.publication_name}"

    def add_author(self, firstname: str, lastname: str, **kwargs) -> None:
        """
        Add an author to the citation.

        Args:
            firstname: Author's first name
            lastname: Author's last name
            **kwargs: Additional author fields including:
                - Basic: middleinitial, phone, email, photoUrl
                - IDs: personId, cid, affiliationId, aliasId
                - Researcher IDs: orcid, researcherId, gsId, scopusId, researchGateId
                - Location: countryResident, ip, host, countrySHORT, countryLONG,
                           ipREGION, ipCITY, ipLATITUDE, ipLONGITUDE
                - Affiliation: organization, organizationName, organizationType,
                              department, departmentName
                - Other: dateUpdated, inNetwork, notes

        Example:
            citation.add_author(
                "John", "Doe",
                email="john@example.com",
                orcid="0000-0001-2345-6789",
                organization=123,
                organizationName="Purdue University",
                department=456,
                departmentName="Computer Science"
            )
        """
        # Normalize field names to match PHP API expectations
        # Map common variations to the expected format
        normalized_kwargs = {}
        for key, value in kwargs.items():
            # Map departmentname -> organizationdept for PHP compatibility
            if key == 'departmentname':
                normalized_kwargs['organizationdept'] = value
            # Map departmentName -> organizationdept for PHP compatibility
            elif key == 'departmentName':
                normalized_kwargs['organizationdept'] = value
            else:
                normalized_kwargs[key] = value

        author = {
            'firstname': firstname,
            'lastname': lastname,
            **normalized_kwargs
        }
        self.authors.append(author)

    def add_keyword(self, keyword: str) -> None:
        """
        Add a keyword to the citation.

        Args:
            keyword: Keyword to add
        """
        if keyword not in self.keywords:
            self.keywords.append(keyword)

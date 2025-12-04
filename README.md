# NanoHub Citation Manager

A Python library for interacting with the HubZero Citation Manager API, built on top of [nanohubremote](https://github.com/denphi/nanohub-remote).

## Features

- **Full CRUD Operations**: Create, read, update, and delete citations
- **PDF Management**: Upload, download, and manage PDF files associated with citations
- **Complete Metadata Support**: All DocumentExp fields including BibTeX, NanoHub-specific fields, and citation metrics
- **LLM Integration**: Example showing automated metadata extraction using local LLMs (Ollama)
- **Type-safe**: Full type hints for better IDE support
- **Built on nanohubremote**: Leverages the robust NanoHub API client

## Installation

```bash
pip install nanohub-citmanager
```

For LLM features:
```bash
pip install nanohub-citmanager[llm]
```

For development:
```bash
pip install nanohub-citmanager[dev]
```

## Quick Start

### Basic Usage

```python
from nanohubremote import Session
from nanohubcitmanager import CitationManagerClient, Citation

# Create session
credentials = {
    "grant_type": "personal_token",
    "token": "your-api-token"
}
session = Session(credentials)

# Create client
client = CitationManagerClient(session)

# Get a citation
citation = client.get(123)
print(f"Title: {citation.title}")
print(f"Authors: {len(citation.authors)}")
print(f"Keywords: {', '.join(citation.keywords)}")

# Update citation
citation.abstract = "Updated abstract text..."
citation.add_keyword("machine learning")
client.update(citation)
```

### Creating a New Citation

```python
# Create new citation
citation = Citation()
citation.title = "My Research Paper"
citation.abstract = "This paper presents..."
citation.year = 2024
citation.doi = "10.1234/example"

# Add authors
citation.add_author("John", "Doe", email="john@example.com")
citation.add_author("Jane", "Smith", orcid="0000-0001-2345-6789")

# Add keywords
citation.add_keyword("deep learning")
citation.add_keyword("neural networks")

# Create in Citation Manager
citation_id = client.create(citation)
print(f"Created citation ID: {citation_id}")
```

### PDF Management

```python
# Upload PDF
client.upload_pdf(citation_id, "paper.pdf")

# Download PDF
client.download_pdf(citation_id, "downloaded_paper.pdf")

# Get PDF info
info = client.get_pdf_info(citation_id)
print(f"PDF: {info['filename']}, Size: {info['size']} bytes")

# Delete PDF
client.pdf_manager.delete(citation_id)
```

### Searching Citations

```python
# Search by text
results = client.search("machine learning", limit=20)
for citation in results:
    print(f"{citation.year}: {citation.title}")

# List with filtering
documents = client.list(
    search="deep learning",
    status=100,  # Published
    limit=50,
    offset=0
)
```

## LLM-Powered Metadata Extraction

The library includes an example showing how to use a local LLM (via Ollama) to automatically extract and complete citation metadata from PDF files.

### Prerequisites

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3

# Install additional dependencies
pip install pypdf2
```

### Running the Example

```bash
# Set your API token
export NANOHUB_TOKEN="your-api-token"

# Run the extraction
python examples/llm_metadata_extraction.py 123 llama3
```

The script will:
1. Load the citation from the Citation Manager
2. Download the associated PDF
3. Extract text from the PDF
4. Use the LLM to extract metadata (title, authors, abstract, keywords, etc.)
5. Show you the extracted data and ask for confirmation before updating

### Example Output

```
Citation Manager LLM Metadata Extraction
=========================================
Citation ID: 123
LLM Model: llama3

Fetching citation 123...
Current title: Incomplete Title
Current authors: 0 author(s)
Current keywords: 0 keyword(s)

Downloading PDF...
PDF saved to: citation_123.pdf

Extracting text from PDF...
Extracted 12543 characters from PDF

Calling LLM (model: llama3)...

Extracted metadata:
{
  "title": "Deep Learning for Scientific Computing: A Survey",
  "abstract": "This paper presents a comprehensive survey of deep learning...",
  "authors": [
    {"firstname": "John", "lastname": "Doe"},
    {"firstname": "Jane", "lastname": "Smith"}
  ],
  "year": 2024,
  "doi": "10.1234/dl.2024.001",
  "journal": "Journal of Machine Learning Research",
  "keywords": ["deep learning", "scientific computing", "neural networks"]
}

Save changes to Citation Manager? (yes/no): yes
Citation updated successfully!
```

## API Reference

### CitationManagerClient

Main client for interacting with the Citation Manager API.

**Methods:**

- `create(citation: Citation) -> int`: Create a new citation
- `get(citation_id: int) -> Citation`: Retrieve a citation
- `update(citation: Citation) -> bool`: Update a citation
- `delete(citation_id: int) -> bool`: Delete a citation
- `list(search, status, limit, offset) -> List[Dict]`: List citations
- `search(query: str, limit: int) -> List[Citation]`: Search citations
- `download_pdf(citation_id, output_path) -> bool`: Download PDF
- `upload_pdf(citation_id, pdf_path) -> bool`: Upload PDF
- `get_pdf_info(citation_id) -> Dict`: Get PDF metadata

### Citation

Represents a citation/document with all metadata fields.

**Core Fields:**
- `id`, `title`, `abstract`
- `year`, `doi`, `isbn`, `url`
- `publisher`, `publication_name`
- `volume`, `issue`, `begin_page`, `end_page`

**BibTeX Fields:**
- `address`, `booktitle`, `chapter`, `edition`
- `editor`, `institution`, `school`
- `note`, `organization`, `series`

**NanoHub Fields:**
- `status`, `affiliated`, `fundedby`
- `software_use`, `res_edu`
- `date_submit`, `date_accept`, `date_publish`

**Citation Metrics:**
- `cnt_citations`, `url_citations`
- `date_citations`

**Related Data:**
- `authors`: List of author dictionaries
- `keywords`: List of keywords

**Methods:**
- `add_author(firstname, lastname, **kwargs)`: Add an author
- `add_keyword(keyword)`: Add a keyword
- `to_dict()`: Convert to dictionary for API
- `from_dict(data)`: Load from API response

### PDFManager

Handles PDF file operations.

**Methods:**
- `download(citation_id, output_path) -> bool`: Download PDF
- `upload(citation_id, pdf_path, filename) -> bool`: Upload PDF
- `get_info(citation_id) -> Dict`: Get PDF metadata
- `delete(citation_id) -> bool`: Delete PDF

## Complete Example

See [examples/llm_metadata_extraction.py](examples/llm_metadata_extraction.py) for a complete example showing:
- Citation retrieval
- PDF download
- Text extraction from PDF
- LLM-based metadata extraction
- Citation update

## Supported Fields

The library supports all fields from the HubZero DocumentExp model:

### Core Document Fields
- ID, title, abstract
- Publication ID/name
- Document genre ID/name
- Publication date
- Full text path (PDF)
- Timestamp

### BibTeX Fields
All standard BibTeX fields including address, booktitle, chapter, edition, editor, eprint, howpublished, institution, key, month, note, organization, publisher, series, school, type

### Paper/Journal Fields
- Volume, issue
- Begin page, end page

### NanoHub-Specific Fields
- URL, year, ISBN, cite
- Affiliated, funded by
- Created, DOI, reference type
- Status (workflow)
- Dates: submit, accept, publish
- Software use, research/education flags
- Experimental data fields
- Notes

### Citation Metrics
- Citation URL, search URL
- Citation count
- Last citation check date

### Related Data
- Authors (with full details: name, email, ORCID, etc.)
- Keywords

## Workflow Status Codes

- `0`: UNDEFINED
- `1`: RELATED
- `2`: PREVIEW
- `3`: REVIEW
- `4`: VOTING
- `5`: PROCESS
- `6`: POSTPROCESS
- `100`: PUBLISHED
- `-1` to `-6`: REJECTED
- `-9`: JUNK (deleted)

## Development

```bash
# Clone repository
git clone https://github.com/denphi/nanohub-citmanager.git
cd nanohub-citmanager

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black nanohubcitmanager/
```

## Requirements

- Python >= 3.8
- nanohubremote >= 0.2.0
- requests >= 2.25.0

Optional:
- ollama >= 0.1.0 (for LLM features)
- pypdf2 (for PDF text extraction)

## License

MIT License - see LICENSE file for details.

## Authors

- Daniel Mejia (denphi@denphi.com)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- [GitHub Repository](https://github.com/denphi/nanohub-citmanager)
- [nanohubremote](https://github.com/denphi/nanohub-remote)
- [NanoHub](https://nanohub.org)
- [API Documentation](../API_DOCUMENTATION.md)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{nanohub_citmanager,
  author = {Mejia, Daniel},
  title = {NanoHub Citation Manager Python Library},
  year = {2025},
  url = {https://github.com/denphi/nanohub-citmanager}
}
```

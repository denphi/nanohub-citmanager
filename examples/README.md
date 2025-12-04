# Citation Manager Examples

This directory contains example scripts for working with the NanoHub Citation Manager.

## Quick Start

```bash
# 1. Setup environment (first time only)
cp .env.example .env
# Edit .env with your credentials

# 2. Load environment (each session)
source load_env.sh

# 3. Run scripts
python batch_process_citations.py --status 3 --year 2025 --limit 5
```

## Files

- **`.env.example`** - Template for environment configuration
- **`.env`** - Your actual credentials (git-ignored)
- **`load_env.sh`** - Helper to load environment variables
- **`llm_metadata_extraction.py`** - Process single citation
- **`batch_process_citations.py`** - Batch process multiple citations

## Environment Variables

Required in `.env`:
- `NANOHUB_TOKEN` - Your NanoHub API token
- `OPENWEBUI_KEY` - Your LLM API key
- `NANOHUB_URL` - NanoHub API endpoint (default: https://nanohub.org/api)
- `OPENWEBUI_URL` - LLM API endpoint
- `LLM_MODEL` - Model to use (default: gpt-oss:120b)

## Common Commands

```bash
# Process recent citations from 2025
python batch_process_citations.py --status 3 --year 2025 --limit 5

# Preview without processing (dry run)
python batch_process_citations.py --status 3 --year 2025 --limit 10 --dry-run

# Process single citation
python llm_metadata_extraction.py 12345

# Get help
python batch_process_citations.py --help
```

## Documentation

- [Quick Start Guide](../QUICKSTART.md) - Get started in 5 minutes
- [Usage Examples](../USAGE_EXAMPLES.md) - Detailed examples
- [API Documentation](../../API_DOCUMENTATION.md) - API reference

## Troubleshooting

**Environment not loaded?**
```bash
source load_env.sh
```

**Missing .env file?**
```bash
cp .env.example .env
# Edit with your credentials
```

**Permission denied?**
```bash
chmod +x load_env.sh batch_process_citations.py llm_metadata_extraction.py
```

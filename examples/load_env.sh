#!/bin/bash
# Load environment variables from .env file
# Usage: source load_env.sh
# Must be sourced from the examples directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="${SCRIPT_DIR}/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please create one from .env.example:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  cp .env.example .env"
    echo "  # Then edit .env with your credentials"
    return 1
fi

# Load environment variables
echo "Loading environment variables from $ENV_FILE"
set -a
source "$ENV_FILE"
set +a

echo "Environment variables loaded:"
echo "  NANOHUB_URL: ${NANOHUB_URL}"
echo "  NANOHUB_TOKEN: ${NANOHUB_TOKEN:0:10}..." # Show only first 10 chars
echo "  OPENWEBUI_KEY: ${OPENWEBUI_KEY:0:10}..."
echo "  OPENWEBUI_URL: ${OPENWEBUI_URL}"
echo "  LLM_MODEL: ${LLM_MODEL}"
echo ""
echo "Ready to run scripts!"

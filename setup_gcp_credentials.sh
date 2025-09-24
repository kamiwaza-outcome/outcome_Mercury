#!/bin/bash
# Setup Google Cloud credentials for the RFP Discovery System

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to service account file
SERVICE_ACCOUNT_PATH="$SCRIPT_DIR/credentials/service-account.json"

# Check if service account file exists
if [ ! -f "$SERVICE_ACCOUNT_PATH" ]; then
    echo "✗ Service account file not found at: $SERVICE_ACCOUNT_PATH"
    exit 1
fi

# Export environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$SERVICE_ACCOUNT_PATH"

echo "✓ Google Cloud credentials configured successfully!"
echo "  GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
echo ""
echo "Project details:"
python3 -c "import json; f=open('$SERVICE_ACCOUNT_PATH'); d=json.load(f); print(f'  Project ID: {d[\"project_id\"]}'); print(f'  Service Account: {d[\"client_email\"]}')"

echo ""
echo "To make this permanent, add the following to your shell profile (~/.bashrc or ~/.zshrc):"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"$SERVICE_ACCOUNT_PATH\""
#!/bin/bash

echo "Starting Mercury Blue ALLY Application..."

# Create environment file if it doesn't exist
if [ ! -f backend/.env ]; then
    echo "Creating backend/.env from example..."
    cat > backend/.env << EOF
# Kamiwaza Configuration (for local model deployment)
KAMIWAZA_ENDPOINT=http://localhost:7777/api/
KAMIWAZA_VERIFY_SSL=false
KAMIWAZA_DEFAULT_MODEL=llama3
KAMIWAZA_FALLBACK_MODEL=mistral
KAMIWAZA_STREAMING=false
KAMIWAZA_MAX_RETRIES=3
KAMIWAZA_RETRY_DELAY=5

# Google Services Configuration (optional - will work without these)
GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/service-account.json
OUTPUT_FOLDER_ID=your_google_drive_output_folder_id
TEMPLATE_FOLDER_ID=your_google_drive_template_folder_id
TRACKING_SHEET_ID=your_google_sheets_tracking_id

# SAM.gov API Configuration (optional)
SAM_API_KEY=your_sam_api_key

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6380

# Processing Options
SKIP_REVISION_PHASE=false

# Retry Configuration
MAX_RETRY_ATTEMPTS=5
RETRY_MIN_WAIT=4
RETRY_MAX_WAIT=60

# Heartbeat Configuration
HEARTBEAT_INTERVAL=30

# Checkpoint Directory
CHECKPOINT_DIR=checkpoints
EOF
fi

# Start backend
echo "Starting backend server on port 8000..."
cd backend
python3 -m uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Start frontend
echo "Starting frontend server on port 3003..."
cd ../frontend
npm install
PORT=3003 npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "========================================="
echo "Mercury Blue ALLY is starting up!"
echo "========================================="
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:3003"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "New Kamiwaza Model Endpoints:"
echo "- List Models: http://localhost:8000/api/models"
echo "- Model Health: http://localhost:8000/api/models/health"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "========================================="

# Function to handle cleanup
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up trap to catch Ctrl+C
trap cleanup INT

# Keep script running
wait
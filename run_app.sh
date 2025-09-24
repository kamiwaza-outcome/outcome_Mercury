#!/bin/bash

echo "🚀 Starting Mercury Blue ALLY with Kamiwaza SDK"
echo "================================================"

# Create .env if it doesn't exist
if [ ! -f backend/.env ]; then
    echo "📝 Creating backend/.env configuration..."
    cat > backend/.env << EOF
# Kamiwaza Configuration
KAMIWAZA_ENDPOINT=http://localhost:7777/api/
KAMIWAZA_VERIFY_SSL=false
KAMIWAZA_DEFAULT_MODEL=llama3
KAMIWAZA_FALLBACK_MODEL=mistral
EOF
fi

# Start Frontend on port 3003
echo "🌐 Starting Frontend on port 3003..."
cd frontend
npm install --silent
PORT=3003 npm run dev &
FRONTEND_PID=$!

cd ..

# Start Backend on port 8000
echo "⚙️  Starting Backend on port 8000..."
cd backend
python3 -m uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

cd ..

echo ""
echo "================================================"
echo "✅ Application Started Successfully!"
echo "================================================"
echo ""
echo "🌐 Frontend: http://localhost:3003"
echo "⚙️  Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "🆕 Kamiwaza Model Management:"
echo "   • List Models: http://localhost:8000/api/models"
echo "   • Model Health: http://localhost:8000/api/models/health"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "================================================"

# Handle cleanup
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $FRONTEND_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT

# Keep running
wait
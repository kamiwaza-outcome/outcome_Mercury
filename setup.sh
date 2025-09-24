#!/bin/bash

echo "🚀 Setting up Mercury RFP Automation System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi

# Create virtual environment for backend
echo "📦 Creating Python virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright browsers for Browser Use
echo "🌐 Installing Playwright browsers..."
playwright install chromium

# Deactivate virtual environment
deactivate

cd ..

# Install frontend dependencies
echo "📦 Installing Node.js dependencies..."
cd frontend
npm install

cd ..

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p credentials
mkdir -p company_documents

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your Google service account JSON to: ./credentials/service-account.json"
echo "2. Add company documents to: ./company_documents/"
echo "3. Start the backend: cd backend && source venv/bin/activate && python main.py"
echo "4. Start the frontend: cd frontend && npm run dev"
echo ""
echo "The application will be available at:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
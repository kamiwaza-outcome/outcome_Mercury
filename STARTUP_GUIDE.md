# Mercury RFP System - Quick Start Guide

## Current Status ✅
The system is now ready to run! All components have been configured and tested.

### What's Working:
- ✅ Google Cloud Service Account credentials configured
- ✅ Backend server with FastAPI
- ✅ Frontend with Next.js
- ✅ Milvus vector database
- ✅ All dependencies installed
- ✅ Environment variables configured

### Quick Start Commands:

#### Option 1: Use the start script (Recommended)
```bash
./start.sh
```
This will start both backend and frontend services together.

#### Option 2: Start services individually

**Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Frontend (in a new terminal):**
```bash
cd frontend
npm run dev
```

### Access Points:
- 🌐 Frontend: http://localhost:3001
- 🔧 Backend API: http://localhost:8000
- 📚 API Documentation: http://localhost:8000/docs

### Important Files:
- `.env` - Contains all API keys and configuration
- `credentials/service-account.json` - Google Cloud authentication
- `company_documents/` - Place company docs here for RAG system

### Next Steps:
1. Add company documents to `company_documents/` folder
2. Access the frontend at http://localhost:3001
3. Start processing RFPs!

### Google Sheets Setup:
Make sure to share your Google Sheet (ID: 1vHu0YgUuKjelv8pMEYQ0Bbcz1Cu_pYIaMhzBDJD_144) with:
- Service Account: rfp-discovery-bot@rfp-discovery-system.iam.gserviceaccount.com

### Troubleshooting:
- If port 3000 is in use, the frontend runs on port 3001
- Backend always runs on port 8000
- Check logs in terminal for any errors
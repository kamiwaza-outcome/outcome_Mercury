# Mercury RFP Automation System

An advanced RFP response automation system powered by GPT-5, designed to streamline the process of responding to government RFPs from SAM.gov.

## Features

- **Automated RFP Detection**: Monitors Google Sheets for checked RFPs
- **Intelligent Document Scraping**: Uses Browser Use and SAM.gov API to download all RFP materials
- **AI-Powered Analysis**: Creates comprehensive Northstar documents with GPT-5
- **Dynamic Document Generation**: Generates all required RFP response documents
- **RAG System**: Uses Milvus Lite for company knowledge retrieval
- **Quality Assurance**: Multi-agent evaluation and revision system
- **Human Review Guide**: Detailed guidance on what needs human attention
- **Google Drive Integration**: Automatic upload and organization of all documents
- **Dual Architecture Support**: Switch between Classic (monolithic) and Dynamic (multi-agent) processing
- **17 Specialized Agents**: Domain-specific expertise for federal RFP requirements
- **Anti-Template Engine**: Ensures unique, non-templated responses

## Architecture

### Technology Stack

#### Core Technologies
- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Backend**: FastAPI with async support
- **LLM**: Direct OpenAI API (GPT-5 with GPT-5-mini fallback)
- **Vector DB**: Milvus Lite for RAG
- **Web Automation**: Browser Use with Playwright
- **APIs**: Google Sheets, Google Drive, SAM.gov

#### Orchestration Approach
- **Custom-Built Stack**: No LangChain, DSPy, or Haystack dependencies
- **Direct OpenAI Integration**: Full control over prompts and agent behavior
- **Native Async**: Python asyncio for parallel processing
- **Session Management**: Stateful architecture configuration
- **Confidence Routing**: Custom scoring system (0-1) for agent assignment

### Processing Architectures

The system supports two processing architectures:

#### Classic Architecture (Monolithic)
- Single orchestration agent handles all document generation
- Sequential processing of documents
- Unified context management
- Best for: Smaller RFPs, consistent formatting

#### Dynamic Multi-Agent Architecture
- **17 Specialized Agents** working in parallel:
  - **9 Base Agents**: TechnicalArchitecture, CostPricing, Compliance, Security, PastPerformance, VideoScript, ExecutiveSummary, Generalist, Adaptive
  - **8 Federal Agents**: CybersecurityCompliance, OCI, SmallBusinessSubcontracting, KeyPersonnel, QualityAssurancePlan, RiskManagement, TransitionPlan, Sustainability
- Confidence-based agent assignment (0-1 scoring)
- Parallel document generation for faster processing
- Anti-template engine for unique responses
- Best for: Complex RFPs, high-quality requirements, faster turnaround

## Setup

### Prerequisites

- Python 3.8+
- Node.js 18+
- Google Service Account with Sheets and Drive API access
- OpenAI API key (GPT-5 access)
- SAM.gov API key (optional but recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/finnegannorris/Mercury_RFP.git
cd Mercury_RFP
```

2. Run the setup script:
```bash
./setup.sh
```

3. Configure credentials:
   - Add your Google service account JSON to `./credentials/service-account.json`
   - Verify `.env` file contains all required API keys

4. Add company documents:
   - Place company information documents in `./company_documents/`
   - Supported formats: PDF, DOCX, TXT, MD

### Running the Application

1. Start the backend server:
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

2. In a new terminal, start the frontend:
```bash
cd frontend
npm run dev
```

3. Access the application:
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

## Usage

1. **Prepare Google Sheet**:
   - Ensure your RFP tracking sheet has checkboxes in column B
   - Check the box next to RFPs you want to process

2. **Process RFPs**:
   - Click "Process RFPs" button in the web interface
   - Monitor progress in real-time

3. **Review Results**:
   - Documents are automatically uploaded to Google Drive
   - Each RFP gets its own folder with:
     - Source documents
     - Generated responses
     - Northstar analysis
     - Human review guide

4. **Complete Submission**:
   - Review the Human Review Guide
   - Make necessary edits
   - Submit through appropriate channels

## Document Generation Process

1. **Northstar Creation**: Comprehensive analysis of all RFP requirements
2. **Document Identification**: AI identifies all required deliverables
3. **Content Generation**: Specialized agents create each document
4. **Evaluation**: Quality checks and compliance verification
5. **Revision**: Automatic fixes for identified issues
6. **Review Guide**: Clear instructions for human completion

## API Endpoints

### RFP Processing
- `GET /api/rfps/pending` - Get RFPs with checked boxes
- `POST /api/rfps/process` - Start processing checked RFPs
- `GET /api/rfps/status/{notice_id}` - Get status of specific RFP
- `GET /api/rfps/status` - Get all processing statuses
- `POST /api/rfps/process-with-architecture` - Process with specific architecture

### Architecture Management
- `POST /api/architecture/config` - Set processing architecture (classic/dynamic)
- `GET /api/architecture/config/{session_id}` - Get current architecture config
- `GET /api/architecture/metrics` - Get processing metrics for comparison
- `POST /api/architecture/ab-test` - Configure A/B testing

### Knowledge Management
- `POST /api/rag/index` - Index company documents
- `GET /api/health` - System health check

## Environment Variables

```env
# OpenAI
OPENAI_API_KEY=your_gpt5_api_key
OPENAI_MODEL=gpt-5
OPENAI_FALLBACK_MODEL=gpt-5-mini

# SAM.gov
SAM_API_KEY=your_sam_api_key

# Google Workspace
TRACKING_SHEET_ID=your_sheet_id
OUTPUT_FOLDER_ID=drive_folder_id
TEMPLATE_FOLDER_ID=template_folder_id
SHEET_RANGE=A1:Z1000
GOOGLE_SERVICE_ACCOUNT_FILE=./credentials/service-account.json
```

## Troubleshooting

### Common Issues

1. **Google API Authentication**:
   - Ensure service account has proper permissions
   - Share Google Sheet and Drive folders with service account email

2. **Milvus Initialization**:
   - Delete `milvus_rfp.db` if corrupted
   - System will recreate on next startup

3. **Browser Automation**:
   - Run `playwright install chromium` if browser not found
   - Check firewall settings for headless browser

## Security Notes

- Never commit `.env` or credential files
- Use environment-specific configurations
- Regularly rotate API keys
- Monitor API usage and costs

## Performance Metrics

### Dynamic Architecture Results
- **Quality Score**: 96.3/100
- **Processing Time**: ~14 minutes for complex RFPs
- **Parallel Execution**: All agents work concurrently
- **Success Rate**: 100% document generation completion

### Classic vs Dynamic Comparison
| Metric | Classic | Dynamic |
|--------|---------|---------|
| Processing Time | ~45 minutes | ~14 minutes |
| Quality Score | 68-86/100 | 96.3/100 |
| Revision Success | Degradation observed | Quality maintained |
| Agent Utilization | Single agent | 17 parallel agents |
| Context Management | Unified | Distributed |

## Support

For issues or questions, please contact the development team or create an issue in the repository.
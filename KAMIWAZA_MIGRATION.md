# Kamiwaza SDK Migration Guide

## Overview
This document describes the migration from OpenAI/GPT-5 APIs to Kamiwaza SDK for local model deployment in the Mercury Blue ALLY application.

## What Changed

### 1. **Dependencies**
- Added Kamiwaza SDK to `backend/requirements.txt`
- Removed dependency on external OpenAI API keys

### 2. **New Services**
- **`backend/services/kamiwaza_service.py`**: Core Kamiwaza SDK wrapper providing:
  - Model discovery and listing
  - OpenAI-compatible client interface
  - Health checks
  - Automatic fallback handling

### 3. **Modified Services**
- **`backend/services/openai_service.py`**: Refactored to use Kamiwaza SDK while maintaining the same interface
  - All OpenAI API calls now route through Kamiwaza
  - Automatic model name mapping (GPT-5 → Kamiwaza models)
  - Circuit breaker pattern retained for resilience

### 4. **New API Endpoints**
Added three new endpoints for model management:
- `GET /api/models` - List available Kamiwaza models
- `POST /api/models/select` - Select a model for the session
- `GET /api/models/health` - Check Kamiwaza connection health

### 5. **Environment Configuration**
Updated `.env.example` with Kamiwaza settings:
```env
# Kamiwaza Configuration
KAMIWAZA_ENDPOINT=http://host.docker.internal:7777/api/
KAMIWAZA_VERIFY_SSL=false
KAMIWAZA_DEFAULT_MODEL=llama3
KAMIWAZA_FALLBACK_MODEL=mistral
KAMIWAZA_STREAMING=false
KAMIWAZA_MAX_RETRIES=3
KAMIWAZA_RETRY_DELAY=5
```

## Benefits of Migration

### 1. **Data Security**
- All AI processing happens locally on your Kamiwaza deployment
- No data sent to external APIs
- Complete control over your infrastructure

### 2. **Cost Savings**
- No API usage fees
- Predictable infrastructure costs
- Better resource utilization

### 3. **Performance**
- Lower latency (no internet round-trips)
- Better throughput for batch processing
- Ability to scale horizontally

### 4. **Flexibility**
- Choose from multiple models based on task requirements
- Deploy custom fine-tuned models
- Switch models on-the-fly

## How It Works

### Model Selection Flow
1. Application queries available models from Kamiwaza
2. User can select a model via API or use auto-selection
3. All AI calls use the selected model through OpenAI-compatible interface

### Backward Compatibility
The migration maintains 100% backward compatibility:
- Existing code using `OpenAIService` continues to work
- Model names are automatically mapped (GPT-5 → default Kamiwaza model)
- Same async/sync interfaces preserved

### Example Usage

#### List Available Models
```python
from services.openai_service import OpenAIService

service = OpenAIService()
models = await service.list_available_models()
print(f"Available models: {[m['name'] for m in models]}")
```

#### Select a Model
```python
selected = await service.select_model(model_name="llama3")
# or by capability
selected = await service.select_model(capability="chat")
```

#### Generate Completion (unchanged from before)
```python
response = await service.get_completion(
    prompt="Analyze this RFP requirement...",
    model="llama3",  # or use "gpt-5" for auto-mapping
    temperature=0.7,
    max_tokens=1000
)
```

## Setup Instructions

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
# Edit .env to set your Kamiwaza endpoint
```

### 3. Verify Kamiwaza Connection
```bash
# Start the backend
python -m uvicorn main:app --reload

# Check health
curl http://localhost:8000/api/models/health
```

### 4. List Available Models
```bash
curl http://localhost:8000/api/models
```

## Troubleshooting

### Connection Issues
- Verify Kamiwaza is running at the configured endpoint
- Check firewall/network settings
- Ensure SSL settings match your Kamiwaza deployment

### Model Not Found
- Run `/api/models` to see available models
- Update `KAMIWAZA_DEFAULT_MODEL` in `.env`
- Ensure selected model is deployed and active

### Performance Issues
- Check Kamiwaza server resources
- Monitor model loading times
- Consider using smaller models for faster response

## Migration Checklist

- [x] Add Kamiwaza SDK to requirements.txt
- [x] Create kamiwaza_service.py wrapper
- [x] Update openai_service.py to use Kamiwaza
- [x] Add model management API endpoints
- [x] Update environment configuration
- [x] Create migration documentation
- [ ] Test with all existing workflows
- [ ] Update frontend to show model selection UI
- [ ] Deploy to staging environment
- [ ] Monitor performance metrics

## Next Steps

1. **Frontend Integration**: Add UI for model selection in the frontend
2. **Model Optimization**: Test different models for various tasks
3. **Performance Tuning**: Optimize model loading and caching
4. **Monitoring**: Add detailed metrics for model usage and performance

## Support

For issues or questions about the Kamiwaza integration:
1. Check Kamiwaza SDK documentation
2. Review logs in `backend/logs/`
3. Test connection with `/api/models/health` endpoint

## Notes

- The app_garden_template_test1 directory contains a reference implementation
- All changes maintain backward compatibility
- No changes required to existing business logic
- Model selection can be automated based on task type
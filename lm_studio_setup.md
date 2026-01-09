# LM Studio Setup for Local LLM Integration

## Prerequisites
1. Download and install LM Studio from: https://lmstudio.ai/
2. Download the model: qwen/qwen3-4b-thinking-2507

## Setup Steps

### 1. Start LM Studio
- Open LM Studio application

### 2. Load the Model
- Go to "Models" tab
- Search for "qwen/qwen3-4b-thinking-2507"
- Download the model (if not already downloaded)
- Load the model

### 3. Start Local Server
- Go to "Local Server" tab
- Click "Start Server"
- Verify it's running on `http://localhost:1234`

### 4. Test Connection
```bash
curl http://localhost:1234/v1/models
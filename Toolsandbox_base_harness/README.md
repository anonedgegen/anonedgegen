# ToolSandbox Agent — base (Azure GPT)

**Paper:** [Tool Sandbox (arXiv)](https://arxiv.org/abs/2408.04682)  
**Original repo:** [apple/ToolSandbox](https://github.com/apple/ToolSandbox/tree/main)

The baseline ToolSandbox harness, backed by an Azure OpenAI model.

## Setup

```bash
cd ToolSandbox
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create an `.env` file and fill in your Azure OpenAI
credentials. Example `.env`:

```bash
OPENAI_API_TYPE="azure"
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_API_VERSION="2024-12-01-preview"
AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
```

Start the server (port **8000**):

```bash
uvicorn fastapi_server.main:app --reload --port 8000
```

## Model

The model is chosen per session via `agent_type` (from `src/tool_sandbox/cli/utils.py`,
e.g. `GPT_4_1`):

```json
{
    "agent_type": "GPT_4_1",
    "scenario": "optional_scenario_name"
}
```

`scenario` is optional; omit it to use the seeded base context.

## API

Server runs at `http://localhost:8000` (interactive docs at `/docs`).

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/start_session` | Start a session → `{"session_id": "..."}` |
| POST | `/message` | Send a message → `{"content": "..."}` |
| GET  | `/trajectory/{session_id}` | Full conversation trajectory |
| POST | `/end_session/{session_id}` | End the session |

```bash
# start, then send a message
curl -X POST http://localhost:8000/start_session \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "GPT_4_1"}'

curl -X POST http://localhost:8000/message \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>", "message": "What is the weather in Paris?"}'
```

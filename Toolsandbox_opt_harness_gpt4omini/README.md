# ToolSandbox Agent — optimized (GPT-4o-mini)

**Paper:** [Tool Sandbox (arXiv)](https://arxiv.org/abs/2408.04682)  
**Original repo:** [apple/ToolSandbox](https://github.com/apple/ToolSandbox/tree/main)

The optimized ToolSandbox harness,
backed by an Azure OpenAI `gpt-4o-mini` deployment.


## Setup

Use Python 3.12. Install the pinned requirements first, then the project without
re-resolving dependencies:

```bash
cd ToolSandbox
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
cp .env.example .env
```

Fill in your Azure OpenAI credentials in `.env`. Example `.env`:

```bash
OPENAI_API_TYPE="azure"
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_API_VERSION="2024-12-01-preview"
AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
```

Start the server (port **5501**):

```bash
python -m uvicorn fastapi_server.main:app --host 127.0.0.1 --port 5501
```

## Model

Select the agent per session via `agent_type` (uses the Azure deployment named
`gpt-4o-mini`):

```json
{
    "agent_type": "GPT_4_o_mini",
    "scenario": "optional_scenario_name"
}
```

`scenario` is optional; omit it to use the seeded base context.

## API

Server runs at `http://localhost:5501` (interactive docs at `/docs`).

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/start_session` | Start a session → `{"session_id": "..."}` |
| POST | `/message` | Send a message → `{"content": "..."}` |
| GET  | `/trajectory/{session_id}` | Full conversation trajectory |
| POST | `/end_session/{session_id}` | End the session |

```bash
# start, then send a message
curl -X POST http://localhost:5501/start_session \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "GPT_4_o_mini"}'

curl -X POST http://localhost:5501/message \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>", "message": "What is the weather in Paris?"}'
```

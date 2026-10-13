# ToolSandbox Agent — base (Gemma)

**Paper:** [Tool Sandbox (arXiv)](https://arxiv.org/abs/2408.04682)  
**Original repo:** [apple/ToolSandbox](https://github.com/apple/ToolSandbox/tree/main)

The baseline ToolSandbox harness, backed by a **separately deployed open-source Gemma
model** served behind an OpenAI-compatible endpoint (e.g. vLLM). Unlike the Tau2 Gemma
harness, the endpoint is read from the server's `GEMMA_API_BASE` environment variable,
and the Gemma agent is selected per session with `agent_type: "Gemma_4_E4B"`.

## Setup

```bash
cd ToolSandbox
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No Azure key is needed for Gemma;
the model endpoint and key are set when starting the server (see below).

Start the server (port **8000**) with your Gemma endpoint in the environment:

```bash
GEMMA_API_BASE=http://localhost:8001/v1 \
OPENAI_API_KEY=dummy \
uvicorn fastapi_server.main:app --reload --port 8000
```

- `GEMMA_API_BASE` — base URL of the Gemma endpoint (default `http://localhost:8001/v1`).
- `OPENAI_API_KEY` — key for that endpoint; `dummy` works if it doesn't enforce auth.

## Model

Select the Gemma agent per session via `agent_type` (model name `gemma-4-e4b`):

```json
{
    "agent_type": "Gemma_4_E4B",
    "scenario": "optional_scenario_name"
}
```

Other agent types from `src/tool_sandbox/cli/utils.py` (e.g. `GPT_4_1`) select different
providers. `scenario` is optional.

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
  -d '{"agent_type": "Gemma_4_E4B"}'

curl -X POST http://localhost:8000/message \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>", "message": "What is the weather in Paris?"}'
```

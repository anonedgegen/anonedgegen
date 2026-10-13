# ToolSandbox Agent — optimized (Gemma)

**Paper:** [Tool Sandbox (arXiv)](https://arxiv.org/abs/2408.04682)  
**Original repo:** [apple/ToolSandbox](https://github.com/apple/ToolSandbox/tree/main)

The optimized ToolSandbox harness,
backed by a **separately deployed open-source Gemma model** served behind an
OpenAI-compatible endpoint. The endpoint is read from the server's `GEMMA_API_BASE`
environment variable, and the Gemma agent is selected per session with
`agent_type: "Gemma_4_E4B"`.

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

No Azure key is needed for Gemma; the model endpoint and key are set when starting
the server (see below).

Start the server (port **5501**) with your Gemma endpoint in the environment:

```bash
GEMMA_API_BASE=http://localhost:8001/v1 \
OPENAI_API_KEY=dummy \
python -m uvicorn fastapi_server.main:app --host 127.0.0.1 --port 5501
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

Other agent types from `src/tool_sandbox/cli/utils.py` select different providers.
`scenario` is optional.

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
  -d '{"agent_type": "Gemma_4_E4B"}'

curl -X POST http://localhost:5501/message \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>", "message": "What is the weather in Paris?"}'
```

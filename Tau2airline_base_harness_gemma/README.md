# Tau2 Airline Agent — base (Gemma)

**Paper:** [Tau2 Bench (arXiv)](https://arxiv.org/abs/2506.07982)  
**Original repo:** [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)

The baseline Tau2 Airline harness, backed by a **separately deployed open-source
Gemma model** served behind an OpenAI-compatible endpoint (e.g. vLLM). The harness
does not host the model; it is given the model and endpoint when a conversation starts.

## Setup

```bash
cd tau2-bench
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
pip install fastapi
```

Rename `.env copy` to `.env`. No Azure agent key is needed for Gemma; set
`OPENAI_API_KEY` (a dummy value is fine if your endpoint allows it).

Start the server (port **5500**):

```bash
uvicorn fastapi_server.main:app --reload --port 5500
```

## Model

Point the agent at your deployed Gemma endpoint per conversation via `agent_llm`
(LiteLLM id) and `api_base`:

```json
{
    "domain": "airline",
    "agent_llm": "openai/gemma-4-e4b",
    "api_base": "http://localhost:8001/v1"
}
```

`api_base` is optional — omit it to fall back to a provider configured in `.env`.

## API

Server runs at `http://localhost:5500` (interactive docs at `/docs`).

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/start_conversation` | Start a session → `{"session_id": "..."}` |
| POST | `/conversation/{session_id}/message` | Send a message → `{"response": ..., "history": [...]}` |
| GET  | `/conversation/{session_id}/trajectory` | Full trajectory (incl. tool calls) |
| POST | `/conversation/{session_id}/end` | End the session |
| POST | `/agent_description` | Domain policy text for `{"domain": "airline"}` |

```bash
# start, then send a message
curl -X POST http://localhost:5500/start_conversation \
  -H "Content-Type: application/json" \
  -d '{"domain": "airline", "agent_llm": "openai/gemma-4-e4b", "api_base": "http://localhost:8001/v1"}'

curl -X POST http://localhost:5500/conversation/<session_id>/message \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to cancel my reservation."}'
```

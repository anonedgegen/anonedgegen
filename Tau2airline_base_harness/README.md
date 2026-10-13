# Tau2 Airline Agent — base (Azure GPT)

**Paper:** [Tau2 Bench (arXiv)](https://arxiv.org/abs/2506.07982)  
**Original repo:** [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)

The baseline Tau2 Airline harness, backed by an Azure OpenAI model.

## Setup

```bash
cd tau2-bench
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
pip install fastapi
```

Rename `.env copy` to `.env` and fill in your Azure OpenAI credentials
(`AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`). The Azure endpoint must be
the Azure Studio one (not ending in `/models`).

Start the server (port **5500**):

```bash
uvicorn fastapi_server.main:app --reload --port 5500
```

## Model

The model is chosen per conversation via `agent_llm` when starting a session:

```json
{
    "domain": "airline",
    "agent_llm": "azure/gpt-4.1"
}
```

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
  -d '{"domain": "airline", "agent_llm": "azure/gpt-4.1"}'

curl -X POST http://localhost:5500/conversation/<session_id>/message \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to cancel my reservation."}'
```

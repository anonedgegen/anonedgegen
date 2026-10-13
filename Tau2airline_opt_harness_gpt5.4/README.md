# Tau2 Airline Agent — optimized (Azure GPT-5.4)

**Paper:** [Tau2 Bench (arXiv)](https://arxiv.org/abs/2506.07982)  
**Original repo:** [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)

The optimized Tau2 Airline harness, backed by an Azure OpenAI model. 

## Setup

Use Python 3.12. Install the pinned requirements first, then the project without
re-resolving dependencies:

```bash
cd tau2-bench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
cp .env.example .env
```

Set `AZURE_API_KEY`, `AZURE_API_BASE`, and `AZURE_API_VERSION` in `.env`.

Start the server (port **5500**):

```bash
python -m uvicorn fastapi_server.main:app --host 127.0.0.1 --port 5500
```

## Model

The model is chosen per conversation via `agent_llm` (defaults to `azure/gpt-4.1` if
omitted, so pass it explicitly). Replace `gpt-5.4` with your Azure deployment name if
it differs:

```json
{
    "domain": "airline",
    "agent_llm": "azure/gpt-5.4"
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
  -d '{"domain": "airline", "agent_llm": "azure/gpt-5.4"}'

curl -X POST http://localhost:5500/conversation/<session_id>/message \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to cancel my reservation."}'
```

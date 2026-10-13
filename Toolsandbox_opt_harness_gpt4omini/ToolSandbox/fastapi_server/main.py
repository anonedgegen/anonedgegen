import os
import sys

# Add project root and src/ to path BEFORE any local imports that depend on tool_sandbox
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fastapi import FastAPI, HTTPException
from .session_manager import SessionManager
from .schemas import StartSessionRequest, StartSessionResponse, MessageRequest, MessageResponse
from pathlib import Path
import json

app = FastAPI()
sessions = SessionManager()

@app.post("/start_session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    # Use scenario if provided, otherwise None for scenario-less mode
    session_id = sessions.start_session(req.agent_type, req.scenario)
    return StartSessionResponse(session_id=session_id)

@app.post("/message", response_model=MessageResponse)
def message(req: MessageRequest):
    session = sessions.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # Add user message to context
    sessions.add_user_message(session, req.message)
    # Get agent response
    response, usage = sessions.agent_respond(session, session_id=req.session_id)
    if response is None:
        response = "Agent did not respond."
    return MessageResponse(content=response, usage=usage)

@app.post("/end_session/{session_id}")
def end_session(session_id: str):
    removed = sessions.end_session(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok", "session_id": session_id}

@app.get("/trajectory/{session_id}")
def get_trajectory(session_id: str):
    conversation_path = Path("data") / session_id / "conversation.json"
    if not conversation_path.exists():
        return [{"role": "system", "content": "Error 404: No conversation found."}]
    with open(conversation_path, "r", encoding="utf-8") as f:
        conversation = json.load(f)
    return conversation

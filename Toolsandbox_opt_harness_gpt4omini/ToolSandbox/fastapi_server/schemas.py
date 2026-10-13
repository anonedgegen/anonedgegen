from pydantic import BaseModel
from typing import Optional

class StartSessionRequest(BaseModel):
    agent_type: str
    scenario: Optional[str] = None  # Make scenario optional

class StartSessionResponse(BaseModel):
    session_id: str

class MessageRequest(BaseModel):
    session_id: str
    message: str

class MessageResponse(BaseModel):
    content: str
    usage: Optional[dict] = None

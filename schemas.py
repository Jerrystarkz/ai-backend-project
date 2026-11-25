from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class DocumentCreate(BaseModel):
    title: str
    content: str
    embedding: Optional[List[float]] = None

class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str
    embedding: Optional[List[float]] = None  # Ensure this line is present
    created_at: datetime

    class Config:
        from_attributes = True



class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"
    top_k: Optional[int] = 3

class SourceChunk(BaseModel):
    title: str
    score: float
    snippet: str

class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]
    session_id: str
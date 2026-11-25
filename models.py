from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship
from database import Base


# To make chat persistent across sessions even after the current page has been refreshed
class ChatMessage(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    chunk_metadata = Column(JSONB, nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)  # Standard Postgres array instead of Vector

    document = relationship("Document", back_populates="chunks")
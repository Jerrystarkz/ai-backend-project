from sqlalchemy.orm import Session
from models import ChatMessage
import models
import schemas


def get_documents(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Document).offset(skip).limit(limit).all()


def create_document_with_chunks(db: Session, filename: str, chunks_data: list):
    # 1. Create the parent Document record
    db_document = models.Document(filename=filename)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    # 2. Create the associated DocumentChunk records with metadata
    for index, chunk in enumerate(chunks_data):
        db_chunk = models.DocumentChunk(
            document_id=db_document.id,
            chunk_index=chunk.get("chunk_index", index),
            page_number=chunk.get("page_number"),
            chunk_metadata=chunk.get("metadata", {}),
            content=chunk["content"],
            embedding=chunk["embedding"]
        )
        db.add(db_chunk)

    db.commit()
    return db_document


# Helper functions to save messages and fetch history by session ID
def save_message(db: Session, session_id: str, role: str, content: str):
    db_message = models.ChatMessage(session_id=session_id, role=role, content=content)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


def get_chat_history(db: Session, session_id: str):
    return (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.timestamp.asc())
        .all()
    )


def save_chat_message(db: Session, session_id: str, role: str, content: str):
    db_message = models.ChatMessage(session_id=session_id, role=role, content=content)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


def get_session_history(db: Session, session_id: str, limit: int = 10):
    return (
        db.query(models.ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.timestamp.asc())
        .limit(limit)
        .all()
    )


create_document = create_document_with_chunks
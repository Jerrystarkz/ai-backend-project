from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import SessionLocal
import schemas
import crud
from pypdf import PdfReader
import io
from services.embedding import generate_embedding
from services.llm import generate_grounded_answer, generate_agent_response, stream_agent_response
from crud import save_chat_message, get_session_history, save_message, get_chat_history
import models
from utils import cosine_similarity

router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def chunk_text_with_overlap(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            search_window = cleaned[start + (chunk_size // 2):end]
            best_break = -1
            for sep in [". ", "? ", "! ", "; ", ", ", " "]:
                idx = search_window.rfind(sep)
                if idx != -1:
                    best_break = start + (chunk_size // 2) + idx + len(sep)
                    break
            if best_break > start:
                end = best_break

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break

        start = max(end - overlap, start + 1)

    return chunks


@router.post("/agent/")
async def agent_chat(payload: dict, db: Session = Depends(get_db)):
    query = payload.get("query", "")
    session_id = payload.get("session_id", "default_session")
    
    # 1. Save user query
    save_message(db, session_id, role="user", content=query)
    
    # 2. Generate response via Ollama + tools
    answer = generate_agent_response(query, db=db)
    
    # Safeguard against None or empty response
    if not answer:
        answer = "I'm sorry, I couldn't generate a response."
    
    # 3. Save assistant response safely
    save_message(db, session_id, role="assistant", content=answer)
    
    return {"query": query, "answer": answer, "session_id": session_id}


@router.get("/history/{session_id}")
async def get_history(session_id: str, db: Session = Depends(get_db)):
    messages = get_chat_history(db, session_id)
    return [{"role": msg.role, "content": msg.content} for msg in messages]


@router.post("/upload/")
async def upload_document_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Read file content and extract text with page & metadata awareness
    content = await file.read()
    chunks_data = []
    global_chunk_index = 0

    if file.filename and file.filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_chunks = chunk_text_with_overlap(page_text, chunk_size=800, overlap=150)
            for idx_on_page, text_piece in enumerate(page_chunks):
                enriched_content = (
                    f"[File: {file.filename} | Page: {page_num} | Chunk: {global_chunk_index}]\n"
                    f"{text_piece}"
                )
                vector = generate_embedding(enriched_content)
                chunks_data.append({
                    "content": enriched_content,
                    "embedding": vector,
                    "chunk_index": global_chunk_index,
                    "page_number": page_num,
                    "metadata": {
                        "filename": file.filename,
                        "source_type": "pdf",
                        "page_number": page_num,
                        "page_chunk_index": idx_on_page,
                        "chunk_index": global_chunk_index,
                        "char_length": len(text_piece),
                    },
                })
                global_chunk_index += 1
    else:
        text_content = content.decode("utf-8", errors="ignore")
        text_chunks = chunk_text_with_overlap(text_content, chunk_size=800, overlap=150)
        for text_piece in text_chunks:
            enriched_content = (
                f"[File: {file.filename} | Source: TXT | Chunk: {global_chunk_index}]\n"
                f"{text_piece}"
            )
            vector = generate_embedding(enriched_content)
            chunks_data.append({
                "content": enriched_content,
                "embedding": vector,
                "chunk_index": global_chunk_index,
                "page_number": None,
                "metadata": {
                    "filename": file.filename,
                    "source_type": "txt",
                    "page_number": None,
                    "chunk_index": global_chunk_index,
                    "char_length": len(text_piece),
                },
            })
            global_chunk_index += 1

    if not chunks_data:
        raise HTTPException(status_code=400, detail="Could not extract any text from the uploaded file.")

    # 2. Save to database using your CRUD helper
    db_document = crud.create_document_with_chunks(
        db=db,
        filename=file.filename,
        chunks_data=chunks_data,
    )

    return {
        "message": "Document uploaded and chunked successfully",
        "filename": db_document.filename,
        "chunks_created": len(chunks_data),
    }


@router.post("/search/")
def search_documents(query: str, top_k: int = 3, db: Session = Depends(get_db)):
    """
    Searches uploaded document chunks by computing semantic cosine similarity 
    against the query embedding vector.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    # 1. Generate embedding for the incoming question
    query_vector = generate_embedding(query)

    # 2. Fetch all document chunks from the database
    chunks = db.query(models.DocumentChunk).all()

    scored_results = []
    for chunk in chunks:
        if chunk.embedding:
            # 3. Compute similarity score between query and chunk embedding
            score = cosine_similarity(query_vector, chunk.embedding)
            scored_results.append({
                "id": chunk.id,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "metadata": chunk.chunk_metadata,
                "content": chunk.content,
                "similarity_score": round(score, 4)
            })

    # 4. Sort by highest similarity score and slice top_k results
    scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_matches = scored_results[:top_k]

    return {
        "query": query,
        "matches_found": len(top_matches),
        "results": top_matches
    }


@router.post("/ask/", response_model=schemas.AskResponse)
async def ask_documents(request: schemas.AskRequest, db: Session = Depends(get_db)):
    # 1. Fetch past conversation history
    history = get_session_history(db, session_id=request.session_id, limit=6)

    # 2. Embed query and retrieve chunks
    query_vector = generate_embedding(request.query)
    chunks = db.query(models.DocumentChunk).all()
    
    scored_chunks = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk.embedding)
        scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = scored_chunks[:request.top_k]

    retrieved_data = [
        {"title": chunk.document.filename, "content": chunk.content, "score": float(score)}
        for chunk, score in top_chunks
    ]

    # 3. Save user query to history
    save_chat_message(db, session_id=request.session_id, role="user", content=request.query)

    # 4. Generate answer with history and context
    answer = await generate_grounded_answer(request.query, retrieved_data, history=history)

    # 5. Save assistant response to history
    save_chat_message(db, session_id=request.session_id, role="assistant", content=answer)

    return schemas.AskResponse(
        query=request.query,
        answer=answer,
        sources=[schemas.SourceChunk(title=d["title"], score=d["score"], snippet=d["content"][:150]) for d in retrieved_data],
        session_id=request.session_id
    )


@router.post("/agent/stream/")
async def agent_chat_stream(payload: dict, db: Session = Depends(get_db)):
    query = payload.get("query", "")
    session_id = payload.get("session_id", "default_session")

    # 1. Save user query immediately
    save_message(db, session_id, role="user", content=query)

    def event_stream():
        collected_chunks = []
        stream_db = SessionLocal()
        try:
            for chunk in stream_agent_response(query, db=stream_db):
                collected_chunks.append(chunk)
                yield chunk

            final_answer = "".join(collected_chunks).strip()
            if not final_answer:
                final_answer = "I'm sorry, I couldn't generate a response."

            # 2. Save complete assistant response once streaming finishes
            save_message(stream_db, session_id, role="assistant", content=final_answer)
        finally:
            stream_db.close()

    return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")



from sqlalchemy import func

@router.get("/sessions/")
async def list_sessions(db: Session = Depends(get_db)):
    rows = (
        db.query(
            models.ChatMessage.session_id,
            func.max(models.ChatMessage.timestamp).label("last_active")
        )
        .filter(models.ChatMessage.session_id.isnot(None))
        .group_by(models.ChatMessage.session_id)
        .order_by(func.max(models.ChatMessage.timestamp).desc())
        .all()
    )
    return [row.session_id for row in rows if row.session_id]
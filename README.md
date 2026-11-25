# Local RAG & Autonomous Tool-Calling Assistant

A full-stack local Retrieval-Augmented Generation (RAG) and tool-calling agent built with **FastAPI**, **PostgreSQL**, **SQLAlchemy**, **Ollama (`llama3.2`)**, and **Streamlit**.

## Key Features

- **Page-Aware Ingestion & Overlapping Chunking:** Extracts `.pdf` (page-by-page) and `.txt` files using boundary-aware overlapping chunks (`800` characters, `150` overlap) with `page_number` and `JSONB` `chunk_metadata` persisted in PostgreSQL.
- **Autonomous Tool-Calling Agent:** Uses Ollama function calling to inspect database document counts (`get_document_count`), evaluate mathematical expressions (`calculate_expression`), and search stored document chunks (`search_document_content`).
- **Live Token Streaming:** Streams responses progressively from FastAPI (`StreamingResponse`) to Streamlit (`st.write_stream`) with explicit timeouts and graceful connection error handling.
- **Session-Persistent Chat History:** Persists all conversations in PostgreSQL (`chat_history`), allowing users to start new sessions, switch via a **Previous Sessions** selector, or resume past threads across browser refreshes.
- **Semantic Vector Search:** Supports cosine-similarity ranking over stored chunk embeddings (`/documents/search/` and `/documents/ask/`).

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, SQLAlchemy, Pydantic Settings, `pypdf`
- **Database:** PostgreSQL (`aidb` database with `ARRAY(Float)` embeddings and `JSONB` chunk metadata)
- **LLM & Embeddings:** Local Ollama (`llama3.2`, `nomic-embed-text`)
- **Frontend:** Streamlit

## Project Structure

```text
├── main.py                  # FastAPI application entrypoint
├── config.py                # Centralized Pydantic environment configuration
├── database.py              # SQLAlchemy engine and session management
├── models.py                # ORM models (Document, DocumentChunk, ChatMessage)
├── schemas.py               # Pydantic request/response schemas
├── crud.py                  # Database persistence and session history queries
├── utils.py                 # Cosine similarity helper
├── routers/
│   └── documents.py         # Upload, streaming agent, sessions, history, search & RAG routes
├── services/
│   ├── embedding.py         # Local embedding generation
│   ├── llm.py               # Ollama tool orchestration, streaming, and grounded QA
│   └── tools.py             # Local Python tools exposed to the LLM
├── app.py                   # Streamlit UI
├── .env.example             # Environment variable template
└── README.md                # Project documentation
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import SessionLocal, engine, Base
import models
from routers import documents
from database import engine, Base

# Automatically create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Backend API", version="1.0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Include the documents router
app.include_router(documents.router)

@app.get("/")
def read_root():
    return {"message": "AI Backend with Modular Architecture is active!"}

@app.get("/test-db")
def test_database_connection(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT version();")).fetchone()
        return {"status": "success", "postgresql_version": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
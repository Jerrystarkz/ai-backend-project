import math
from sqlalchemy.orm import Session
import models

# --- Tool 1: Real Database Query ---
def get_document_count(db: Session) -> str:
    """Returns the total number of ingested documents in the PostgreSQL database."""
    count = db.query(models.Document).count()
    return f"There are currently {count} document(s) stored in the database."

# --- Tool 2: Safe Python Math Evaluator ---
def calculate_expression(expression: str) -> str:
    """Evaluates basic mathematical expressions safely."""
    try:
        # Restrict execution environment for security
        allowed_names = {"sqrt": math.sqrt, "pow": math.pow, "abs": abs}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

#function that queries your database for document chunks matching a search term:

def search_document_content(keyword: str, db: Session, limit: int = 3):
    """
    Searches document chunks in the database containing the given keyword phrase.
    """
    results = db.query(models.DocumentChunk).filter(models.DocumentChunk.content.ilike(f"%{keyword}%")).limit(limit).all()
    
    if not results:
        return f"No document content found matching '{keyword}'."
    
    matches = []
    for r in results:
        doc_name = r.document.filename if r.document else "Unknown Document"
        matches.append(f"[File: {doc_name}] {r.content[:300]}...")
        
    return "\n\n".join(matches)

# Map string tool names to executable Python functions
AVAILABLE_TOOLS = {
    "get_document_count": get_document_count,
    "calculate_expression": calculate_expression,
    "search_document_content": search_document_content  # <--- Add this line
}
import json
import httpx
import requests
from config import settings
from services.tools import AVAILABLE_TOOLS, search_document_content

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "llama3.2"
OLLAMA_TIMEOUT = (5, 120)  # (connect timeout, read timeout) in seconds

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_document_count",
            "description": "Get the total number of documents uploaded and stored in the database.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "Perform mathematical calculations like square roots, addition, multiplication, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '25 * 40' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_document_content",
            "description": "Search through uploaded document chunks in the database using a keyword or specific phrase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The specific keyword, term, or phrase to search for inside the document contents."
                    }
                },
                "required": ["keyword"]
            }
        }
    }
]


def _execute_tool_call(tool_call: dict, db=None) -> str:
    func_name = tool_call.get("function", {}).get("name", "")
    args = tool_call.get("function", {}).get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    try:
        if func_name == "get_document_count":
            return str(AVAILABLE_TOOLS[func_name](db))
        elif func_name == "calculate_expression":
            return str(AVAILABLE_TOOLS[func_name](args.get("expression", "")))
        elif func_name == "search_document_content":
            return str(search_document_content(keyword=args.get("keyword", ""), db=db))
        return f"Tool '{func_name}' not found."
    except Exception as exc:
        return f"Tool '{func_name}' failed: {exc}"


def generate_agent_response(user_query: str, db=None) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to local tools for checking database records, "
                "searching uploaded document content, and calculating math. When a user asks for document counts, "
                "document contents, or math calculations, you MUST use the provided tool calls rather than answering from general knowledge."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": TOOLS_SCHEMA,
                "stream": False
            },
            timeout=OLLAMA_TIMEOUT
        )
        res.raise_for_status()
        response = res.json()
    except requests.exceptions.ConnectionError:
        return "Could not connect to the local Ollama service. Please ensure Ollama is running."
    except requests.exceptions.Timeout:
        return "The request to Ollama timed out. Please try again."
    except (requests.exceptions.RequestException, ValueError) as exc:
        return f"Error communicating with Ollama: {exc}"

    message = response.get("message", {})

    if "tool_calls" in message and message["tool_calls"]:
        tool_output = _execute_tool_call(message["tool_calls"][0], db=db)

        messages.append(message)
        messages.append({"role": "tool", "content": tool_output})
        messages.append({
            "role": "user",
            "content": "Answer the user's question directly and clearly based on the tool output above."
        })

        try:
            final_res = requests.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "messages": messages, "stream": False},
                timeout=OLLAMA_TIMEOUT
            )
            final_res.raise_for_status()
            return final_res.json().get("message", {}).get("content", "")
        except requests.exceptions.Timeout:
            return "Ollama timed out while summarizing the tool result."
        except (requests.exceptions.RequestException, ValueError) as exc:
            return f"Error generating final response from Ollama: {exc}"

    return message.get("content", "")


def stream_agent_response(user_query: str, db=None):
    """
    Yields response chunks progressively for streaming endpoints.
    Handles tool calls first if needed, then streams the final answer safely.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to local tools for checking database records, "
                "searching uploaded document content, and calculating math. When a user asks for document counts, "
                "document contents, or math calculations, you MUST use the provided tool calls rather than answering from general knowledge."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    try:
        initial_res = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": TOOLS_SCHEMA,
                "stream": False
            },
            timeout=OLLAMA_TIMEOUT
        )
        initial_res.raise_for_status()
        initial_response = initial_res.json()
    except requests.exceptions.ConnectionError:
        yield "Could not connect to the local Ollama service. Please ensure Ollama is running."
        return
    except requests.exceptions.Timeout:
        yield "The request to Ollama timed out. Please try again."
        return
    except (requests.exceptions.RequestException, ValueError) as exc:
        yield f"Error communicating with Ollama: {exc}"
        return

    message = initial_response.get("message", {})

    if "tool_calls" in message and message["tool_calls"]:
        tool_output = _execute_tool_call(message["tool_calls"][0], db=db)

        messages.append(message)
        messages.append({"role": "tool", "content": tool_output})
        messages.append({
            "role": "user",
            "content": "Answer the user's question directly and clearly based on the tool output above."
        })

        stream_payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True
        }
    else:
        direct_content = message.get("content", "")
        if direct_content:
            yield direct_content
            return

        stream_payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True
        }

    try:
        with requests.post(OLLAMA_URL, json=stream_payload, stream=True, timeout=OLLAMA_TIMEOUT) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
    except requests.exceptions.ConnectionError:
        yield "\n\n[Connection to Ollama was lost while streaming.]"
    except requests.exceptions.Timeout:
        yield "\n\n[Ollama stream timed out.]"
    except requests.exceptions.RequestException as exc:
        yield f"\n\n[Streaming error: {exc}]"


async def generate_grounded_answer(query: str, retrieved_chunks: list[dict], history: list = None) -> str:
    context_text = "\n\n".join(
        [f"--- Context Block {i+1} ({chunk['title']}) ---\n{chunk['content']}"
         for i, chunk in enumerate(retrieved_chunks)]
    )

    system_prompt = (
        "You are an intelligent and helpful assistant. Use the provided document context to "
        "answer the user's question as accurately as possible. \n\n"
        "Guidelines:\n"
        "1. Prioritize information found in the retrieved context.\n"
        "2. If the context partially addresses the question, answer using what is available and "
        "note what details were missing.\n"
        "3. If the context is relevant but doesn't directly state the answer, use reasonable logical inference.\n"
        "4. Only state that you cannot answer if the provided context is completely unrelated to the question."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

    user_prompt = f"""Question: {query}

Retrieved Documents Context:
{context_text}

Instructions: Provide a clear, concise answer grounded in the context above."""

    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(settings.llm_api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
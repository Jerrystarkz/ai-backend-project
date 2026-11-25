from datetime import datetime
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Jerrystarkz RAG  ", page_icon="🤖", layout="wide")

st.title("📂 Jerrystarkz RAG   (FastAPI + Ollama)")
st.sidebar.header("Document Management")

# --- 1. File Upload Section ---
uploaded_file = st.sidebar.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])
if uploaded_file is not None:
    if st.sidebar.button("Ingest Document"):
        with st.spinner("Processing and embedding document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/documents/upload/", files=files, timeout=(5, 120))
                if response.status_code == 200:
                    data = response.json()
                    chunks_created = data.get("chunks_created", "")
                    suffix = f" ({chunks_created} chunks)" if chunks_created else ""
                    st.sidebar.success(f"Successfully uploaded: {uploaded_file.name}{suffix}")
                else:
                    st.sidebar.error(f"Upload failed: {response.text}")
            except requests.exceptions.RequestException as exc:
                st.sidebar.error(f"Could not connect to FastAPI server: {exc}")

# --- 2. Chat Session State & Persistence ---
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"
if "session_input" not in st.session_state:
    st.session_state.session_input = st.session_state.session_id
if "loaded_session_id" not in st.session_state:
    st.session_state.loaded_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def start_new_session():
    new_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.session_id = new_id
    st.session_state.session_input = new_id
    st.session_state.loaded_session_id = new_id
    st.session_state.messages = []


def on_session_input_change():
    cleaned = st.session_state.session_input.strip() or "default_session"
    st.session_state.session_id = cleaned
    st.session_state.loaded_session_id = None
    st.session_state.messages = []


def on_select_saved_session():
    chosen = st.session_state.saved_session_picker
    if chosen != "-- Select a saved session --":
        st.session_state.session_id = chosen
        st.session_state.session_input = chosen
        st.session_state.loaded_session_id = None
        st.session_state.messages = []


st.sidebar.divider()
st.sidebar.header("Chat Session")

# Fetch saved session IDs from PostgreSQL backend
recent_sessions = []
try:
    sess_res = requests.get(f"{API_URL}/documents/sessions/", timeout=(5, 15))
    if sess_res.status_code == 200:
        recent_sessions = sess_res.json()
except requests.exceptions.RequestException:
    recent_sessions = []

if recent_sessions:
    options = ["-- Select a saved session --"] + recent_sessions
    current_idx = (
        options.index(st.session_state.session_id)
        if st.session_state.session_id in options
        else 0
    )
    st.sidebar.selectbox(
        "Previous Sessions",
        options=options,
        index=current_idx,
        key="saved_session_picker",
        on_change=on_select_saved_session,
    )

st.sidebar.text_input(
    "Session ID",
    key="session_input",
    on_change=on_session_input_change,
)
st.sidebar.button("➕ New Session", on_click=start_new_session, use_container_width=True)

# Fetch chat history from PostgreSQL backend whenever the active session changes
if st.session_state.loaded_session_id != st.session_state.session_id:
    try:
        res = requests.get(
            f"{API_URL}/documents/history/{st.session_state.session_id}",
            timeout=(5, 30),
        )
        if res.status_code == 200:
            st.session_state.messages = res.json()
        else:
            st.session_state.messages = []
    except requests.exceptions.RequestException:
        st.session_state.messages = []
    st.session_state.loaded_session_id = st.session_state.session_id


def stream_from_backend(url: str, payload: dict):
    with requests.post(url, json=payload, stream=True, timeout=(5, 120)) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


# --- 3. Chat Interface ---
st.subheader(f"Chat with your Knowledge Base (`{st.session_state.session_id}`)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id,
            }
            streamed_answer = st.write_stream(
                stream_from_backend(f"{API_URL}/documents/agent/stream/", payload)
            )
            final_answer = streamed_answer.strip() if isinstance(streamed_answer, str) else ""
            if not final_answer:
                final_answer = "I'm sorry, I couldn't generate a response."
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to FastAPI server: {e}")
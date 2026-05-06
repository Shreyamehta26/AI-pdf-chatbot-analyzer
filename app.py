import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0f0f12; color: #e2e2e6; }
    .app-header { text-align: center; padding: 2rem 0 1.2rem; }
    .app-header h1 { font-size: 2rem; font-weight: 700; color: #ffffff; margin-bottom: 0.3rem; }
    .app-header p { color: #7a7a8c; font-size: 0.92rem; }
    .status-ok { background: #0d2b1a; border: 1px solid #1a5c38; color: #4ade80; border-radius: 6px; padding: 0.5rem 0.9rem; font-size: 0.84rem; }
    .status-warn { background: #1f1a0a; border: 1px solid #5c430e; color: #fbbf24; border-radius: 6px; padding: 0.5rem 0.9rem; font-size: 0.84rem; }
    .msg-user { background: #1a1a2e; border: 1px solid #2d2d4a; border-radius: 12px 12px 2px 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.9rem; color: #c9c9e0; }
    .msg-ai { background: #111118; border: 1px solid #1e1e30; border-radius: 2px 12px 12px 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.9rem; color: #e2e2ee; line-height: 1.65; }
    .msg-label { font-size: 0.68rem; color: #3d3d55; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem; }
    .stTextInput > div > div > input { background-color: #16161f !important; border: 1px solid #2a2a3d !important; color: #e2e2f0 !important; border-radius: 8px !important; }
    .stButton > button { background: #2828c0 !important; border: none !important; color: #ffffff !important; border-radius: 8px !important; padding: 0.5rem 1.4rem !important; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

def init_session():
    defaults = {"chat_history": [], "chunks": [], "pdf_name": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

def extract_text_from_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    reader = PdfReader(tmp_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    os.unlink(tmp_path)
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_relevant_chunks(query, chunks, top_k=4):
    query_words = set(query.lower().split())
    scored = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        scored.append((score, i, chunk))
    scored.sort(reverse=True)
    return [chunk for _, _, chunk in scored[:top_k]]

def ask_groq(question, context, history, api_key):
    client = Groq(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions based ONLY on the provided document content.
If the answer is not in the document, say 'I could not find that information in the document.'

Document content:
{context}"""
        }
    ]
    for q, a in history[-3:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content

st.markdown("""
<div class="app-header">
    <h1>📄 PDF Chatbot</h1>
    <p>Upload a document. Ask anything inside it. Powered by Groq — free & fast.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Setup")
    groq_key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    groq_api_key = groq_key_input.strip() or os.getenv("GROQ_API_KEY", "")

    if groq_api_key:
        st.markdown('<div class="status-ok">✓ Groq API key loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warn">⚠ No API key found</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        if st.session_state.pdf_name != file_name:
            if not groq_api_key:
                st.error("Add your Groq API key first.")
            else:
                with st.spinner(f"Reading {file_name}..."):
                    try:
                        text = extract_text_from_pdf(file_bytes)
                        chunks = split_text(text)
                        st.session_state.chunks = chunks
                        st.session_state.pdf_name = file_name
                        st.session_state.chat_history = []
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}")

    if st.session_state.pdf_name:
        st.markdown(f'<div class="status-ok">✓ {st.session_state.pdf_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<small style="color:#3d3d55">{len(st.session_state.chunks)} chunks ready</small>', unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if st.session_state.chat_history:
    for question, answer in st.session_state.chat_history:
        st.markdown(f'<div class="msg-user"><div class="msg-label">You</div>{question}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-ai"><div class="msg-label">AI</div>{answer}</div>', unsafe_allow_html=True)
    st.markdown("---")

if not st.session_state.chunks:
    st.info("👈 Add your Groq API key and upload a PDF in the sidebar to start.")
else:
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Ask", placeholder="Ask something about your PDF...", label_visibility="collapsed")
        with col2:
            submit = st.form_submit_button("Send")

    if submit and user_input.strip():
        with st.spinner("Thinking..."):
            try:
                relevant = get_relevant_chunks(user_input.strip(), st.session_state.chunks)
                context = "\n\n".join(relevant)
                answer = ask_groq(user_input.strip(), context, st.session_state.chat_history, groq_api_key)
                st.session_state.chat_history.append((user_input.strip(), answer))
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if not st.session_state.chat_history:
        st.markdown("""
<div style="color:#3d3d55; font-size:0.82rem; font-family:monospace; margin-top:1.5rem;">
Try asking:<br><br>
- What is this document about?<br>
- Summarize the key points.<br>
- What are the main conclusions?
</div>
""", unsafe_allow_html=True)

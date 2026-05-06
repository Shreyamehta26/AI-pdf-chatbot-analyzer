# PDF Chatbot — RAG Project

Ask questions from any PDF using OpenAI + LangChain + ChromaDB.

---

## Setup (5 minutes)

### 1. Create project folder
```bash
mkdir pdf-chatbot
cd pdf-chatbot
```

### 2. Copy these files into it
- `app.py`
- `requirements.txt`
- `.env.example`

### 3. Create your `.env` file
```bash
cp .env.example .env
```
Open `.env` and replace `your_api_key_here` with your real OpenAI key.

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`

---

## How to use

1. Paste your OpenAI API key in the sidebar (or set it in `.env`)
2. Upload any PDF
3. Wait a few seconds for indexing
4. Ask anything about the document

---

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF loading | PyPDF + LangChain |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | ChromaDB (in-memory) |
| LLM | GPT-3.5-turbo |
| Chain | ConversationalRetrievalChain |
| Memory | ConversationBufferMemory |

---

## Notes

- API key can be entered in the sidebar or stored in `.env`
- Chat memory is maintained per session (ask follow-up questions naturally)
- Uploading a new PDF resets the chat
- ChromaDB runs in-memory — no files written to disk

# Enterprise RAG Service | FastAPI + LangChain LCEL

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL_v0.2+-1C3C3C.svg?style=flat&logo=chainlink&logoColor=white)](https://python.langchain.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-ready, interview-grade **Retrieval-Augmented Generation (RAG)** application built with **FastAPI** and modern **LangChain Expression Language (LCEL)**.

It provides a complete pipeline for ingesting unstructured documents, recursive chunking, dense vector similarity retrieval, multi-turn conversational query reformulation, anti-hallucination prompt constraints, source citations attribution, and real-time Server-Sent Events (SSE) token streaming.

Includes an interactive **Web UI Dashboard** and **Swagger API documentation** out-of-the-box.

---

## 🌟 Key Features

- **Modern LangChain LCEL**: Composable, type-safe chains using pipe syntax (`|`), avoiding deprecated legacy `RetrievalQA` chains.
- **Conversational Memory & Reformulation**: Resolves ambiguous follow-up questions in multi-turn dialogues by reformulating queries based on chat history.
- **Strict Anti-Hallucination Guardrails**: Similarity score filtering and prompt constraints prevent speculation when knowledge is absent.
- **Source Citations Attribution**: Every answer is paired with source names, chunk IDs, text snippets, and relevance scores.
- **Real-Time Token Streaming**: High-throughput Server-Sent Events (SSE) streaming endpoint (`/api/v1/rag/stream`).
- **Flexible Vector Store Architecture**: Modular adapter supporting zero-dependency persistent `InMemoryVectorStore`, `ChromaDB`, and `FAISS`.
- **Multi-Provider LLM & Embedding Support**: Drop-in switching between **OpenAI**, **Google Gemini**, local **Ollama**, and an offline **Mock Engine** (works instantly with zero API keys).
- **Built-in Web UI**: A sleek, dark-themed responsive chat interface served directly at `/` for live interactive demos.
- **Production-Ready**: Complete test suite, Pydantic v2 validation, structured logging, Dockerfile, and docker-compose.

---

## 📂 Project Architecture

```
Rag_project/
├── app/
│   ├── main.py                    # FastAPI application factory, CORS, static routes
│   ├── core/
│   │   ├── config.py              # Pydantic v2 settings & environment variables
│   │   └── logging.py             # Structured application logging
│   ├── api/
│   │   ├── schemas.py             # Request & Response Pydantic models
│   │   └── routes/
│   │       ├── health.py          # System health check & vector store metrics
│   │       ├── documents.py       # Ingest text, files, batch docs, clear index
│   │       └── rag.py             # RAG query & SSE real-time streaming routes
│   ├── services/
│   │   ├── document_loader.py     # Loader for txt, md, json, and pdf
│   │   ├── text_splitter.py       # RecursiveCharacterTextSplitter with overlap
│   │   ├── embedding_factory.py   # Factory for OpenAI, Gemini, Ollama, SemanticHash
│   │   ├── vector_store.py        # VectorStore manager with disk persistence
│   │   └── rag_engine.py          # LCEL RAG chain, conversational reformulator
│   └── static/                    # Frontend Demo UI
│       ├── index.html
│       ├── style.css
│       └── app.js
├── data/
│   ├── sample_documents/          # Ready-to-index enterprise sample files
│   └── vector_store_data/         # Persisted vector database storage
├── docs/
│   ├── ARCHITECTURE.md            # Mermaid diagrams & system architecture deep dive
│   └── INTERVIEW_GUIDE.md         # Interview questions, trade-offs, and pitch guide
├── tests/
│   ├── test_document_ingestion.py # Tests for document parsing & chunking
│   ├── test_vector_store.py       # Tests for vector indexing & similarity search
│   └── test_api.py                # Tests for FastAPI endpoints & anti-hallucination
├── .env.example                   # Environment configuration template
├── requirements.txt               # Categorized dependencies
├── Dockerfile                     # Production container build
└── docker-compose.yml             # Container orchestration
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
cd Rag_project
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env`:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

> **Note**: The system defaults to `LLM_PROVIDER="mock"` and `EMBEDDING_PROVIDER="tfidf"`. It is **100% functional immediately out-of-the-box without entering any API keys**! When ready to use OpenAI or Gemini, simply set your API key in `.env`.

### 4. Run the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Interactive Web UI**: Open [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: Open [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🧪 Running Automated Tests

Run the test suite with Python's built-in test runner or pytest:

```bash
python -m unittest discover tests
```

---

## 📡 API Reference

### Document Management

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/documents/text` | Ingest a raw text string with custom metadata |
| `POST` | `/api/v1/documents/batch` | Ingest multiple documents in a single batch request |
| `POST` | `/api/v1/documents/index-file` | Ingest a local file (`.txt`, `.md`, `.json`, `.pdf`) |
| `POST` | `/api/v1/documents/index-directory` | Ingest all supported documents from a directory |
| `POST` | `/api/v1/documents/index-samples` | Seed bundled enterprise sample documents |
| `GET` | `/api/v1/documents/stats` | Retrieve total chunk count, sources, and storage info |
| `DELETE` | `/api/v1/documents` | Reset the vector store and delete all indexed data |

### RAG Inference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/rag/query` | Synchronous RAG query with conversational history & citations |
| `POST` | `/api/v1/rag/stream` | Real-time Server-Sent Events (SSE) token stream |
| `GET` | `/api/v1/health` | System health, model provider status, and chunk metrics |

---

## 💬 Example Request & Response

### Query Request (`POST /api/v1/rag/query`)

```json
{
  "query": "What are the model selection criteria across tiers?",
  "chat_history": [
    {"role": "user", "content": "Tell me about the AI architecture."},
    {"role": "assistant", "content": "The architecture is grounded on verified documents."}
  ],
  "top_k": 3
}
```

### Query Response

```json
{
  "query": "What are the model selection criteria across tiers?",
  "answer": "Tier 1 uses fast models like gpt-4o-mini or llama3.2-3b for intent extraction and filtering, while Tier 2 uses gpt-4o or gemini-1.5-pro for deep synthesis and multi-document consolidation.",
  "citations": [
    {
      "source": "enterprise_ai_guidelines.md",
      "chunk_id": 1,
      "content_snippet": "Tier 1 (Fast / Edge): Models such as gpt-4o-mini or local llama3.2-3b... Tier 2 (Primary Reasoning)...",
      "similarity_score": 0.4821,
      "metadata": {
        "file_type": ".md",
        "total_chunks": 3
      }
    }
  ],
  "execution_time_ms": 34.2,
  "model_used": "gpt-4o-mini",
  "retrieval_count": 1
}
```

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Check logs
docker-compose logs -f
```

---

## 🎓 Preparing for Interviews?

Check out the dedicated interview guides in this repository:
- 📖 [**Architecture Deep Dive**](docs/ARCHITECTURE.md): Mermaid sequence and component diagrams.
- 🎯 [**Interview Pitch & FAQ Guide**](docs/INTERVIEW_GUIDE.md): 60-second elevator pitch, technical trade-offs (chunking, vector DBs, LCEL vs legacy chains), and common interview questions with model answers.

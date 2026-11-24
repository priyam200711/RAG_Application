# Architecture & Design Specifications

## 1. System Overview

The **Enterprise RAG System** is designed following clean layered architecture principles, decoupling HTTP handling, business orchestration, vector retrieval, and model inference.

```mermaid
flowchart TD
    Client["Client / Web UI / External Service"]
    
    subgraph "FastAPI Application Layer"
        Main["FastAPI App Factory (app/main.py)"]
        HealthRoute["Health Check Route (/health)"]
        DocRoute["Document Management Route (/documents)"]
        RAGRoute["RAG Query & SSE Stream Route (/rag)"]
    end
    
    subgraph "Service Layer"
        DocLoader["Document Loader (txt, md, json, pdf)"]
        Chunker["Document Chunker (RecursiveCharacterTextSplitter)"]
        RAGEngine["RAG Engine (LangChain LCEL Pipeline)"]
        EmbedFactory["Embedding Factory (OpenAI / Google / Ollama / TF-IDF)"]
        VStore["Vector Store Service (InMemory / Chroma / FAISS)"]
    end
    
    subgraph "Storage & Persistence"
        DiskStore[("Vector Store Disk Storage (data/vector_store_data/)")]
        SampleDocs[("Sample Knowledge Base (data/sample_documents/)")]
    end
    
    subgraph "LLM & Embedding Providers"
        OpenAI["OpenAI (GPT-4o-mini / Embeddings)"]
        Google["Google Gemini (1.5-flash / Embeddings)"]
        Ollama["Local Ollama (Llama 3.2 / Nomic)"]
        Mock["Offline Grounded Mock Engine"]
    end

    Client -->|HTTP / SSE| Main
    Main --> HealthRoute
    Main --> DocRoute
    Main --> RAGRoute
    
    DocRoute --> DocLoader
    DocLoader --> Chunker
    Chunker --> VStore
    
    RAGRoute --> RAGEngine
    RAGEngine --> VStore
    VStore --> EmbedFactory
    VStore --> DiskStore
    
    EmbedFactory -.-> OpenAI
    EmbedFactory -.-> Google
    EmbedFactory -.-> Ollama
    
    RAGEngine -.-> OpenAI
    RAGEngine -.-> Google
    RAGEngine -.-> Ollama
    RAGEngine -.-> Mock
```

---

## 2. Document Ingestion Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / Admin
    participant API as Documents API
    participant Loader as Document Loader
    participant Chunker as Text Splitter
    participant Embedder as Embedding Model
    participant VDB as Vector Store

    User->>API: POST /api/v1/documents/text or /index-file
    API->>Loader: Parse document text & extract file metadata
    Loader-->>API: LangChain Document(s)
    API->>Chunker: Split into chunks (chunk_size=600, overlap=100)
    Chunker-->>API: Chunks with chunk_id, total_chunks, source metadata
    API->>VDB: add_documents(chunks)
    VDB->>Embedder: embed_documents(texts)
    Embedder-->>VDB: Dense vector representations
    VDB->>VDB: Index vectors & write snapshot to disk
    VDB-->>API: Added count confirmation
    API-->>User: 201 Created (total_chunks_created, status)
```

---

## 3. Conversational RAG Query Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client
    participant API as RAG API (/rag/query or /stream)
    participant Engine as RAG Engine (LCEL)
    participant LLM as Language Model
    participant VDB as Vector Store

    User->>API: POST query + optional chat_history
    API->>Engine: query(query, chat_history, top_k)
    alt Chat History Present
        Engine->>LLM: Reformulate conversational follow-up into standalone query
        LLM-->>Engine: Standalone query string
    else No History
        Engine->>Engine: Use raw query
    end

    Engine->>VDB: similarity_search_with_score(query, k=4)
    VDB-->>Engine: Top-k Chunks + Cosine Similarity Scores
    
    Engine->>Engine: Filter by score threshold & format context string
    Engine->>LLM: Prompt = [System Anti-Hallucination] + [Context] + [Chat History] + [Query]
    LLM-->>Engine: Grounded Answer
    Engine->>Engine: Extract source citations & measure latency
    Engine-->>API: Answer + Citations + Latency (ms)
    API-->>User: 200 OK Response / SSE Token Stream
```

---

## 4. Key Architectural Choices

| Architectural Component | Decision | Rationale |
| :--- | :--- | :--- |
| **Orchestration** | Modern LangChain LCEL (`\|`) | Eliminates legacy deprecated `RetrievalQA` chains; provides native streaming (`.stream()`), composability, and clean typing. |
| **API Framework** | FastAPI + Pydantic v2 | High async throughput, automatic OpenAPI documentation, schema validation, and dependency injection. |
| **Vector Database** | Modular Adapter (`InMemoryVectorStore`, `ChromaDB`, `FAISS`) | Provides instant zero-install out-of-the-box local operation while supporting drop-in switching to Chroma or FAISS for enterprise scale. |
| **Multi-Turn Handling** | Query Reformulation Chain | Resolves core conversational ambiguity (e.g. *"What are its disadvantages?"*) without polluting vector search with unrelated dialogue turns. |
| **Anti-Hallucination** | System Constraint + Fallback | Strictly prohibits speculative responses when retrieved context similarity falls below threshold or lacks necessary information. |

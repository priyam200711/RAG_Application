# RAG Engineering Interview Guide & Pitch Cheatsheet

Use this guide to master your technical interviews when presenting this project.

---

## 1. The 60-Second Elevator Pitch

> *"I designed and built an enterprise-grade Retrieval-Augmented Generation (RAG) service using FastAPI and LangChain LCEL (LangChain Expression Language). The system allows organizations to index heterogeneous knowledge documents (PDFs, Markdown, text, JSON) into a vector store with sliding-window chunking. When queries arrive, the system reformulates multi-turn conversational questions into standalone queries, retrieves top-k relevant context chunks with similarity score filtering, and feeds them into a strictly constrained anti-hallucination prompt. It exposes both standard JSON REST endpoints and real-time Server-Sent Events (SSE) token streaming, and includes complete source attribution with page/chunk references and similarity scores. It also supports multiple LLM and embedding backends (OpenAI, Google Gemini, Ollama, and offline mode)."*

---

## 2. Key Technical Decisions & How to Defend Them

### Q1: Why LangChain LCEL instead of traditional `RetrievalQA` or legacy Chains?
- **Answer**:
  1. **Deprecation & Maintainability**: Legacy LangChain chains like `RetrievalQA` and `ConversationalRetrievalChain` are legacy/deprecated and lack transparency.
  2. **Streaming Support**: LCEL runnables natively implement `.stream()` and `.astream()`, allowing token-by-token streaming over Server-Sent Events without complex callback managers.
  3. **Composability**: Chains are composed using the standard pipe operator (`|`), making prompt engineering, query reformulation, and output parsing modular, debuggable, and testable in isolation.

### Q2: Why did you choose `RecursiveCharacterTextSplitter` and what are the chunking trade-offs?
- **Answer**:
  - **Trade-off**:
    - **Too Small (e.g. < 200 chars)**: Lacks surrounding semantic context; the retrieved snippet may be incomplete or lack subject-verb clarity.
    - **Too Large (e.g. > 1500 chars)**: Introduces noise, pollutes the context window, risks the *"Lost in the Middle"* problem, and increases LLM inference cost.
  - **Our Configuration**:
    - `CHUNK_SIZE = 600` characters, `CHUNK_OVERLAP = 100` characters.
    - Uses recursive separators (`["\n\n", "\n", ". ", " ", ""]`) so natural paragraph and sentence boundaries are preserved whenever possible. Overlap guarantees that sentences crossing chunk boundaries aren't truncated.

### Q3: How do you handle conversational memory in RAG?
- **Answer**:
  - Naive RAG fails in multi-turn dialogues because vector search on *"What are its disadvantages?"* produces terrible semantic similarity against documents.
  - **Our Solution**: We use a two-step LCEL pipeline:
    1. **Contextual Query Reformulation**: An LLM chain takes the chat history and the latest user turn and rewrites it into a standalone question (e.g., *"What are the disadvantages of Vector Database X?"*).
    2. **Retrieval & Answer Generation**: The reformulated query performs the vector search, while the actual answer generation prompt receives both the retrieved context and conversation turns.

### Q4: How do you prevent hallucinations in your RAG system?
- **Answer**:
  1. **Similarity Score Thresholding**: We filter out chunks with similarity scores below `0.05` (or configurable threshold) to prevent inserting irrelevant noise into the prompt.
  2. **Negative Constraint in System Prompt**: Explicit instructions force the LLM to admit ignorance: *"If the provided context does not contain enough information, state clearly that you do not have enough information to answer."*
  3. **Source Attribution**: The response returns verified citations (chunk ID, source title, snippet, and similarity score) so users can audit answers against primary source text.

---

## 3. High-Frequency Interview Questions

### Q5: What is the difference between Cosine Similarity, Dot Product, and Euclidean Distance?
- **Cosine Similarity**: Measures the angle between two vectors ($\cos \theta = \frac{A \cdot B}{\|A\| \|B\|}$), normalized between -1 and 1. Invariant to vector magnitude (document length). **Best for semantic document retrieval.**
- **Dot Product**: Measures angle and magnitude ($A \cdot B$). If embeddings are pre-normalized to unit length ($\|A\| = 1$), Dot Product is mathematically identical to Cosine Similarity but significantly faster to compute.
- **Euclidean Distance (L2)**: Measures straight-line geometric distance ($\sqrt{\sum (A_i - B_i)^2}$). Sensitive to vector magnitude.

### Q6: What is the "Lost in the Middle" phenomenon and how does your architecture address it?
- **Explanation**: Research (Liu et al., 2023) showed that LLMs are significantly better at retrieving and reasoning over information placed at the very beginning or very end of their prompt context; information placed in the middle is frequently overlooked.
- **Mitigation**:
  - Keep `TOP_K` tight (3 to 5 chunks) rather than dumping 20 chunks.
  - Place primary system instructions at the start and end of the prompt context.
  - Order retrieved chunks by descending similarity score or place the highest-scoring chunk closest to the query.

### Q7: What are Server-Sent Events (SSE) and why use them over WebSockets for RAG?
- **SSE vs WebSockets**:
  - **SSE**: Unidirectional (server $\to$ client), built on standard HTTP/1.1 or HTTP/2, automatic reconnection, simple proxy/firewall traversal, minimal overhead. **Ideal for LLM text streaming** where the client sends one prompt and receives a stream of tokens.
  - **WebSockets**: Full-duplex bidirectional communication. Needed for real-time multiplayer editing or voice audio streaming, but adds unnecessary protocol complexity for standard RAG chat.

### Q8: Why use Groq LPU inference for RAG?
- **Answer**:
  - **Ultra-Low Time-to-First-Token (TTFT)**: Groq's Language Processing Units (LPUs) achieve deterministic inference speeds exceeding **500 tokens/second**, dropping TTFT below 150ms compared to 1–2s on traditional GPUs.
  - **Real-Time Streaming UX**: Paired with FastAPI's Server-Sent Events (`/api/v1/rag/stream`), users see instantaneous token generation without perceptual lag.
  - **OpenAI-Compatible & Modular**: Integrates via standard OpenAI-compatible base URL (`https://api.groq.com/openai/v1`), enabling zero vendor lock-in.

---

## 4. Production Scaling & Future Enhancements

When the interviewer asks: *"How would you take this system to enterprise scale?"*, discuss these:

1. **Hybrid Search (Dense + Sparse)**:
   - Combine dense vector embeddings (semantic intent) with sparse BM25 / SPLADE lexical search (exact keyword match for acronyms, part numbers, and error codes) via Reciprocal Rank Fusion (RRF).
2. **Re-Ranking Layer**:
   - Use a Cross-Encoder re-ranker (e.g., Cohere Rerank or `bge-reranker-large`) on the top-20 retrieved candidates to select the top-4 most relevant chunks before prompting the LLM.
3. **Semantic Caching**:
   - Integrate Redis with vector similarity (e.g. GPTCache) to cache previous Q&A pairs; identical or near-identical questions return in < 15ms without invoking the LLM.
4. **Automated Evaluation with RAGAS**:
   - Measure 4 standard metrics in CI/CD:
     - **Faithfulness**: Is the answer grounded in the context?
     - **Answer Relevance**: Did the answer address the question?
     - **Context Precision**: Were the retrieved chunks actually relevant?
     - **Context Recall**: Did we retrieve all the information needed to answer?
5. **Security & Guardrails**:
   - Integrate NeMo Guardrails or Llama Guard for prompt injection detection, PII masking, and output safety checks.

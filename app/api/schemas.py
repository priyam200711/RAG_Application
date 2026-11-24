from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="Message text content")


class SourceCitation(BaseModel):
    source: str = Field(..., description="Document source identifier or file path")
    chunk_id: Optional[int] = Field(None, description="Index of the chunk in the document")
    content_snippet: str = Field(..., description="Snippet of text retrieved from the document")
    similarity_score: Optional[float] = Field(None, description="Similarity or relevance score (higher is closer)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata associated with the chunk")


class IngestTextRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=256, description="Title or name of the document")
    content: str = Field(..., min_length=10, description="Raw text content to be chunked and indexed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value metadata to store")


class IngestBatchItem(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)
    content: str = Field(..., min_length=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestBatchRequest(BaseModel):
    documents: List[IngestBatchItem] = Field(..., min_length=1, description="List of documents to ingest")


class IngestFilePathRequest(BaseModel):
    file_path: str = Field(..., min_length=1, description="Path to a .txt, .md, or .json file to ingest")
    title: Optional[str] = Field(None, description="Optional override title for the document")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str = Field("success", description="Status of ingestion")
    documents_ingested: int = Field(..., description="Number of primary documents processed")
    total_chunks_created: int = Field(..., description="Total number of chunks added to vector store")
    message: str = Field(..., description="Summary message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="The user query or prompt")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional prior conversation turns for multi-turn conversational RAG"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override default number of top matching chunks to retrieve"
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score filter"
    )


class QueryResponse(BaseModel):
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="Generated answer from the RAG pipeline")
    citations: List[SourceCitation] = Field(default_factory=list, description="Retrieved source document snippets")
    execution_time_ms: float = Field(..., description="Total processing time in milliseconds")
    model_used: str = Field(..., description="LLM model identifier used for generation")
    retrieval_count: int = Field(..., description="Number of context chunks retrieved")


class VectorStoreStatsResponse(BaseModel):
    vector_db_type: str
    total_chunks: int
    unique_sources: List[str]
    storage_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_name: str
    version: str
    environment: str
    llm_provider: str
    llm_model: str
    embedding_provider: str
    vector_db_type: str
    total_indexed_chunks: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

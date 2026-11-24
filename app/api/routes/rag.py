import json
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest, QueryResponse
from app.core.config import settings
from app.core.logging import logger
from app.services.rag_engine import rag_engine

router = APIRouter(prefix="/rag", tags=["RAG Query Engine"])


@router.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    """
    Execute a full RAG retrieval and generation cycle.
    Supports conversational memory (pass chat_history), top-k override,
    and returns source citations with similarity scores.
    """
    try:
        answer, citations, execution_time_ms = rag_engine.query(
            query=payload.query,
            chat_history=payload.chat_history,
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold
        )

        return QueryResponse(
            query=payload.query,
            answer=answer,
            citations=citations,
            execution_time_ms=execution_time_ms,
            model_used=settings.LLM_MODEL,
            retrieval_count=len(citations)
        )
    except Exception as e:
        logger.error("RAG Query endpoint error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@router.post("/stream")
def stream_rag(payload: QueryRequest):
    """
    Real-time streaming RAG endpoint using standard Server-Sent Events (SSE).
    Emits citations first, followed by answer tokens, and final completion metadata.
    """
    def event_generator():
        try:
            for packet in rag_engine.stream_query(
                query=payload.query,
                chat_history=payload.chat_history,
                top_k=payload.top_k,
                similarity_threshold=payload.similarity_threshold
            ):
                event_type = packet.get("event", "message")
                data_str = json.dumps(packet.get("data", ""))
                yield f"event: {event_type}\ndata: {data_str}\n\n"
        except Exception as e:
            logger.error("SSE stream error: %s", e)
            err_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

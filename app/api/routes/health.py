from fastapi import APIRouter
from app.api.schemas import HealthResponse
from app.core.config import settings
from app.services.vector_store import vector_store_service

router = APIRouter(prefix="/health", tags=["System Health"])


@router.get("", response_model=HealthResponse)
def get_health():
    """Return system health, configuration, and vector store metrics."""
    stats = vector_store_service.get_stats()
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        vector_db_type=settings.VECTOR_DB_TYPE,
        total_indexed_chunks=stats["total_chunks"]
    )

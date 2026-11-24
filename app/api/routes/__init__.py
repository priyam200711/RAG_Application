from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.rag import router as rag_router

__all__ = ["health_router", "documents_router", "rag_router"]

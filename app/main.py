import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import documents_router, health_router, rag_router
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.services.vector_store import vector_store_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for application startup and shutdown."""
    setup_logging()
    logger.info("Starting %s v%s in %s environment...", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT)
    
    # Auto-seed sample documents on startup if the vector store is empty and samples exist
    try:
        stats = vector_store_service.get_stats()
        if stats["total_chunks"] == 0:
            samples_dir = Path("data") / "sample_documents"
            if samples_dir.exists():
                from app.services.document_loader import DocumentLoaderService
                from app.services.text_splitter import DocumentChunker
                
                docs = DocumentLoaderService.load_directory(str(samples_dir))
                if docs:
                    chunker = DocumentChunker()
                    chunks = chunker.split_documents(docs)
                    vector_store_service.add_documents(chunks)
                    logger.info("Auto-seeded %d sample chunks into vector store on startup.", len(chunks))
    except Exception as e:
        logger.warning("Auto-seed on startup skipped: %s", e)

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)


def create_application() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-grade Retrieval-Augmented Generation (RAG) API built with "
            "LangChain, FastAPI, and modular Vector Stores. Features multi-turn conversational "
            "reformulation, anti-hallucination prompt architecture, source citations, and real-time SSE streaming."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled error processing %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": str(exc) if settings.DEBUG else "An unexpected error occurred."
            }
        )

    # Include API Routers
    api_prefix = settings.API_V1_PREFIX
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(documents_router, prefix=api_prefix)
    app.include_router(rag_router, prefix=api_prefix)

    # Mount Static Files & Frontend UI
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            return {"message": f"{settings.APP_NAME} is running. Visit /docs for API documentation."}

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

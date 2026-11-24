import os
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from app.api.schemas import (
    IngestBatchRequest,
    IngestFilePathRequest,
    IngestResponse,
    IngestTextRequest,
    VectorStoreStatsResponse,
)
from app.core.logging import logger
from app.services.document_loader import DocumentLoaderService
from app.services.text_splitter import DocumentChunker
from app.services.vector_store import vector_store_service

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])
chunker = DocumentChunker()


@router.post("/upload-file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_file(file: UploadFile = File(...)):
    """
    Upload and index a PDF, TXT, Markdown, or JSON file directly.
    PDF pages are extracted, chunked with sliding window, embedded, and saved to the vector database.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        docs = DocumentLoaderService.load_from_bytes(content, filename=file.filename)
        if not docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No readable text could be extracted from '{file.filename}'."
            )

        chunks = chunker.split_documents(docs)
        added_count = vector_store_service.add_documents(chunks)

        return IngestResponse(
            status="success",
            documents_ingested=len(docs),
            total_chunks_created=added_count,
            message=f"File '{file.filename}' ({len(docs)} pages/sections) indexed into {added_count} chunks!"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process uploaded file '%s': %s", file.filename, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )


@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_text(payload: IngestTextRequest):
    """
    Ingest a single text document into the RAG knowledge base.
    Splits the text into overlapping chunks, computes embeddings, and stores in the vector DB.
    """
    try:
        doc = DocumentLoaderService.load_from_text(
            text=payload.content,
            title=payload.title,
            metadata=payload.metadata
        )
        chunks = chunker.split_documents([doc])
        added_count = vector_store_service.add_documents(chunks)

        return IngestResponse(
            status="success",
            documents_ingested=1,
            total_chunks_created=added_count,
            message=f"Document '{payload.title}' successfully ingested into {added_count} chunks."
        )
    except Exception as e:
        logger.error("Failed to ingest text document '%s': %s", payload.title, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.post("/batch", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_batch(payload: IngestBatchRequest):
    """Batch ingest multiple text documents."""
    try:
        all_chunks = []
        for item in payload.documents:
            doc = DocumentLoaderService.load_from_text(
                text=item.content,
                title=item.title,
                metadata=item.metadata
            )
            chunks = chunker.split_documents([doc])
            all_chunks.extend(chunks)

        added_count = vector_store_service.add_documents(all_chunks)

        return IngestResponse(
            status="success",
            documents_ingested=len(payload.documents),
            total_chunks_created=added_count,
            message=f"Batch of {len(payload.documents)} documents indexed into {added_count} chunks."
        )
    except Exception as e:
        logger.error("Failed to ingest batch documents: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch ingestion failed: {str(e)}"
        )


@router.post("/index-file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_file_path(payload: IngestFilePathRequest):
    """Index a local file (.txt, .md, .json, or .pdf) by path."""
    try:
        docs = DocumentLoaderService.load_from_file(
            file_path=payload.file_path,
            title=payload.title
        )
        for d in docs:
            d.metadata.update(payload.metadata)

        chunks = chunker.split_documents(docs)
        added_count = vector_store_service.add_documents(chunks)

        return IngestResponse(
            status="success",
            documents_ingested=len(docs),
            total_chunks_created=added_count,
            message=f"File '{payload.file_path}' successfully indexed into {added_count} chunks."
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error("Failed to index file '%s': %s", payload.file_path, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File indexing failed: {str(e)}"
        )


@router.post("/index-directory", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_directory(
    directory_path: str = Query(..., description="Directory path containing documents"),
    recursive: bool = Query(True, description="Whether to scan subdirectories recursively")
):
    """Scan and index all supported files within a directory."""
    try:
        docs = DocumentLoaderService.load_directory(directory_path, recursive=recursive)
        if not docs:
            return IngestResponse(
                status="success",
                documents_ingested=0,
                total_chunks_created=0,
                message=f"No supported documents found in '{directory_path}'."
            )

        chunks = chunker.split_documents(docs)
        added_count = vector_store_service.add_documents(chunks)

        return IngestResponse(
            status="success",
            documents_ingested=len(docs),
            total_chunks_created=added_count,
            message=f"Indexed {len(docs)} files from '{directory_path}' into {added_count} chunks."
        )
    except NotADirectoryError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error("Failed to index directory '%s': %s", directory_path, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Directory indexing failed: {str(e)}"
        )


@router.post("/index-samples", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_sample_documents():
    """
    Quick helper endpoint: indexes sample enterprise documents bundled with this project.
    Ideal for interview demos.
    """
    samples_dir = Path("data") / "sample_documents"
    if not samples_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample documents folder 'data/sample_documents' does not exist."
        )

    docs = DocumentLoaderService.load_directory(str(samples_dir))
    chunks = chunker.split_documents(docs)
    added_count = vector_store_service.add_documents(chunks)

    return IngestResponse(
        status="success",
        documents_ingested=len(docs),
        total_chunks_created=added_count,
        message=f"Successfully seeded {len(docs)} sample enterprise documents ({added_count} chunks)!"
    )


@router.get("/stats", response_model=VectorStoreStatsResponse)
def get_document_stats():
    """Return count of chunks, unique sources, and storage statistics."""
    stats = vector_store_service.get_stats()
    return VectorStoreStatsResponse(
        vector_db_type=stats["vector_db_type"],
        total_chunks=stats["total_chunks"],
        unique_sources=stats["unique_sources"],
        storage_path=stats["storage_path"]
    )


@router.delete("", status_code=status.HTTP_200_OK)
def clear_documents():
    """Reset the knowledge base and delete all indexed documents."""
    vector_store_service.clear()
    return {"status": "success", "message": "Vector store index cleared successfully."}

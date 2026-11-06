"""API routes."""
import logging

from fastapi import APIRouter, HTTPException

from app.models.requests import QueryRequest
from app.models.responses import MessageResponse, QueryResponse
from app.services import rag_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=MessageResponse)
async def root() -> MessageResponse:
    """Root endpoint (main)."""

    return MessageResponse(message="/ endpoint (main)")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question using retrieval-augmented generation."""

    try:
        result = rag_service.answer_query(
            request.question,
            top_k=request.top_k,
            provider=request.provider,
            prompt=request.prompt,
        )
    except Exception as exc:
        logger.exception("Query failed: %s", request.question)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return QueryResponse(**result)

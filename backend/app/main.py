"""
main.py
FastAPI entrypoint. Deliberately minimal right now — this is scaffolding
so the project is runnable and deployable from day one. Real endpoints
(/ingest, /ask, /history) get added in Phase 3, once the pipeline logic
in ingestion/ is fully wired to the vector store.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("filingiq")

app = FastAPI(
    title="FilingIQ API",
    description="Grounded Q&A over SEC 10-K filings, powered by RAG.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Used by Render/Railway to confirm the container is alive, and by
    your CI pipeline to smoke-test a deploy before marking it successful."""
    logger.info("Health check hit")
    return {"status": "ok", "service": "filingiq-api"}


@app.get("/")
def root():
    return {"message": "FilingIQ API is running. See /docs for endpoints."}

"""
DocuMind Web API - FastAPI Application

Main entry point for the DocuMind REST API.
Provides endpoints for chat, documents, and search.

Run with:
    uvicorn src.documind.api.main:app --reload --port 8000
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE other imports
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import os

from .routes import chat, documents, search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("DocuMind API starting up...")
    logger.info("Initializing services...")
    yield
    # Shutdown
    logger.info("DocuMind API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="DocuMind API",
    description="""
DocuMind is an AI-powered knowledge management system that enables
you to upload documents and ask questions using natural language.

## Features

* **Chat**: Ask questions about your documents and get AI-generated answers with citations
* **Documents**: Upload, manage, and organize your document library
* **Search**: Semantic and hybrid search across all your documents

## Authentication

Currently running in single-tenant mode (no authentication required).
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development (Codespaces, localhost, etc.)
    allow_origin_regex=r"https://.*\.app\.github\.dev",  # GitHub Codespaces
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to all responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred",
            "type": "internal_error"
        }
    )


# Include routers
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(search.router)


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the API status and version.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "documind-api"
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "DocuMind API",
        "version": "1.0.0",
        "description": "AI-powered document Q&A system",
        "docs": "/docs",
        "health": "/health"
    }


# API info endpoint
@app.get("/api", tags=["Root"])
async def api_info():
    """
    API information and available endpoints.
    """
    return {
        "version": "1.0.0",
        "endpoints": {
            "chat": {
                "conversations": "/api/chat/conversations",
                "messages": "/api/chat/conversations/{id}/messages"
            },
            "documents": {
                "list": "/api/documents",
                "upload": "/api/documents",
                "get": "/api/documents/{id}",
                "delete": "/api/documents/{id}",
                "chunks": "/api/documents/{id}/chunks"
            },
            "search": {
                "query": "/api/search",
                "suggestions": "/api/search/suggestions",
                "popular": "/api/search/popular"
            }
        }
    }

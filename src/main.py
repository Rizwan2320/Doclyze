from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
from src.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events - runs at startup and shutdown"""
    logger.info("🚀 Starting Doclyze - Advanced Document Analysis Portal")
    logger.info(f"Log Level          : {settings.LOG_LEVEL}")
    logger.info(f"Embedding Model    : {settings.EMBEDDING_MODEL_NAME}")
    logger.info(f"Groq Model         : {settings.GROQ_MODEL}")
    logger.info(f"Chroma DB Path     : {settings.CHROMA_PERSIST_DIR}")
    
    if settings.LANGSMITH_TRACING:
        logger.info("✅ LangSmith tracing ENABLED")
    else:
        logger.info("ℹ️  LangSmith tracing disabled")

    logger.info("✅ Doclyze is ready to accept requests!")
    yield

    logger.info("🛑 Shutting down Doclyze...")


# Create the FastAPI application
app = FastAPI(
    title="Doclyze",
    description="Production-grade Document Analysis Portal with Advanced RAG",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


from src.api.routes import router as api_router
app.include_router(api_router)


@app.get("/")
async def root():
    """Root welcome endpoint"""
    return {
        "message": "Welcome to Doclyze 👋",
        "status": "running",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check for monitoring / deployment"""
    return {
        "status": "healthy",
        "groq_model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "chunk_size": settings.CHUNK_SIZE,
        "langsmith_enabled": settings.LANGSMITH_TRACING,
    }


if __name__ == "__main__":
    import uvicorn


    logger.info("Starting Uvicorn development server...")
    
    uvicorn.run(
        "src.main:app",      # Important: correct import path  uv run uvicorn src.main:app --reload
        host="127.0.0.1",    # Change to "0.0.0.0" only when deploying
        port=8000,
        reload=True,         # Auto-reload on code changes (perfect for dev)
        log_level=settings.LOG_LEVEL.lower(),
    )
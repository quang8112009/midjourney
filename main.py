"""FastAPI application for the image-generation service."""

import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat_routes import router as chat_router
from app.api.routes import router as generation_router
from app.core.config import settings
from app.core.dependencies import chat_service, model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optionally warm the default model and always release it on shutdown."""
    print("🚀 AI Image Generation API starting...")
    print(f"📝 Model will be loaded on first request: {settings.MODEL_ID}")
    print(f"🔧 Device: {settings.DEVICE}")
    print(f"📐 Default size: {settings.DEFAULT_WIDTH}x{settings.DEFAULT_HEIGHT}")
    print(f"🌐 API docs: http://localhost:{settings.PORT}/docs")
    if not settings.LAZY_LOAD_MODEL:
        await asyncio.to_thread(model_service.load_model)
    try:
        yield
    finally:
        await chat_service.aclose()
        await asyncio.to_thread(model_service.unload_model)
        print("👋 Shutting down...")


app = FastAPI(
    title="AI Image Generation API",
    description="Text-to-image generation and masked editing service inspired by Midjourney",
    version="1.3.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials="*" not in settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(generation_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/")
async def root():
    return {
        "message": "AI Image Generation API",
        "docs": "/docs",
        "version": "1.3.0",
        "chat": "/api/v1/chat",
        "edit": "/api/v1/edit",
        "models": {
            "stable-diffusion": settings.MODEL_ID,
            "pixart-alpha": settings.PIXART_MODEL_ID,
        },
        "device": settings.DEVICE,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        **model_service.status(),
    }


@app.get("/ready")
async def readiness_check():
    status = model_service.status()
    reasons = []
    if settings.DEVICE == "cuda" and not status["cuda_available"]:
        reasons.append("CUDA is configured but unavailable")
    if status["last_load_error"]:
        reasons.append(status["last_load_error"])
    if not os.path.isdir(settings.OUTPUT_DIR) or not os.access(settings.OUTPUT_DIR, os.W_OK):
        reasons.append("Output directory is not writable")
    if reasons:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reasons": reasons, **status},
        )
    return {"status": "ready", **status}


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

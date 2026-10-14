"""
Deep Learning Model Management & Benchmarking API Routes.
"""
from __future__ import annotations

import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.models.registry import get_model_registry

router = APIRouter(prefix="/api/v1/models", tags=["deep-learning-models"])


class BenchmarkRequest(BaseModel):
    iterations: int = Field(default=10, ge=1, le=50, description="Number of inference forward passes to benchmark")


@router.get("/info")
async def get_models_info():
    """Return architecture specifications, layer details, and parameter counts for all active deep learning models."""
    try:
        registry = get_model_registry()
        return registry.get_all_model_specs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to query model registry: {exc}")


@router.post("/benchmark")
async def run_models_benchmark(req: BenchmarkRequest = BenchmarkRequest()):
    """Execute live forward-pass benchmark on server hardware across all deep learning networks."""
    try:
        registry = get_model_registry()
        # Run in threadpool so it doesn't block async event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, registry.run_benchmark, req.iterations)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {exc}")


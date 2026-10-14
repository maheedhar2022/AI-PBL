"""
Evaluation API routes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import settings
from backend.database.db import get_db
from backend.database.models import ExperimentResult
from backend.detection.detector import get_detector
from backend.evaluation.evaluator import EvaluationConfig, run_evaluation

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvalRequest(BaseModel):
    dataset_path: str
    dataset_name: str
    iou_threshold: float = 0.50
    confidence_threshold: float = 0.25
    experiment_name: str = ""
    notes: str = ""


class EvalResultOut(BaseModel):
    id: str
    experiment_name: str
    model_name: str
    dataset_name: str
    run_timestamp: str
    metrics: dict
    config: dict
    notes: str | None = None
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


_running_eval: bool = False


@router.post("/run")
async def run_evaluation_endpoint(
    body: EvalRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    global _running_eval
    if _running_eval:
        raise HTTPException(status_code=409, detail="An evaluation is already running.")

    detector = get_detector()
    if detector is None or not detector.is_loaded:
        from backend.detection.detector import load_detector
        detector = load_detector()

    config = EvaluationConfig(
        model_name=detector.model_name if detector else settings.model_name,
        dataset_name=body.dataset_name,
        dataset_path=body.dataset_path,
        iou_threshold=body.iou_threshold,
        confidence_threshold=body.confidence_threshold,
        experiment_name=body.experiment_name or f"eval_{body.dataset_name}",
        notes=body.notes,
    )

    background_tasks.add_task(_eval_task, config)
    _running_eval = True
    return {"message": "Evaluation started in background.", "status": "running"}


async def _eval_task(config: EvaluationConfig) -> None:
    global _running_eval
    from dataclasses import asdict
    from backend.database.models import ExperimentResult
    from backend.database.db import AsyncSessionLocal
    from loguru import logger
    import traceback

    try:
        # Check path resolution
        p = Path(config.dataset_path)
        if not p.exists() and (settings.dataset_dir / config.dataset_path).exists():
            config.dataset_path = str(settings.dataset_dir / config.dataset_path)

        detector = get_detector()
        if detector is None or not detector.is_loaded:
            from backend.detection.detector import load_detector
            detector = load_detector()

        metrics = await run_evaluation(config, detector=detector)

        exp = ExperimentResult(
            experiment_name=config.experiment_name,
            model_name=config.model_name,
            dataset_name=config.dataset_name,
            metrics=asdict(metrics),
            config={
                "iou_threshold": config.iou_threshold,
                "confidence_threshold": config.confidence_threshold,
            },
            notes=config.notes,
        )
        async with AsyncSessionLocal() as session:
            session.add(exp)
            await session.commit()
    except Exception as exc:
        logger.error(f"Evaluation task failed: {exc}\n{traceback.format_exc()}")
        try:
            err_exp = ExperimentResult(
                experiment_name=config.experiment_name or "eval_error",
                model_name=config.model_name,
                dataset_name=config.dataset_name,
                metrics={
                    "status": "error",
                    "error_message": str(exc),
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "map50": 0.0,
                },
                config={
                    "iou_threshold": config.iou_threshold,
                    "confidence_threshold": config.confidence_threshold,
                },
                notes=f"Error: {exc}",
            )
            async with AsyncSessionLocal() as session:
                session.add(err_exp)
                await session.commit()
        except Exception:
            pass
    finally:
        _running_eval = False


@router.get("/status")
async def eval_status() -> dict:
    return {"running": _running_eval}


@router.get("/results", response_model=List[EvalResultOut])
async def list_results(db: AsyncSession = Depends(get_db)) -> Any:
    rows = (
        await db.execute(
            select(ExperimentResult).order_by(ExperimentResult.run_timestamp.desc())
        )
    ).scalars().all()
    return [
        EvalResultOut(
            id=r.id,
            experiment_name=r.experiment_name,
            model_name=r.model_name,
            dataset_name=r.dataset_name,
            run_timestamp=r.run_timestamp.isoformat() if r.run_timestamp else "",
            metrics=r.metrics,
            config=r.config,
            notes=r.notes,
        )
        for r in rows
    ]


@router.get("/results/{result_id}", response_model=EvalResultOut)
async def get_result(result_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    row = await db.get(ExperimentResult, result_id)
    if not row:
        raise HTTPException(status_code=404, detail="Experiment result not found.")
    return EvalResultOut(
        id=row.id,
        experiment_name=row.experiment_name,
        model_name=row.model_name,
        dataset_name=row.dataset_name,
        run_timestamp=row.run_timestamp.isoformat() if row.run_timestamp else "",
        metrics=row.metrics,
        config=row.config,
        notes=row.notes,
    )


@router.delete("/results/{result_id}")
async def delete_result(result_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    row = await db.get(ExperimentResult, result_id)
    if not row:
        raise HTTPException(status_code=404, detail="Experiment result not found.")
    await db.delete(row)
    await db.commit()
    return {"message": "Experiment result deleted.", "id": result_id}


@router.get("/datasets")
async def list_datasets() -> dict:
    """List available datasets in the configured dataset directory."""
    dataset_root = settings.dataset_dir
    if not dataset_root.exists():
        return {"datasets": [], "message": f"Dataset directory not found: {dataset_root}"}

    datasets = []
    for d in dataset_root.iterdir():
        if d.is_dir():
            img_count = len(list((d / "images" / "val").glob("*.*"))) if (d / "images" / "val").exists() else 0
            datasets.append({
                "name": d.name,
                "path": str(d),
                "image_count_val": img_count,
            })
    return {"datasets": datasets}


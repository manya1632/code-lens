"""
CodeLens REST API.

Endpoints:
  POST /review          — analyze a code snippet
  POST /review/file     — analyze an uploaded file
  GET  /health          — health check
  GET  /model/info      — model metadata

Run with:
  uvicorn src.api.app:app --reload --port 8000
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scripts.infer import CodeLensInference, ReviewReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codelens-api")

app = FastAPI(
    title="CodeLens AI",
    description="Deep Learning–based code review: bug detection, complexity analysis, explainability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once at startup)
_pipeline: Optional[CodeLensInference] = None


@app.on_event("startup")
async def startup_event():
    global _pipeline
    checkpoint = os.getenv("CODELENS_CHECKPOINT", "checkpoints/best_model.pt")
    backbone = os.getenv("CODELENS_BACKBONE", "microsoft/graphcodebert-base")
    explain_method = os.getenv("CODELENS_EXPLAIN", "attention_rollout")

    if not Path(checkpoint).exists():
        logger.warning(f"Checkpoint not found at {checkpoint}. Running without model (mock mode).")
        _pipeline = None
    else:
        logger.info(f"Loading model from {checkpoint}...")
        _pipeline = CodeLensInference(
            checkpoint_path=checkpoint,
            backbone_name=backbone,
            explain_method=explain_method,
        )
        logger.info("Model loaded successfully.")


# ─── Request / Response schemas ───────────────────────────────────────────────

class ReviewRequest(BaseModel):
    code: str = Field(..., description="Source code to review", min_length=10)
    language: str = Field("python", description="Programming language")
    file_name: str = Field("<snippet>", description="Optional filename for display")
    explain_method: Optional[str] = Field(None, description="Override explanation method")


class IssueResponse(BaseModel):
    bug_type: str
    confidence: float
    severity: str
    highlighted_lines: list[int]
    description: str
    suggestion: str


class ReviewResponse(BaseModel):
    file: str
    language: str
    overall_severity: str
    severity_score: float
    issues: list[IssueResponse]
    complexity_before: str
    complexity_after: str
    explanation: str
    token_heatmap: list[list]      # [[token, score], ...]
    score: int
    processing_time_ms: float


# ─── Helpers ──────────────────────────────────────────────────────────────────

def report_to_response(report: ReviewReport, elapsed_ms: float) -> ReviewResponse:
    return ReviewResponse(
        file=report.file,
        language=report.language,
        overall_severity=report.overall_severity,
        severity_score=report.severity_score,
        issues=[IssueResponse(**{
            "bug_type": issue.bug_type,
            "confidence": issue.confidence,
            "severity": issue.severity,
            "highlighted_lines": issue.highlighted_lines,
            "description": issue.description,
            "suggestion": issue.suggestion,
        }) for issue in report.issues],
        complexity_before=report.complexity_before,
        complexity_after=report.complexity_after,
        explanation=report.explanation,
        token_heatmap=[[tok, score] for tok, score in report.token_heatmap],
        score=report.score,
        processing_time_ms=elapsed_ms,
    )


def _mock_review(code: str, language: str, file_name: str) -> ReviewReport:
    """Returns mock output when model not loaded (dev/testing mode)."""
    from scripts.infer import ReviewReport, Issue
    return ReviewReport(
        file=file_name,
        language=language,
        overall_severity="high",
        severity_score=0.72,
        issues=[Issue(
            bug_type="performance",
            confidence=0.91,
            severity="high",
            highlighted_lines=[3, 4, 5],
            description="Nested loops detected — likely O(n²) complexity.",
            suggestion="Use a set or dict for O(1) lookup instead of inner loop.",
        )],
        complexity_before="O(n²)",
        complexity_after="O(n)",
        explanation="[MOCK] Performance issue: nested loops cause quadratic complexity.",
        token_heatmap=[("for", 0.92), ("range", 0.81), ("append", 0.74)],
        score=42,
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _pipeline is not None,
    }


@app.get("/model/info")
async def model_info():
    if _pipeline is None:
        return {"status": "not_loaded", "mode": "mock"}
    return {
        "backbone": _pipeline.model.backbone_name,
        "device": str(_pipeline.device),
        "bug_types": list(__import__("src.model.model", fromlist=["BUG_TYPES"]).BUG_TYPES),
        "complexity_classes": list(__import__("src.model.model", fromlist=["COMPLEXITY_CLASSES"]).COMPLEXITY_CLASSES),
    }


@app.post("/review", response_model=ReviewResponse)
async def review_code(request: ReviewRequest):
    """Analyze a code snippet and return a full review report."""
    t0 = time.perf_counter()

    try:
        if _pipeline is not None:
            report = _pipeline.review(
                code=request.code,
                language=request.language,
                file_name=request.file_name,
            )
        else:
            logger.warning("Model not loaded — returning mock response.")
            report = _mock_review(request.code, request.language, request.file_name)
    except Exception as e:
        logger.exception("Review failed")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = (time.perf_counter() - t0) * 1000
    return report_to_response(report, elapsed)


@app.post("/review/file", response_model=ReviewResponse)
async def review_file(
    file: UploadFile = File(...),
    language: Optional[str] = None,
):
    """Upload a source file for review."""
    content = await file.read()
    try:
        code = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    suffix = Path(file.filename).suffix if file.filename else ".py"
    lang = language or {
        ".py": "python", ".js": "javascript", ".ts": "javascript",
        ".java": "java", ".cpp": "cpp", ".go": "go",
    }.get(suffix, "python")

    t0 = time.perf_counter()
    try:
        if _pipeline is not None:
            report = _pipeline.review(code=code, language=lang, file_name=file.filename or "upload")
        else:
            report = _mock_review(code, lang, file.filename or "upload")
    except Exception as e:
        logger.exception("File review failed")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = (time.perf_counter() - t0) * 1000
    return report_to_response(report, elapsed)

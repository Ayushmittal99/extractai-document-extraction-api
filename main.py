"""ExtractAI - FastAPI application. Run: uvicorn main:app --reload"""
import logging
import uuid
from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import Settings, get_settings
from extractor import ExtractionError, Extractor
from schemas import ErrorResponse, ExtractionRequest, ExtractionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("extractai")

app = FastAPI(
    title="ExtractAI - Intelligent Document Data Extraction API",
    description="Extract validated, structured JSON from resume or invoice text using Gemini.",
    version="1.0.0",
)


@lru_cache
def settings_dep() -> Settings:
    return get_settings()


@lru_cache
def _extractor_singleton() -> Extractor:
    return Extractor(settings_dep())


def get_extractor() -> Extractor:
    """FastAPI dependency. Tests override this with a fake-backed Extractor."""
    return _extractor_singleton()


def _error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(error=code, message=message).model_dump(), headers=headers)


# ---------------------------------------------------------------- middleware / handlers
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Cheap early rejection of huge bodies before they are parsed."""
    limit = settings_dep().max_input_chars * 4 + 1024  # allows JSON escaping + multibyte chars
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit:
        return _error(413, "payload_too_large", "Request body is too large.")
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default 422 echoes the submitted input. We return only field + message
    # so document text can never leak back or into logs.
    details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
    body = ErrorResponse(error="invalid_request", message="Request validation failed.").model_dump()
    body["details"] = details
    return JSONResponse(status_code=422, content=body)


@app.exception_handler(ExtractionError)
async def extraction_handler(request: Request, exc: ExtractionError):
    headers = {"Retry-After": "30"} if exc.status_code == 429 else None
    return _error(exc.status_code, exc.code, exc.message, headers)


@app.exception_handler(Exception)
async def unexpected_handler(request: Request, exc: Exception):
    log.error("unhandled_error type=%s", type(exc).__name__)  # type only: no traceback, no content
    return _error(500, "internal_error", "Unexpected error while processing the request.")


# ---------------------------------------------------------------- endpoints
@app.get("/", summary="API information")
def root():
    return {"name": "ExtractAI", "version": app.version, "docs": "/docs",
            "endpoints": {"health": "GET /health", "extract": "POST /extract"},
            "supported_document_types": ["resume", "invoice"]}


@app.get("/health", summary="Health check")
def health(settings: Settings = Depends(settings_dep)):
    # Does not call Gemini (free + fast). Reports only whether a key is configured, never the key.
    return {"status": "ok", "gemini_key_configured": bool(settings.google_api_key), "model": settings.model}


@app.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extract structured data from resume or invoice text",
    responses={413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse},
               502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
def extract(req: ExtractionRequest, extractor: Extractor = Depends(get_extractor),
            settings: Settings = Depends(settings_dep)):
    # Sync endpoint: FastAPI runs it in a worker thread, so the blocking SDK call does not block the event loop.
    if len(req.text) > settings.max_input_chars:
        raise_too_long = ExtractionError(f"Text is longer than the {settings.max_input_chars}-character limit.")
        raise_too_long.status_code, raise_too_long.code = 413, "payload_too_large"
        raise raise_too_long
    request_id = uuid.uuid4().hex[:8]
    log.info("extract_start id=%s type=%s chars=%d", request_id, req.document_type.value, len(req.text))
    result = extractor.extract(req.document_type, req.text)
    log.info("extract_done id=%s status=%s", request_id, result.status)
    return result

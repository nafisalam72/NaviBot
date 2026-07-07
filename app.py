"""
app.py
------
FastAPI application for the FIFA World Cup 2026
Smart Multilingual Navigation & Crowd Management Assistant.

Endpoints
---------
GET  /             – Serve the main chat UI (index.html)
POST /ask          – Accept a fan query and return an LLM-powered response
GET  /stadiums     – List available stadiums
GET  /sections     – List sections for a given stadium
GET  /health       – Health-check / liveness probe

Security
--------
- CORS is restricted to localhost origins in this demo; tighten for production.
- All user input is sanitised inside ``llm_handler.get_navigation_response``.
- The Groq API key is read from the environment; it never appears in responses.

Usage
-----
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from llm_handler import (
    SUPPORTED_LANGUAGES,
    get_navigation_response,
    sanitise_input,
)
from stadium_data import DEFAULT_STADIUM, STADIUMS, get_stadium_names

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FIFA WC 2026 – NaviBot",
    description=(
        "Smart Multilingual Navigation & Crowd Management Assistant "
        "for FIFA World Cup 2026 stadiums."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS – restrict to localhost origins or allow via env var
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Serve the static frontend
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Schema for the POST /ask request body."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The fan's question in natural language.",
        examples=["Where is the nearest restroom near Section B?"],
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code: 'en', 'es', or 'fr'.",
        examples=["en"],
    )
    stadium: str = Field(
        default=DEFAULT_STADIUM,
        description="Name of the stadium (must match available stadiums).",
        examples=["MetLife Stadium"],
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        """Ensure the language code is supported.

        Parameters
        ----------
        value : str
            The supplied language code.

        Returns
        -------
        str
            The validated language code.

        Raises
        ------
        ValueError
            If the code is not in SUPPORTED_LANGUAGES.
        """
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{value}'. "
                f"Choose from: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        return value

    @field_validator("stadium")
    @classmethod
    def validate_stadium(cls, value: str) -> str:
        """Ensure the requested stadium exists in the registry.

        Parameters
        ----------
        value : str
            The supplied stadium name.

        Returns
        -------
        str
            The validated stadium name.

        Raises
        ------
        ValueError
            If the stadium is not found in STADIUMS.
        """
        if value not in STADIUMS:
            raise ValueError(
                f"Stadium '{value}' not found. "
                f"Available: {get_stadium_names()}"
            )
        return value


class AskResponse(BaseModel):
    """Schema for the POST /ask JSON response."""

    response: str = Field(..., description="Natural-language answer from NaviBot.")
    section: str | None = Field(None, description="Detected section identifier.")
    intent: str = Field(..., description="Classified query intent.")
    cached: bool = Field(..., description="True if the response was served from cache.")
    crowd_level: int | None = Field(None, description="Numeric crowd level (1–10).")
    crowd_category: str | None = Field(
        None, description="Human-readable crowd category: Low / Medium / High."
    )
    stadium: str = Field(..., description="Stadium the query was directed to.")
    language: str = Field(..., description="Language code of the response.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the main frontend HTML page.

    Returns
    -------
    FileResponse
        The ``static/index.html`` file.

    Raises
    ------
    HTTPException
        404 if the static file is not found.
    """
    index_path: Path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index_path))


@app.post("/ask", response_model=AskResponse, summary="Ask NaviBot a stadium question")
async def ask(payload: AskRequest) -> AskResponse:
    """Process a fan's natural-language question and return a multilingual response.

    This endpoint orchestrates:
    1. Input validation (via Pydantic).
    2. Intent extraction and entity recognition.
    3. Mock stadium data retrieval.
    4. LLM prompt construction and Groq API call (with caching).
    5. Structured JSON response.

    Parameters
    ----------
    payload : AskRequest
        The validated request body containing the query, language, and stadium.

    Returns
    -------
    AskResponse
        A structured response including the LLM answer and metadata.

    Raises
    ------
    HTTPException
        422 on validation errors.
        503 if the Groq API key is missing.
        500 on unexpected server errors.
    """
    logger.info(
        "POST /ask | stadium=%s | lang=%s | query=%s",
        payload.stadium,
        payload.language,
        payload.query[:80],
    )
    try:
        result: dict[str, object] = get_navigation_response(
            query=payload.query,
            language_code=payload.language,
            stadium=payload.stadium,
        )
    except EnvironmentError as env_err:
        logger.error("Environment error: %s", env_err)
        raise HTTPException(
            status_code=503,
            detail="Groq API key is not configured. Please set GROQ_API_KEY in .env.",
        ) from env_err
    except Exception as exc:
        logger.error("Unhandled error in /ask: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        ) from exc

    return AskResponse(
        **result,
        stadium=payload.stadium,
        language=payload.language,
    )


@app.get(
    "/stadiums",
    response_model=list[str],
    summary="List all available stadiums",
)
async def list_stadiums() -> list[str]:
    """Return a sorted list of all stadium names in the registry.

    Returns
    -------
    list[str]
        Alphabetically sorted stadium names.
    """
    return get_stadium_names()


@app.get(
    "/sections",
    response_model=dict[str, object],
    summary="List sections for a stadium",
)
async def list_sections(
    stadium: str = Query(
        default=DEFAULT_STADIUM,
        description="Name of the stadium.",
    )
) -> dict[str, object]:
    """Return all section identifiers and their crowd levels for a stadium.

    Parameters
    ----------
    stadium : str
        Query parameter – the stadium name.

    Returns
    -------
    dict
        ``{stadium: str, sections: list[dict]}`` with section id and crowd info.

    Raises
    ------
    HTTPException
        404 if the stadium is not found.
    """
    if stadium not in STADIUMS:
        raise HTTPException(
            status_code=404,
            detail=f"Stadium '{stadium}' not found. Available: {get_stadium_names()}",
        )
    sections_info = [
        {
            "id": sec_id,
            "crowd_level": sec_data["crowd_level"],
            "wheelchair_accessible": sec_data["wheelchair_accessible"],
        }
        for sec_id, sec_data in STADIUMS[stadium].items()
    ]
    return {"stadium": stadium, "sections": sections_info}


@app.get("/health", summary="Health check")
async def health_check() -> JSONResponse:
    """Liveness probe – confirms the server is running.

    Returns
    -------
    JSONResponse
        ``{"status": "ok"}``
    """
    return JSONResponse(content={"status": "ok", "version": "1.0.0"})

# Project: NaviBot - AI Assistant
# Category: [Challenge 4] Smart Stadiums & Tournament
# Target: FIFA World Cup 2026 Crowd Management & Navigation
"""
app.py
------
FastAPI application for the FIFA World Cup 2026
Smart Multilingual Navigation & Crowd Management Assistant.

Endpoints
---------
GET  /             – Serve the main chat UI (index.html)
POST /ask          – Accept a fan stadium query and return an LLM-powered response
GET  /stadiums     – List available FIFA 2026 host stadiums
GET  /sections     – List stadium zones and crowd levels for a given stadium
GET  /health       – Health-check / liveness probe

Security
--------
- CORS is restricted to configurable origins via the ALLOWED_ORIGINS env var.
- All fan input is sanitised inside ``llm_handler.get_navigation_response``.
- Fan stadium queries are hard-capped at 300 characters to prevent token exhaustion.
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
# Constants
# ---------------------------------------------------------------------------

MAX_FAN_QUERY_LENGTH: int = 300
"""Hard limit on fan stadium query length to prevent token exhaustion attacks."""

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

# CORS – restrict to configurable origins via environment variable
allowed_origins_env = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
)
allowed_origins = [
    origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()
]
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


class FanStadiumQueryRequest(BaseModel):
    """Schema for the POST /ask request body.

    Attributes
    ----------
    fan_stadium_query : str
        The fan's natural-language question about the stadium (max 300 chars).
    language : str
        ISO 639-1 language code for the response language.
    stadium_name : str
        Name of the FIFA 2026 host stadium.
    """

    fan_stadium_query: str = Field(
        ...,
        alias="query",
        min_length=1,
        max_length=MAX_FAN_QUERY_LENGTH,
        description="The fan's stadium question in natural language (max 300 chars).",
        examples=["Where is the nearest restroom near Section B?"],
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code: 'en', 'es', or 'fr'.",
        examples=["en"],
    )
    stadium_name: str = Field(
        default=DEFAULT_STADIUM,
        alias="stadium",
        description="Name of the FIFA 2026 host stadium (must match registry).",
        examples=["MetLife Stadium"],
    )

    model_config = {"populate_by_name": True}

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

    @field_validator("stadium_name")
    @classmethod
    def validate_stadium_name(cls, value: str) -> str:
        """Ensure the requested stadium exists in the FIFA 2026 registry.

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


class NaviBotResponse(BaseModel):
    """Schema for the POST /ask JSON response.

    Attributes
    ----------
    fifa_agent_response : str
        Natural-language answer from NaviBot.
    stadium_zone : str or None
        Detected stadium section/zone identifier.
    fan_query_intent : str
        Classified intent of the fan's query.
    cached : bool
        True if the response was served from the in-memory cache.
    crowd_level : int or None
        Numeric crowd level (1–10) for the detected stadium zone.
    crowd_category : str or None
        Human-readable crowd category: Low / Medium / High.
    stadium_name : str
        Stadium the query was directed to.
    language : str
        Language code of the response.
    """

    fifa_agent_response: str = Field(
        ..., alias="response", description="Natural-language answer from NaviBot."
    )
    stadium_zone: str | None = Field(
        None, alias="section", description="Detected stadium zone identifier."
    )
    fan_query_intent: str = Field(
        ..., alias="intent", description="Classified fan query intent."
    )
    cached: bool = Field(
        ..., description="True if the response was served from cache."
    )
    crowd_level: int | None = Field(
        None, description="Numeric crowd level (1–10)."
    )
    crowd_category: str | None = Field(
        None, description="Human-readable crowd category: Low / Medium / High."
    )
    stadium_name: str = Field(
        ..., alias="stadium", description="Stadium the query was directed to."
    )
    language: str = Field(..., description="Language code of the response.")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the main frontend HTML page for NaviBot.

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


@app.post(
    "/ask",
    response_model=NaviBotResponse,
    summary="Ask NaviBot a stadium question",
)
async def ask_navibot(payload: FanStadiumQueryRequest) -> NaviBotResponse:
    """Process a fan's natural-language stadium question and return a response.

    This endpoint orchestrates:
    1. Input validation and length enforcement (via Pydantic, max 300 chars).
    2. Intent extraction and stadium zone entity recognition.
    3. Mock stadium data retrieval for the detected zone.
    4. LLM prompt construction and Groq API call (with SHA-256 caching).
    5. Structured JSON response with crowd metadata.

    Parameters
    ----------
    payload : FanStadiumQueryRequest
        The validated request body containing the fan query, language, and stadium.

    Returns
    -------
    NaviBotResponse
        A structured response including the LLM answer and stadium metadata.

    Raises
    ------
    HTTPException
        422 on validation errors (bad language, stadium, or query too long).
        503 if the Groq API key is missing.
        500 on unexpected server errors.
    """
    logger.info(
        "POST /ask | stadium=%s | lang=%s | fan_query=%s",
        payload.stadium_name,
        payload.language,
        payload.fan_stadium_query[:80],
    )
    try:
        fifa_agent_result: dict[str, object] = get_navigation_response(
            fan_stadium_query=payload.fan_stadium_query,
            language_code=payload.language,
            stadium_name=payload.stadium_name,
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

    return NaviBotResponse(
        **fifa_agent_result,
        stadium_name=payload.stadium_name,
        language=payload.language,
    )


@app.get(
    "/stadiums",
    response_model=list[str],
    summary="List all available FIFA 2026 host stadiums",
)
async def list_stadiums() -> list[str]:
    """Return a sorted list of all FIFA 2026 host stadium names.

    Returns
    -------
    list[str]
        Alphabetically sorted stadium names from the registry.
    """
    return get_stadium_names()


@app.get(
    "/sections",
    response_model=dict[str, object],
    summary="List stadium zones for a given stadium",
)
async def list_stadium_zones(
    stadium: str = Query(
        default=DEFAULT_STADIUM,
        description="Name of the FIFA 2026 host stadium.",
    )
) -> dict[str, object]:
    """Return all stadium zone identifiers and their crowd levels.

    Parameters
    ----------
    stadium : str
        Query parameter – the FIFA 2026 host stadium name.

    Returns
    -------
    dict
        ``{stadium: str, sections: list[dict]}`` with zone id and crowd info.

    Raises
    ------
    HTTPException
        404 if the stadium is not found in the registry.
    """
    if stadium not in STADIUMS:
        raise HTTPException(
            status_code=404,
            detail=f"Stadium '{stadium}' not found. Available: {get_stadium_names()}",
        )
    stadium_zone_info = [
        {
            "id": zone_id,
            "crowd_level": zone_data["crowd_level"],
            "wheelchair_accessible": zone_data["wheelchair_accessible"],
        }
        for zone_id, zone_data in STADIUMS[stadium].items()
    ]
    return {"stadium": stadium, "sections": stadium_zone_info}


@app.get("/health", summary="Health check")
async def health_check() -> JSONResponse:
    """Liveness probe – confirms the NaviBot server is running.

    Returns
    -------
    JSONResponse
        ``{"status": "ok", "version": "1.0.0"}``
    """
    return JSONResponse(content={"status": "ok", "version": "1.0.0"})

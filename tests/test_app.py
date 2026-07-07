# Project: NaviBot - AI Assistant
# Category: [Challenge 4] Smart Stadiums & Tournament
# Target: FIFA World Cup 2026 Crowd Management & Navigation
"""
tests/test_app.py
-----------------
Comprehensive unit tests for NaviBot core logic.

Covers:
- Stadium data helpers (zone lookup, crowd categorisation)
- Input sanitisation (XSS stripping, length cap, type checking)
- Intent classification and stadium zone extraction
- In-memory SHA-256 response caching
- Full LLM pipeline with mocked Groq API
- FastAPI route integration tests
- Edge cases (empty input, oversized input, special characters)

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stubs – prevent real Groq / dotenv imports during tests
# ---------------------------------------------------------------------------

_groq_stub = types.ModuleType("groq")
_groq_stub.Groq = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("groq", _groq_stub)

_dotenv_stub = types.ModuleType("dotenv")
_dotenv_stub.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("dotenv", _dotenv_stub)

import stadium_data as sd  # noqa: E402
import llm_handler as lh  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from app import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test: stadium_data helpers
# ---------------------------------------------------------------------------


class TestStadiumData(unittest.TestCase):
    """Tests for stadium_data.py helper functions."""

    def test_get_stadium_names_returns_sorted_list(self) -> None:
        """get_stadium_names() must return a non-empty sorted list of FIFA 2026 venues."""
        names = sd.get_stadium_names()
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)
        self.assertEqual(names, sorted(names))

    def test_get_section_data_valid_stadium_zone(self) -> None:
        """get_section_data returns a dict for a known stadium/zone."""
        zone_data = sd.get_section_data("MetLife Stadium", "A")
        self.assertIsNotNone(zone_data)
        self.assertIn("nearest_restroom", zone_data)
        self.assertIn("crowd_level", zone_data)
        self.assertIn("wheelchair_accessible", zone_data)

    def test_get_section_data_invalid_stadium(self) -> None:
        """get_section_data returns None for an unknown stadium."""
        result = sd.get_section_data("Unknown Stadium", "A")
        self.assertIsNone(result)

    def test_get_section_data_invalid_zone(self) -> None:
        """get_section_data returns None for an unknown stadium zone."""
        result = sd.get_section_data("MetLife Stadium", "Z")
        self.assertIsNone(result)

    def test_categorise_crowd_low(self) -> None:
        """crowd_level 2 should be categorised as 'Low'."""
        self.assertEqual(sd.categorise_crowd(2), "Low")

    def test_categorise_crowd_medium(self) -> None:
        """crowd_level 5 should be categorised as 'Medium'."""
        self.assertEqual(sd.categorise_crowd(5), "Medium")

    def test_categorise_crowd_high(self) -> None:
        """crowd_level 9 should be categorised as 'High'."""
        self.assertEqual(sd.categorise_crowd(9), "High")

    def test_categorise_crowd_invalid_below_range(self) -> None:
        """crowd_level 0 (below range) should raise ValueError."""
        with self.assertRaises(ValueError):
            sd.categorise_crowd(0)

    def test_categorise_crowd_invalid_above_range(self) -> None:
        """crowd_level 11 (above range) should raise ValueError."""
        with self.assertRaises(ValueError):
            sd.categorise_crowd(11)

    def test_get_section_data_case_insensitive_zone(self) -> None:
        """get_section_data should handle lowercase zone identifiers."""
        zone_data = sd.get_section_data("MetLife Stadium", "a")
        self.assertIsNotNone(zone_data)


# ---------------------------------------------------------------------------
# Test: input sanitisation
# ---------------------------------------------------------------------------


class TestSanitiseInput(unittest.TestCase):
    """Tests for llm_handler.sanitise_input fan query sanitisation."""

    def test_strips_xss_characters(self) -> None:
        """Angle brackets and backticks are stripped from fan input."""
        result = lh.sanitise_input("<script>alert('xss')</script>")
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertNotIn("`", result)

    def test_collapses_whitespace(self) -> None:
        """Multiple spaces are collapsed to single space in fan query."""
        result = lh.sanitise_input("where   is   section  B")
        self.assertEqual(result, "where is section B")

    def test_raises_on_non_string_input(self) -> None:
        """TypeError raised for non-string fan input."""
        with self.assertRaises(TypeError):
            lh.sanitise_input(12345)  # type: ignore[arg-type]

    def test_max_length_truncation_at_300_chars(self) -> None:
        """Fan queries longer than 300 chars are truncated."""
        long_fan_query = "a" * 500
        result = lh.sanitise_input(long_fan_query)
        self.assertLessEqual(len(result), 300)

    def test_empty_string_returns_empty(self) -> None:
        """Empty string fan input returns empty string after sanitisation."""
        result = lh.sanitise_input("")
        self.assertEqual(result, "")

    def test_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only fan input returns empty string after sanitisation."""
        result = lh.sanitise_input("     ")
        self.assertEqual(result, "")

    def test_special_characters_stripped(self) -> None:
        """Curly braces and pipes are stripped from fan input."""
        result = lh.sanitise_input("{test}|query")
        self.assertNotIn("{", result)
        self.assertNotIn("}", result)
        self.assertNotIn("|", result)


# ---------------------------------------------------------------------------
# Test: intent classification and stadium zone extraction
# ---------------------------------------------------------------------------


class TestFanQueryIntentAndZoneExtraction(unittest.TestCase):
    """Tests for classify_fan_query_intent and extract_stadium_zone."""

    def test_classify_restroom_intent(self) -> None:
        """Fan query about bathroom is classified as 'restroom'."""
        self.assertEqual(lh.classify_fan_query_intent("where is the bathroom?"), "restroom")

    def test_classify_crowd_intent(self) -> None:
        """Fan query about crowding is classified as 'crowd'."""
        self.assertEqual(lh.classify_fan_query_intent("is it crowded near section B?"), "crowd")

    def test_classify_food_intent(self) -> None:
        """Fan query about eating is classified as 'food'."""
        self.assertEqual(lh.classify_fan_query_intent("I want to eat something"), "food")

    def test_classify_accessibility_intent(self) -> None:
        """Fan query about wheelchair access is classified as 'accessibility'."""
        self.assertEqual(lh.classify_fan_query_intent("wheelchair ramp near section C"), "accessibility")

    def test_classify_exit_intent(self) -> None:
        """Fan query about exiting is classified as 'exit'."""
        self.assertEqual(lh.classify_fan_query_intent("how do I exit the stadium?"), "exit")

    def test_classify_general_intent(self) -> None:
        """Generic fan greeting is classified as 'general'."""
        self.assertEqual(lh.classify_fan_query_intent("hello there"), "general")

    def test_extract_zone_with_section_keyword(self) -> None:
        """'section A' is correctly extracted as zone 'A'."""
        self.assertEqual(lh.extract_stadium_zone("section A is noisy"), "A")

    def test_extract_zone_with_zone_keyword(self) -> None:
        """'zone D' is correctly extracted as zone 'D'."""
        self.assertEqual(lh.extract_stadium_zone("zone D food stalls"), "D")

    def test_extract_zone_returns_none(self) -> None:
        """Query without zone reference returns None."""
        self.assertIsNone(lh.extract_stadium_zone("where is the main entrance?"))


# ---------------------------------------------------------------------------
# Test: caching logic
# ---------------------------------------------------------------------------


class TestResponseCaching(unittest.TestCase):
    """Tests for in-memory SHA-256 response cache."""

    def setUp(self) -> None:
        """Clear cache before each test."""
        lh._RESPONSE_CACHE.clear()

    def test_cache_miss_on_empty_cache(self) -> None:
        """Cache returns None when empty."""
        key = lh._cache_key("test query", "en", "MetLife Stadium")
        self.assertIsNone(lh.get_cached_response(key))

    def test_cache_store_and_retrieve(self) -> None:
        """Stored NaviBot response can be retrieved by the same cache key."""
        key = lh._cache_key("test query", "en", "MetLife Stadium")
        lh.store_cached_response(key, "cached FIFA agent answer")
        self.assertEqual(lh.get_cached_response(key), "cached FIFA agent answer")

    def test_cache_key_differs_by_language(self) -> None:
        """Same fan query in different languages produces different cache keys."""
        key_en = lh._cache_key("restroom", "en", "MetLife Stadium")
        key_es = lh._cache_key("restroom", "es", "MetLife Stadium")
        self.assertNotEqual(key_en, key_es)

    def test_cache_key_differs_by_stadium(self) -> None:
        """Same fan query for different stadiums produces different cache keys."""
        key1 = lh._cache_key("restroom", "en", "MetLife Stadium")
        key2 = lh._cache_key("restroom", "en", "SoFi Stadium")
        self.assertNotEqual(key1, key2)


# ---------------------------------------------------------------------------
# Test: get_navigation_response with mocked Groq
# ---------------------------------------------------------------------------


class TestGetNavigationResponse(unittest.TestCase):
    """Integration-style tests for get_navigation_response with mocked Groq API."""

    def setUp(self) -> None:
        """Clear cache before each test."""
        lh._RESPONSE_CACHE.clear()

    @patch("llm_handler._get_groq_client")
    def test_returns_structured_response(self, mock_client_factory: MagicMock) -> None:
        """get_navigation_response returns all expected keys."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="The restroom is nearby."))]
        )
        mock_client_factory.return_value = mock_client

        result = lh.get_navigation_response(
            "where is the restroom near section A?", "en", "MetLife Stadium"
        )

        self.assertIn("response", result)
        self.assertIn("section", result)
        self.assertIn("intent", result)
        self.assertIn("cached", result)
        self.assertEqual(result["cached"], False)
        self.assertEqual(result["section"], "A")

    @patch("llm_handler._get_groq_client")
    def test_second_call_uses_cache(self, mock_client_factory: MagicMock) -> None:
        """Second identical fan query is served from cache; API called only once."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Exit is north."))]
        )
        mock_client_factory.return_value = mock_client

        fan_query = "where is exit near section B?"
        lh.get_navigation_response(fan_query, "en", "MetLife Stadium")
        result2 = lh.get_navigation_response(fan_query, "en", "MetLife Stadium")

        self.assertTrue(result2["cached"])
        mock_client.chat.completions.create.assert_called_once()

    def test_empty_fan_query_returns_fallback(self) -> None:
        """Empty fan query returns a polite fallback without calling the API."""
        result = lh.get_navigation_response("   ", "en", "MetLife Stadium")
        self.assertIn("valid", result["response"])

    @patch("llm_handler._get_groq_client")
    def test_spanish_language_accepted(self, mock_client_factory: MagicMock) -> None:
        """Spanish language code is handled without error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="El baño está cerca."))]
        )
        mock_client_factory.return_value = mock_client

        result = lh.get_navigation_response(
            "donde esta el bano seccion A", "es", "MetLife Stadium"
        )
        self.assertEqual(result["response"], "El baño está cerca.")

    @patch("llm_handler._get_groq_client")
    def test_high_crowd_level_detection(self, mock_client_factory: MagicMock) -> None:
        """Section D at MetLife (crowd=9) returns 'High' crowd category."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Section D is very crowded."))]
        )
        mock_client_factory.return_value = mock_client

        result = lh.get_navigation_response(
            "how crowded is section D?", "en", "MetLife Stadium"
        )
        self.assertEqual(result["crowd_category"], "High")
        self.assertEqual(result["crowd_level"], 9)


# ---------------------------------------------------------------------------
# Test: FastAPI Routes
# ---------------------------------------------------------------------------


class TestFastAPIRoutes(unittest.TestCase):
    """Tests for NaviBot FastAPI application endpoints."""

    def test_health_check_returns_ok(self) -> None:
        """GET /health should return 200 OK and status JSON."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_list_stadiums_returns_list(self) -> None:
        """GET /stadiums should return a list of FIFA 2026 host stadiums."""
        response = client.get("/stadiums")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_list_stadium_zones_valid(self) -> None:
        """GET /sections with valid stadium should return zone info."""
        response = client.get("/sections?stadium=MetLife Stadium")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["stadium"], "MetLife Stadium")
        self.assertIn("sections", data)

    def test_list_stadium_zones_invalid(self) -> None:
        """GET /sections with invalid stadium should return 404."""
        response = client.get("/sections?stadium=Invalid")
        self.assertEqual(response.status_code, 404)

    def test_ask_invalid_language_returns_422(self) -> None:
        """POST /ask with unsupported language should return 422."""
        response = client.post(
            "/ask",
            json={"query": "test", "language": "xx", "stadium": "MetLife Stadium"},
        )
        self.assertEqual(response.status_code, 422)

    def test_ask_invalid_stadium_returns_422(self) -> None:
        """POST /ask with unknown stadium should return 422."""
        response = client.post(
            "/ask",
            json={"query": "test", "language": "en", "stadium": "Unknown"},
        )
        self.assertEqual(response.status_code, 422)

    def test_ask_empty_query_returns_422(self) -> None:
        """POST /ask with empty query string should return 422."""
        response = client.post(
            "/ask",
            json={"query": "", "language": "en", "stadium": "MetLife Stadium"},
        )
        self.assertEqual(response.status_code, 422)

    def test_ask_oversized_query_returns_422(self) -> None:
        """POST /ask with fan query exceeding 300 chars should return 422."""
        oversized_fan_query = "a" * 301
        response = client.post(
            "/ask",
            json={"query": oversized_fan_query, "language": "en", "stadium": "MetLife Stadium"},
        )
        self.assertEqual(response.status_code, 422)

    @patch("app.get_navigation_response")
    def test_ask_valid_fan_query(self, mock_get_nav: MagicMock) -> None:
        """POST /ask with valid fan query should return 200 with structured response."""
        mock_get_nav.return_value = {
            "response": "Here is the restroom.",
            "section": "A",
            "intent": "restroom",
            "cached": False,
            "crowd_level": 3,
            "crowd_category": "Low",
        }
        response = client.post(
            "/ask",
            json={"query": "where is restroom", "language": "en", "stadium": "MetLife Stadium"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["intent"], "restroom")
        self.assertEqual(data["language"], "en")


if __name__ == "__main__":
    unittest.main()

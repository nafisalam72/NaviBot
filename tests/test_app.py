"""
tests/test_app.py
-----------------
Unit tests for NaviBot core logic.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stubs – prevent real Groq / dotenv imports during tests
# ---------------------------------------------------------------------------

# Stub `groq` module
_groq_stub = types.ModuleType("groq")
_groq_stub.Groq = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("groq", _groq_stub)

# Stub `dotenv`
_dotenv_stub = types.ModuleType("dotenv")
_dotenv_stub.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("dotenv", _dotenv_stub)

# Now safe to import our modules
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

    def test_get_stadium_names_returns_list(self) -> None:
        """get_stadium_names() must return a non-empty sorted list."""
        names = sd.get_stadium_names()
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)
        self.assertEqual(names, sorted(names))

    def test_get_section_data_valid(self) -> None:
        """get_section_data returns a dict for a known stadium/section."""
        data = sd.get_section_data("MetLife Stadium", "A")
        self.assertIsNotNone(data)
        self.assertIn("nearest_restroom", data)
        self.assertIn("crowd_level", data)
        self.assertIn("wheelchair_accessible", data)

    def test_get_section_data_invalid_stadium(self) -> None:
        """get_section_data returns None for an unknown stadium."""
        result = sd.get_section_data("Unknown Stadium", "A")
        self.assertIsNone(result)

    def test_get_section_data_invalid_section(self) -> None:
        """get_section_data returns None for an unknown section."""
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

    def test_categorise_crowd_invalid(self) -> None:
        """crowd_level outside 1-10 should raise ValueError."""
        with self.assertRaises(ValueError):
            sd.categorise_crowd(0)
        with self.assertRaises(ValueError):
            sd.categorise_crowd(11)


# ---------------------------------------------------------------------------
# Test: input sanitisation
# ---------------------------------------------------------------------------


class TestSanitiseInput(unittest.TestCase):
    """Tests for llm_handler.sanitise_input."""

    def test_strips_disallowed_chars(self) -> None:
        """Angle brackets and backticks are stripped."""
        result = lh.sanitise_input("<script>alert('xss')</script>")
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertNotIn("`", result)

    def test_collapses_whitespace(self) -> None:
        """Multiple spaces are collapsed to single space."""
        result = lh.sanitise_input("where   is   section  B")
        self.assertEqual(result, "where is section B")

    def test_raises_on_non_string(self) -> None:
        """TypeError raised for non-string input."""
        with self.assertRaises(TypeError):
            lh.sanitise_input(12345)  # type: ignore[arg-type]

    def test_max_length_truncation(self) -> None:
        """Inputs longer than 1000 chars are truncated."""
        long_input = "a" * 2000
        result = lh.sanitise_input(long_input)
        self.assertLessEqual(len(result), 1000)


# ---------------------------------------------------------------------------
# Test: intent classification and section extraction
# ---------------------------------------------------------------------------


class TestIntentAndSection(unittest.TestCase):
    """Tests for classify_intent and extract_section."""

    def test_classify_restroom(self) -> None:
        self.assertEqual(lh.classify_intent("where is the bathroom?"), "restroom")

    def test_classify_crowd(self) -> None:
        self.assertEqual(lh.classify_intent("is it crowded near section B?"), "crowd")

    def test_classify_food(self) -> None:
        self.assertEqual(lh.classify_intent("I want to eat something"), "food")

    def test_classify_accessibility(self) -> None:
        self.assertEqual(lh.classify_intent("wheelchair ramp near section C"), "accessibility")

    def test_classify_exit(self) -> None:
        self.assertEqual(lh.classify_intent("how do I exit the stadium?"), "exit")

    def test_classify_general(self) -> None:
        self.assertEqual(lh.classify_intent("hello there"), "general")

    def test_extract_section_with_keyword(self) -> None:
        self.assertEqual(lh.extract_section("section A is noisy"), "A")

    def test_extract_section_zone(self) -> None:
        self.assertEqual(lh.extract_section("zone D food stalls"), "D")

    def test_extract_section_none(self) -> None:
        self.assertIsNone(lh.extract_section("where is the main entrance?"))


# ---------------------------------------------------------------------------
# Test: caching logic
# ---------------------------------------------------------------------------


class TestCaching(unittest.TestCase):
    """Tests for in-memory response cache."""

    def setUp(self) -> None:
        """Clear cache before each test."""
        lh._RESPONSE_CACHE.clear()

    def test_cache_miss_on_empty(self) -> None:
        """Cache returns None when empty."""
        key = lh._cache_key("test query", "en", "MetLife Stadium")
        self.assertIsNone(lh.get_cached_response(key))

    def test_cache_store_and_retrieve(self) -> None:
        """Stored response can be retrieved by the same key."""
        key = lh._cache_key("test query", "en", "MetLife Stadium")
        lh.store_cached_response(key, "cached answer")
        self.assertEqual(lh.get_cached_response(key), "cached answer")

    def test_cache_key_differs_by_language(self) -> None:
        """Same query in different languages produces different keys."""
        key_en = lh._cache_key("restroom", "en", "MetLife Stadium")
        key_es = lh._cache_key("restroom", "es", "MetLife Stadium")
        self.assertNotEqual(key_en, key_es)

    def test_cache_key_differs_by_stadium(self) -> None:
        """Same query for different stadiums produces different keys."""
        key1 = lh._cache_key("restroom", "en", "MetLife Stadium")
        key2 = lh._cache_key("restroom", "en", "SoFi Stadium")
        self.assertNotEqual(key1, key2)


# ---------------------------------------------------------------------------
# Test: get_navigation_response with mocked Groq
# ---------------------------------------------------------------------------


class TestGetNavigationResponse(unittest.TestCase):
    """Integration-style tests for get_navigation_response with mocked API."""

    def setUp(self) -> None:
        """Clear cache before each test."""
        lh._RESPONSE_CACHE.clear()

    @patch("llm_handler._get_groq_client")
    def test_returns_response_dict(self, mock_client_factory: MagicMock) -> None:
        """get_navigation_response returns expected keys."""
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
        """Second identical call is served from cache; API called only once."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Exit is north."))]
        )
        mock_client_factory.return_value = mock_client

        query = "where is exit near section B?"
        lh.get_navigation_response(query, "en", "MetLife Stadium")
        result2 = lh.get_navigation_response(query, "en", "MetLife Stadium")

        self.assertTrue(result2["cached"])
        mock_client.chat.completions.create.assert_called_once()

    def test_empty_query_returns_prompt(self) -> None:
        """Empty query returns a polite fallback without calling the API."""
        result = lh.get_navigation_response("   ", "en", "MetLife Stadium")
        self.assertIn("valid question", result["response"])

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
    def test_high_crowd_level_present(self, mock_client_factory: MagicMock) -> None:
        """Section D at MetLife (crowd=9) returns 'High' category."""
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

class TestAppRoutes(unittest.TestCase):
    """Tests for FastAPI application endpoints."""

    def test_health_check(self) -> None:
        """GET /health should return 200 OK and status JSON."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_list_stadiums(self) -> None:
        """GET /stadiums should return list of stadiums."""
        response = client.get("/stadiums")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_list_sections_valid(self) -> None:
        """GET /sections with valid stadium should return sections info."""
        response = client.get("/sections?stadium=MetLife Stadium")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["stadium"], "MetLife Stadium")
        self.assertIn("sections", data)

    def test_list_sections_invalid(self) -> None:
        """GET /sections with invalid stadium should return 404."""
        response = client.get("/sections?stadium=Invalid")
        self.assertEqual(response.status_code, 404)

    def test_ask_invalid_language(self) -> None:
        """POST /ask with invalid language should return 422."""
        response = client.post("/ask", json={"query": "test", "language": "xx", "stadium": "MetLife Stadium"})
        self.assertEqual(response.status_code, 422)
        
    def test_ask_invalid_stadium(self) -> None:
        """POST /ask with invalid stadium should return 422."""
        response = client.post("/ask", json={"query": "test", "language": "en", "stadium": "Unknown"})
        self.assertEqual(response.status_code, 422)

    @patch("app.get_navigation_response")
    def test_ask_valid(self, mock_get_nav: MagicMock) -> None:
        """POST /ask with valid payload should return 200."""
        mock_get_nav.return_value = {
            "response": "Here is the restroom.",
            "section": "A",
            "intent": "restroom",
            "cached": False,
            "crowd_level": 3,
            "crowd_category": "Low",
        }
        response = client.post("/ask", json={"query": "where is restroom", "language": "en", "stadium": "MetLife Stadium"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["intent"], "restroom")
        self.assertEqual(data["language"], "en")


if __name__ == "__main__":
    unittest.main()

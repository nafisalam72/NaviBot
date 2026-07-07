# NaviBot – FIFA World Cup 2026 Smart Multilingual Navigation & Crowd Management Assistant

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq LLM](https://img.shields.io/badge/LLM-Groq%20Llama3--70B-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Chosen Vertical

> **Smart Multilingual Navigation & Crowd Management Assistant**

This project was built for **HACK2SKILL Challenge 4 – FIFA World Cup 2026**. The chosen vertical is the **Smart Multilingual Navigation & Crowd Management Assistant**, which uses a Groq-powered LLM (Llama 3 70B) to help stadium fans navigate venues, find facilities, avoid crowds, and receive guidance in English, Spanish, or French.

---

## Problem Statement Alignment

The FIFA World Cup 2026 will be hosted across 16 cities in the USA, Canada, and Mexico — with stadiums holding 60,000–100,000 fans each. This creates real operational challenges:

| Real-World Problem | How NaviBot Solves It |
|---|---|
| Fan can't find the nearest restroom | Asks NaviBot → gets exact block/gate directions for their section |
| Exit routes are congested post-match | NaviBot checks crowd level and suggests low-density alternative exits |
| Wheelchair user stuck in inaccessible zone | NaviBot flags non-accessible sections and redirects to accessible routes |
| International fan doesn't speak English | Language toggle (EN / ES / FR) sends entire response in chosen language |
| Food stall queues are unknown | NaviBot lists food stalls per section with crowd context |
| Fan asks same question twice | In-memory cache returns instant answer, zero API cost |

---

## Architecture & Approach

```
Fan's Query (any language)
        │
        ▼
  Input Sanitisation  ──── strips injection chars
        │
        ▼
  Intent Classifier   ──── crowd / restroom / exit / food / accessibility / general
        │
        ▼
  Section Extractor   ──── regex → Section A–E
        │
        ▼
  Cache Look-up       ──── SHA-256(query+lang+stadium) hit? → return cached
        │ miss
        ▼
  Stadium Data Fetch  ──── mock dict → crowd_level, restroom, exit, food_stalls, wheelchair
        │
        ▼
  Prompt Builder      ──── system role + structured context block + user question
        │
        ▼
  Groq API (Llama 3 70B)  ──── lightning-fast inference (<1s)
        │
        ▼
  Structured Response ──── JSON: response + section + intent + cached + crowd metadata
        │
        ▼
  Frontend (HTML/JS)  ──── renders bubble + crowd badge + cache indicator
```

### Why Groq?
Groq's LPU (Language Processing Unit) delivers **sub-second inference** even for 70B parameter models. This is critical in a stadium context where fans need instant answers while navigating crowded concourses.

---

## How the Solution Works – Step-by-Step User Flow

1. Fan opens the web app and sees the stadium chat interface.
2. Fan selects their **stadium** (MetLife / AT&T / SoFi) from the dropdown.
3. Fan picks their **language** (EN / ES / FR) using the header toggle.
4. Fan types a question (or clicks a quick-prompt button).
5. The frontend POSTs `{ query, language, stadium }` to `/ask`.
6. The backend sanitises input, detects section and intent via regex.
7. Mock stadium data for the section is fetched from `stadium_data.py`.
8. A detailed system+user prompt is constructed and sent to Groq's Llama 3.
9. The LLM returns a friendly, contextual, multilingual response.
10. The response is cached; subsequent identical queries skip the API.
11. The frontend renders the answer with crowd-level badge and cache indicator.

---

## Features

- **Multilingual**: English, Spanish, French responses via Groq LLM
- **Crowd Awareness**: 1–10 crowd scale, auto-suggests alternative routes when High (7+)
- **Accessibility**: Flags non-wheelchair-accessible sections; recommends alternatives
- **Smart Caching**: SHA-256-keyed in-memory cache eliminates duplicate API calls
- **Prompt Injection Protection**: Strips `<`, `>`, `{`, `}`, `` ` ``, `\`, `[`, `]`, `|` from user input
- **Section Grid**: Visual crowd heatmap in the sidebar (Low=green, Medium=yellow, High=red)
- **Quick Prompts**: One-click pre-built questions for restrooms, exits, food, crowd, accessibility
- **ARIA Accessibility**: All inputs and buttons carry `aria-label`; `aria-live` on chat log

---

## Project Structure

```
Fifa/
├── app.py              # FastAPI server – /ask, /stadiums, /sections, /health
├── llm_handler.py      # Groq API integration, prompt engineering, caching
├── stadium_data.py     # Mock static data for MetLife, AT&T, SoFi stadiums
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes .env, __pycache__, .pytest_cache
├── .env.example        # Template – rename to .env and add GROQ_API_KEY
├── static/
│   └── index.html      # Premium dark-theme chat UI (HTML + CSS + JS)
└── tests/
    └── test_app.py     # 15+ unit tests (mocked Groq client)
```

---

## Setup & Running

### Prerequisites
- Python 3.11+
- A Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### 1 – Clone & Install

```bash
git clone https://github.com/<your-username>/fifa-navibot.git
cd fifa-navibot
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2 – Configure API Key

```bash
cp .env.example .env
# Edit .env and set:
# GROQ_API_KEY=gsk_...your_real_key...
```

> ⚠ **Never commit your `.env` file.** It is listed in `.gitignore`.

### 3 – Run the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

### 4 – Run Tests

```bash
pytest tests/ -v
```

All tests mock the Groq client — no real API calls are made during testing.

---

## API Reference

### `POST /ask`
```json
{
  "query": "Where is the nearest restroom near section B?",
  "language": "en",
  "stadium": "MetLife Stadium"
}
```
**Response:**
```json
{
  "response": "The nearest restroom to Section B is at Restroom Block B2...",
  "section": "B",
  "intent": "restroom",
  "cached": false,
  "crowd_level": 7,
  "crowd_category": "High",
  "stadium": "MetLife Stadium",
  "language": "en"
}
```

### `GET /stadiums` – List all stadiums
### `GET /sections?stadium=MetLife+Stadium` – Section crowd overview
### `GET /health` – Liveness probe

---

## Assumptions

1. **Mock Data**: Real-time IoT/GPS crowd data is unavailable; crowd levels (1–10) are static mock values. A production system would ingest live feeds from stadium sensor platforms.
2. **Stadium Sections**: Each stadium has 5 representative sections (A–E). Real venues have dozens more.
3. **Language Detection**: Language is selected explicitly by the user; automatic detection is not implemented to keep scope manageable.
4. **In-Memory Cache**: The cache resets on server restart. A production system would use Redis.
5. **Authentication**: No user authentication is implemented; this is a public demo assistant.
6. **CORS**: Set to `allow_origins=["*"]` for demo purposes; restrict to your domain in production.

---

## Security

- `GROQ_API_KEY` is stored in `.env` and loaded via `python-dotenv`. It never appears in responses or logs.
- All user inputs are sanitised (strips injection characters) before LLM prompt construction.
- `.env` is listed in `.gitignore` to prevent accidental commits.

---

## Submission Notes

- ✅ **GitHub repository is PUBLIC**
- ✅ **Contains ONLY ONE branch** (main)
- ✅ **Repository size is under 10 MB** (no large binary files)
- ✅ **Groq API key** must be placed in `.env` file as `GROQ_API_KEY=<your_key>` to run the project
- ✅ **LLM**: Groq API with `llama3-70b-8192` for lightning-fast inference
- ✅ **Backend**: Python + FastAPI
- ✅ **Frontend**: Vanilla HTML/CSS/JS (no heavy framework)

---

## License

MIT © 2026

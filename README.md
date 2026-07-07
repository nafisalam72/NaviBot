# NaviBot – FIFA World Cup 2026 Smart Multilingual Navigation & Crowd Management Assistant

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq LLM](https://img.shields.io/badge/LLM-Groq%20Llama3--70B-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Problem Statement Alignment (Challenge 4)

NaviBot is a GenAI-enabled solution strictly designed to **enhance stadium operations and the overall tournament experience for fans, organizers, volunteers, or venue staff.** This solution actively leverages Generative AI to improve **navigation, crowd management, accessibility, transportation, sustainability, multilingual assistance, operational intelligence, and real-time decision support** during the **FIFA World Cup 2026.**

## 💡 How the Solution Works
1. **Multilingual Assistance:** Detects language and responds in English, Spanish, or French to assist international fans.
2. **Crowd Management:** Reads zone density (1-10) and redirects fans to alternative gates.
3. **Accessibility:** Flags wheelchair-inaccessible zones and provides alternative accessible routes.
4. **Efficiency:** Uses an in-memory SHA-256 cache to prevent duplicate API calls, returning instant answers.

## ⚙️ Execution Flow
1. User input is strictly sanitised to prevent prompt injection.
2. Regex classifies the intent (restroom, food, exit, crowd).
3. The context is passed to the Groq API (Llama-3 70B) with strict system instructions limiting it ONLY to FIFA 2026 operations.
4. Response is displayed in an accessible (ARIA-compliant) UI.

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

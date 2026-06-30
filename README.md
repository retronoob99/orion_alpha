# Orion Alpha

An open-source, agentic investment research system for pre-seed venture capital investors. Give it a company name, and it autonomously researches founders, funding history, financials, market conditions, and recent news — then outputs a structured **INVEST / PASS / WATCH** recommendation with full reasoning.

Built as a free, self-hosted alternative to tools like Harmonic.ai, using only free-tier APIs and open-source models.

---

## What it does

- Accepts a company name or URL as input
- Runs 6 autonomous research tools in sequence:
  - Founder & team background (LinkedIn, people data)
  - Funding history (Crunchbase, VC funding rounds)
  - Financial signals (SEC EDGAR filings)
  - Market size & competitors (web search)
  - Macro conditions (FRED — interest rates, inflation, GDP)
  - Recent news (Tavily)
- Scores findings against configurable weights (founder quality, market, financials, macro, funding)
- Outputs a structured JSON report with decision, confidence score, strengths, risks, and full reasoning
- Displays everything in a clean analyst-style dashboard

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python |
| Agent orchestration | LangChain |
| RAG pipeline | LlamaIndex + ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Groq (Llama 3 70B) — free tier |
| Web search | Tavily — free tier |
| Macro data | FRED API — free |
| Financial filings | SEC EDGAR — no API key needed |
| Frontend | Vite + React (designed in Google Stitch) |

Everything runs on **free-tier APIs only** — no credit card required.

---

## Datasets

Vektor uses two Kaggle datasets as its base knowledge index. Download both before running ingestion.

### Dataset 1 — Crunchbase Startup Investments
Covers startup funding, sectors, investors, and company metadata.

```
https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase
```

Download and save as:
```
data/raw/crunchbase_investments.csv
```

### Dataset 2 — VC Startup Investments (justinas)
Covers funding rounds, investment details, and founder/people data. Download the full dataset and use these three files:

```
https://www.kaggle.com/datasets/justinas/startup-investments
```

From the unzipped folder, copy:

| File | Save as |
|---|---|
| `funding_rounds.csv` | `data/raw/vc_funding_rounds.csv` |
| `people.csv` | `data/raw/people.csv` |

> Skip `investments.csv`, `objects.csv`, and other files from this dataset — they overlap with Dataset 1 or are too large for the prototype.

---

## Project structure

```
vektor/
├── README.md
├── .env                          # API keys (never commit this)
├── .env.example                  # Safe template to commit
├── requirements.txt
│
├── api/
│   ├── main.py                   # FastAPI app entry point
│   ├── models.py                 # Pydantic request/response schemas
│   └── routes/
│       └── research.py           # POST /api/v1/research/ endpoint
│
├── agent/
│   ├── orchestrator.py           # LangChain AgentExecutor
│   ├── prompts/
│   │   └── system_prompt.txt     # Agent system prompt
│   └── tools/
│       ├── founder_tool.py       # Founder & team research
│       ├── funding_tool.py       # Funding history research
│       ├── financials_tool.py    # SEC EDGAR financial signals
│       ├── market_tool.py        # Market size & competitors
│       ├── macro_tool.py         # FRED macro conditions
│       └── news_tool.py          # Recent news via Tavily
│
├── rag/
│   ├── pipeline.py               # RAG query pipeline
│   └── vector_store.py           # ChromaDB connection
│
├── data/
│   ├── raw/                      # Place your CSV files here
│   │   ├── crunchbase_investments.csv
│   │   ├── vc_funding_rounds.csv
│   │   └── people.csv
│   ├── processed/
│   │   └── embeddings_index/     # Auto-created by ingest.py
│   └── ingest.py                 # One-time data ingestion script
│
├── scorer/
│   └── recommendation.py         # INVEST / PASS / WATCH logic
│
├── output/
│   └── report_builder.py         # Final report assembler
│
└── frontend/                     # React frontend (Vite)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/vektor.git
cd vektor
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Set up environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
FRED_API_KEY=your_fred_api_key_here
```

Where to get each key (all free, no credit card):

| Key | Link |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com |
| `TAVILY_API_KEY` | https://app.tavily.com |
| `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api |

### 5. Download datasets

Download both Kaggle datasets linked above and place the CSV files under `data/raw/` as described in the Datasets section. You will need a free Kaggle account to download them.

### 6. Run data ingestion

This step embeds all three CSV files into ChromaDB. Run it once before starting the server. It takes a few minutes depending on your machine.

```bash
python -m data.ingest
```

When complete you should see:

```
Ingestion complete. XXXXX documents indexed (includes founder/people data).
```

A `data/processed/embeddings_index/` folder will be created automatically containing `chroma.sqlite3` and a UUID vector folder. Do not delete or modify this folder manually.

### 7. Start the backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at: `http://localhost:8000`
API docs (Swagger UI): `http://localhost:8000/docs`

### 8. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## Usage

Open `http://localhost:5173` in your browser, type a company name (e.g. "Notion", "Stripe", "Rippling"), and hit Enter. The system will:

1. Run all 6 research tools in sequence (takes 30–90 seconds)
2. Score the findings against configurable weights
3. Return a structured dashboard with INVEST / PASS / WATCH verdict

You can also call the API directly:

```bash
curl -X POST http://localhost:8000/api/v1/research/ \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Notion"}'
```

---

## Configuration

All scoring weights are configurable in `.env`:

```env
WEIGHT_FOUNDER=0.30       # Founder quality (30%)
WEIGHT_MARKET=0.25        # Market size & competition (25%)
WEIGHT_FINANCIALS=0.20    # Financial signals (20%)
WEIGHT_MACRO=0.15         # Macro conditions (15%)
WEIGHT_FUNDING=0.10       # Funding history (10%)
```

Adjust these to match your fund's investment thesis.

---

## Limitations

This is a 2-day prototype with known limitations:

- Pre-seed companies with little public data will return weaker signals — Tavily web search will do most of the heavy lifting for newer companies
- SEC EDGAR filings are only available for companies that have filed publicly — most pre-seed targets will show "no filings found"
- people.csv founder data is from a 2013–2015 Crunchbase snapshot — recent founders may not be indexed
- Research runs take 30–90 seconds due to sequential tool calls and LLM inference
- The Groq free tier has rate limits — avoid running multiple concurrent research requests

---

## Roadmap

- Async parallel tool execution (cut research time from ~60s to ~15s)
- URL input support (scrape and analyze a company's own website)
- PDF export of research reports
- Batch research mode (research multiple companies from a CSV)
- Feedback loop — mark recommendations as correct/incorrect to improve scoring weights over time
- Swap Groq/Llama3 for Claude API for higher reasoning quality

---

## License

MIT — use freely, attribution appreciated.

---

Built in 2 days as a solo prototype. Contributions welcome.

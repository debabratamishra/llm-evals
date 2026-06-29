# LLM Evaluation Framework

A full-stack framework for benchmarking and evaluating Large Language Models against golden Q&A datasets. Run evaluations against local or cloud-hosted models, score outputs with multiple metrics, and analyse results in an interactive dashboard.

---

## Features

- **Multi-backend LLM support** via [LiteLLM](https://github.com/BerriAI/litellm): Ollama (local & cloud), OpenRouter — with per-run API key input or environment variables
- **Dataset Manager**: import datasets from JSON, CSV, or Hugging Face Hub; create and edit cases manually
- **Evaluation metrics**: Exact Match, Sequence Similarity, and LLM-as-a-Judge (Correctness, Completeness, Clarity scored 1–5)
- **Cost & latency tracking**: per-case token counts, latency, and estimated USD cost
- **Run history & comparison**: leaderboard, performance charts, per-case drill-down with pass/fail filtering
- **Sandbox mode**: deterministic mock responses for pipeline testing without any API keys

---

## Architecture

```
llm-evals/
├── backend/              # FastAPI application
│   ├── main.py           # REST API endpoints
│   ├── evaluator.py      # LiteLLM-backed evaluation engine
│   └── database.py       # File-based JSON storage
├── frontend/             # React + Vite SPA
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── DashboardOverview.jsx
│           ├── DatasetManager.jsx
│           ├── EvaluationRunner.jsx
│           ├── RunsHistory.jsx
│           └── RunDetails.jsx
├── data/
│   ├── datasets/         # Stored golden datasets (JSON)
│   └── runs/             # Completed evaluation runs (JSON)
└── start_dashboard.sh    # One-command startup script
```

---

## Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) v18+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)

### 1. Clone

```bash
git clone https://github.com/debabratamishra/llm-evals
cd llm-evals
```

### 2. Start

```bash
chmod +x start_dashboard.sh
./start_dashboard.sh
```

The script installs all dependencies, starts the FastAPI backend and the Vite dev server, and prints the URLs.

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |

---

## LLM Providers

All model calls are routed through [LiteLLM](https://github.com/BerriAI/litellm). Four providers are available in the **Run Evaluation** tab:

| Provider | Key required | Notes |
|---|---|---|
| **Sandbox** | No | Deterministic mock — no network access |
| **Ollama — Local** | No | Requires `ollama serve` running locally |
| **Ollama — Cloud** | Yes | Any OpenAI-compatible remote Ollama endpoint |
| **OpenRouter** | Yes | Access to 200+ models via a single API key |

API keys can be supplied in two ways — they are never persisted to disk:

**Environment variables** (recommended for repeated use):
```bash
export OLLAMA_API_KEY="..."        # Ollama cloud only
export OLLAMA_BASE_URL="https://…" # Ollama cloud only
export OPENROUTER_API_KEY="..."
```

**Per-run form input**: enter keys directly in the Run Evaluation tab. Keys are sent with the request and discarded after the evaluation completes.

---

## Dataset Formats

Datasets can be uploaded as JSON or CSV, or imported directly from [Hugging Face Hub](https://huggingface.co/datasets).

**JSON**
```json
[
  {
    "id": "case-01",
    "question": "What is the primary mechanism of Metformin?",
    "ideal_answer": "Metformin decreases hepatic glucose production and improves insulin sensitivity."
  }
]
```

**CSV**
```csv
question,ideal_answer
What is the capital of France?,Paris.
Explain first-pass metabolism.,A drug is metabolised by liver enzymes before entering systemic circulation.
```

Column headers are matched case-insensitively. Accepted aliases: `question` / `prompt` / `query`; `ideal_answer` / `answer` / `reference`.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Exact Match | Binary (0/1) after alphanumeric normalisation |
| Similarity | Sequence overlap score in [0, 1] |
| LLM Correctness | Judge score 1–5: factual accuracy |
| LLM Completeness | Judge score 1–5: coverage of reference points |
| LLM Clarity | Judge score 1–5: coherence and readability |
| Latency | Per-case response time in seconds |
| Cost | Estimated USD based on provider token pricing |

The LLM judge cascades through available providers (OpenRouter → local Ollama) and falls back to heuristic scoring when no provider is reachable.

---

## License

[MIT](LICENSE)

<div align="center">

# LLM Evaluation Framework

[![License](https://img.shields.io/github/license/debabratamishra/llm-evals?style=flat-square&color=blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square&logo=python&logoColor=ffd343)](pyproject.toml)
[![Node](https://img.shields.io/badge/node-18%2B-5fa04e?style=flat-square&logo=node.js&logoColor=fff)](frontend/package.json)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138%2B-009688?style=flat-square&logo=fastapi)](backend/main.py)
[![React](https://img.shields.io/badge/React-19-61dafb?style=flat-square&logo=react)](frontend/src/App.jsx)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](https://github.com/debabratamishra/llm-evals/pulls)

**Benchmark, evaluate, and compare LLMs — from your browser.**

Upload or import evaluation datasets, run multi-provider evaluations (OpenRouter,
Nvidia NIM, or local sandbox), and analyse results in a real-time dashboard.
Support standard single-model evals and head-to-head arena comparisons.

[🚀 Quick Start](#-quick-start) · [📖 API Docs](#-api-documentation) · [🗺️ Roadmap](#-roadmap) · [🤝 Contributing](#-contributing)

</div>

---

## ✨ Features

| Capability | Description |
|---|---|
| **Multi-backend LLM** | Evaluate models via [OpenRouter](https://openrouter.ai) (200+ models), [Nvidia NIM](https://build.nvidia.com), or built-in sandbox — all routed through [LiteLLM](https://github.com/BerriAI/litellm). API keys accepted per-run or via environment variables; never persisted to disk. |
| **Dataset Manager** | Import datasets from JSON, CSV, or [Hugging Face Hub](https://huggingface.co/datasets). Create and edit cases manually. Column auto-detection for common QA formats. |
| **Rich Metrics** | Exact Match, Sequence Similarity, and LLM-as-a-Judge (Correctness, Completeness, Clarity scored 1–5). Cost & latency tracking per case with estimated USD totals. |
| **Arena Mode** | Run two or more models head-to-head on the same dataset. Compare metrics side-by-side in a unified leaderboard. |
| **Interactive Dashboard** | Real-time React SPA with run history, pass/fail filtering, per-case drill-down, performance charts, and arena comparison views. |
| **Deterministic Sandbox** | Test the full evaluation pipeline without any API keys. Mock responses let you validate infrastructure before connecting real models. |
| **One-Command Start** | A single shell script installs dependencies, starts the FastAPI backend, launches the Vite dev server, and prints URLs. |
| **Single-Service Deploy** | Deploy to [Render](DEPLOY.md) (free tier) as one web service — the backend serves both the API and the React SPA from the same origin. No CORS, no multi-service complexity. |
| **Security by Default** | SSRF-protected NIM URL validation, strict ID allowlists preventing path traversal, upload size caps, CORS scoped to explicit origins, per-run credential isolation. |

---

## 🧠 Architecture

```
┌──────────────────────────────────────────────┐
│             Browser (User)                   │
│    ┌─────────────────────────────────┐       │
│    │      React SPA (Dashboard)      │       │
│    └──────────┬──────────────────────┘       │
│               │ fetch('/api/...')            │
├───────────────┼──────────────────────────────┤
│               │                              │
│               │                              │
│               ▼                              │
│  ┌────────────────────────────────────┐      │
│  │         FastAPI (uvicorn)          │      │
│  │  ┌───────┐ ┌──────────┐ ┌───────┐  │      │
│  │  │ Router│→│Evaluator │→│ DB    │  │      │
│  │  │   +   │ │ (LiteLLM)│ │ (JSON │  │      │
│  │  │ CORS  │ │ Sandbox  │ │ File) │  │      │
│  │  └───────┘ └──────────┘ └───────┘  │      │
│  └────────────────────────────────────┘      │
│                        │                     │
│              ┌─────────┴──────────┐          │
│              ▼                    ▼          │
│      OpenRouter          Nvidia NIM          │
│      (200+ models)       (cloud/hosted)      │
└──────────────────────────────────────────────┘
```

### Layout

```
llm-evals/
├── backend/
│   ├── main.py         # FastAPI app — API routes, CORS, static SPA mount
│   ├── database.py     # File-based JSON storage (datasets, runs, arena runs)
│   └── evaluator.py    # LiteLLM-backed evaluation engine
├── frontend/
│   ├── src/            # React + Vite SPA (dashboard UI)
│   └── public/         # Static assets (icons, favicon)
├── data/               # Runtime data — gitignored, auto-seeded
│   ├── datasets/
│   ├── runs/
│   └── arena_runs/
├── .github/
│   └── workflows/      # CI (CodeQL)
├── render.yaml         # Render blueprint (infra-as-code)
├── DEPLOY.md           # Deployment guide (Render free tier)
├── pyproject.toml      # Python deps managed by uv
└── start_dashboard.sh  # Local one-command startup
```
---

## 🚀 Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) v18+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager — one command install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Local development

```bash
git clone https://github.com/debabratamishra/llm-evals
cd llm-evals
chmod +x start_dashboard.sh
./start_dashboard.sh
```

The script installs all dependencies, starts the FastAPI backend and the Vite
dev server, and prints the URLs:

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API docs | http://localhost:8000/docs |

```bash
git push origin main
```

---

## 🔑 LLM Providers

All model calls are routed through [LiteLLM](https://github.com/BerriAI/litellm).
Three providers are available in the **Run Evaluation** tab:

| Provider | Key Required | Notes |
|----------|-------------|-------|
| **Sandbox** | No | Deterministic mock — no network access, ideal for testing |
| **Nvidia NIM** | Yes (optional for self-hosted) | Cloud hosted models or self-hosted NIM containers |
| **OpenRouter** | Yes | Access to 200+ models via a single API key |

Keys can be supplied in two ways — they are **never persisted to disk**:

**Environment variables** (recommended for repeated use):
```bash
export NVIDIA_NIM_API_KEY="..."     # Nvidia NIM cloud key
export NVIDIA_NIM_API_BASE="..."    # Nvidia NIM base URL (optional)
export OPENROUTER_API_KEY="..."
```

**Per-run form input**: enter keys directly in the Run Evaluation tab. Keys
are sent with the request and discarded after the evaluation completes.

---

## 📊 Evaluation Metrics

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| Exact Match | Binary | 0/1 | After alphanumeric normalisation |
| Similarity | Float | [0, 1] | Sequence overlap score |
| LLM Correctness | Integer | 1–5 | Judge-scored factual accuracy |
| LLM Completeness | Integer | 1–5 | Judge-scored coverage of reference |
| LLM Clarity | Integer | 1–5 | Judge-scored coherence and readability |
| Latency | Float | seconds | Per-case response time |
| Cost | Float | USD | Estimated cost (provider token pricing) |

The LLM judge cascades through available providers (OpenRouter → Nvidia NIM)
and falls back to heuristic scoring when no provider is reachable.

---

## 📦 Dataset Formats

Datasets can be uploaded as JSON or CSV, or imported directly from
[Hugging Face Hub](https://huggingface.co/datasets).

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

Column headers are matched case-insensitively. Accepted aliases:
`question` / `prompt` / `query` for questions;
`ideal_answer` / `answer` / `reference` for answers.

---

## 📚 API Documentation

When the backend is running, auto-generated Swagger docs are available at: http://localhost:8000/docs

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/check-keys` | Health check + API key status |
| GET | `/api/datasets` | List all datasets |
| POST | `/api/datasets` | Create a dataset |
| POST | `/api/datasets/upload` | Upload dataset file (JSON/CSV) |
| POST | `/api/datasets/import-hf` | Import from Hugging Face Hub |
| GET | `/api/runs` | List evaluation runs |
| POST | `/api/runs` | Execute an evaluation |
| POST | `/api/arena-runs` | Execute an arena comparison |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12+, FastAPI, uvicorn, LiteLLM |
| Frontend | React 19, Vite, Recharts, Lucide React Icons |
| Storage | File-based JSON (extensible to hosted DB) |
| Package | uv (Python), npm (Node) |

---

## 🏷️ Release Cycle

Every merge to `main` creates a new release. The project uses **semantic
versioning** (MAJOR.MINOR.PATCH) with **conventional commits**.

### How releases work

```bash
# 1. Commit using conventional format:
git commit -m "feat: add arena comparison view"
git commit -m "fix: handle empty dataset gracefully"

# 2. When ready, cut a release (auto-detects bump type):
uv run python release.py          # auto: patch/minor/major from commits
uv run python release.py --dry-run  # preview without changes

# 3. Push the release:
git push origin main --tags
```

This bumps `pyproject.toml`, generates the `CHANGELOG.md` section from git
history, commits, and tags (`v1.2.3`). GitHub Actions also auto-tags pushes
to `main` — the CI workflow at `.github/workflows/release.yml` detects patch
bumps from commit history.

### Commit convention

| Prefix   | Meaning       | Bump   |
|----------|---------------|--------|
| `feat:`  | New feature   | minor  |
| `fix:`   | Bug fix       | patch  |
| `BREAKING:` | Breaking change | major |

See `.semver-config` and `.commitlintrc.json` for the full reference.

---

## 🤝 Contributing

Contributions are welcome! This project is open source under the
[Apache 2.0 License](LICENSE).

- **Report bugs** — open a [GitHub Issue](https://github.com/debabratamishra/llm-evals/issues)
- **Submit PRs** — fork, branch, commit, open a pull request
- **Suggest features** — use the issue tracker with the `enhancement` tag

Guidelines:
1. Run `npm run build` in `frontend/` and `uv run python backend/main.py` before opening a PR
2. Match existing code style (FastAPI type-hinted endpoints, React functional components)
3. Keep API keys out of git — use `.env` or `render.yaml` env vars
4. One feature / fix per PR

---

## 📄 License

Copyright 2026 Debabrata Mishra

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for
the full text.

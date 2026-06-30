# Deploy llm-evals to Render (Free Tier)

This project deploys as a **single Render Web Service** — the FastAPI backend serves both
the REST API and the React SPA frontend from the same origin. No separate services, no
CORS configuration, no frontend URL rewriting needed.

## What was changed

These files were modified to enable the single-service deploy:

| File              | Change                                                              |
|-------------------|---------------------------------------------------------------------|
| `backend/main.py` | Added `StaticFiles` mount + SPA catch-all route + `PORT` from env   |
| `render.yaml`     | New — Render infra-as-code (blueprint)                              |

The frontend code is untouched. All `fetch('/api/...')` calls continue to use relative
paths because the API and UI live on the same origin.

## Before you deploy

### 1. Push the changes to GitHub

You need Render to see these files:

```bash
git add backend/main.py render.yaml
git commit -m "deploy: single-service Render config (API + SPA from one origin)"
git push origin main
```

### 2. Set required env vars on Render

These are the LLM provider API keys. **Set them in the Render dashboard** (not in
render.yaml — that file is public in your repo).

| Key                   | Required?   | Description                       |
|-----------------------|-------------|-----------------------------------|
| `OPENROUTER_API_KEY`  | For OpenRouter models | Your openrouter.ai API key   |
| `NVIDIA_NIM_API_KEY`  | For Nvidia NIM models | Nvidia NGC API key (optional) |
| `NVIDIA_NIM_API_BASE` | For Nvidia NIM (self-hosted) | Base URL (optional)       |

**Without these**, only the Sandbox provider works (deterministic mock responses —
good for testing the pipeline but no actual LLM calls).

## Step-by-step deployment

### Option A — Blueprint deploy (recommended for first time)

Render can read `render.yaml` directly from your repo:

1. Log in to [dashboard.render.com](https://dashboard.render.com)
2. Click **New → Blueprint**
3. Connect your GitHub repo (`debabratamishra/llm-evals`)
4. Render detects `render.yaml` and shows the `llm-evals` web service
5. Click **Apply**
6. While the service builds, go to **Environment** → add the API keys from the table above
7. Wait for the build to finish (~3-5 minutes). The logs show:
   - `==> Installing uv`
   - `==> Building frontend`
   - `==> Syncing Python dependencies`
   - `Uvicorn running on ...`
8. Open the `.onrender.com` URL shown in the dashboard

### Option B — Manual service creation

If Blueprint isn't available:

1. Log in to [dashboard.render.com](https://dashboard.render.com)
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Fill in the fields:

   | Field            | Value                                                                 |
   |------------------|-----------------------------------------------------------------------|
   | Name             | `llm-evals`                                                           |
   | Runtime          | Python 3                                                              |
   | Build Command    | See below                                                             |
   | Start Command    | `$HOME/.local/bin/uv run python backend/main.py`                      |
   | Health Check     | `/api/check-keys`                                                     |
   | Plan             | Free                                                                  |

   **Build command** (copy this exactly):
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && cd frontend && npm ci && npm run build && cd .. && $HOME/.local/bin/uv sync --frozen
   ```

5. In **Environment**, add:
   - `PYTHON_VERSION` = `3.12`
   - `PYTHONUNBUFFERED` = `1`
   - `CORS_ALLOWED_ORIGINS` = `http://localhost:3000,http://localhost:5173`
   - Plus the LLM API keys from the table above

6. Click **Create Web Service**

## What you get

After deployment, your single `.onrender.com` URL serves:

| Path              | What                                    |
|-------------------|-----------------------------------------|
| `/`               | React SPA dashboard (index.html)        |
| `/assets/*`       | Vite bundle (JS, CSS, icons, fonts)     |
| `/api/check-keys` | Health check + API key status           |
| `/api/datasets`   | List datasets                           |
| `/api/runs`       | List evaluation runs                    |
| `/api/arena-runs` | List arena runs                         |
| `/docs`           | Swagger API docs (auto-generated)       |

## Known limitations (Free tier)

- **Ephemeral storage**: `data/datasets/`, `data/runs/`, `data/arena_runs/` live on
  the Render VM's local disk. Everything resets when the service restarts, deploys,
  or sleeps (15 min of inactivity on Free). Your datasets and run history disappear.
- **Sleep on idle**: Free web services spin down after 15 minutes of no traffic.
  The first request after sleep takes ~30-60s to cold-start.
- **Timeout**: The Free plan has a 100s hard timeout on HTTP requests. Very large
  evaluation runs (100+ cases with LLM-as-Judge scoring) can exceed this. Keep
  datasets under ~50 cases for reliable results.

## Updating after redeploy

After you `git push`, Render auto-deploys (if `autoDeploy: true` in render.yaml).
To force a manual deploy: Render Dashboard → llm-evals → Manual Deploy → Deploy latest commit.

## Troubleshooting

**Build fails with "uv: command not found"**
The build command installs uv via the astral install script. Check that the build
command includes `export PATH="$HOME/.local/bin:$PATH"` and that the start command
uses `$HOME/.local/bin/uv` (not bare `uv`).

**Frontend shows a blank page or "cannot connect to backend"**
The SPA loads from `/api/*` relative paths. If you see API errors, check:
1. The backend health check passes (`/api/check-keys` returns JSON)
2. The build command ran `npm run build` successfully (check build logs)
3. `frontend/dist/index.html` exists on the server

**Evaluation runs hang or timeout**
On the Free tier, large runs may exceed the 100s HTTP timeout. Reduce dataset size
or switch to Sandbox provider for testing. Pro plan removes this limit.

**Imported datasets disappear after a few hours**
This is expected — Free tier has ephemeral disk. All uploaded datasets, run
history, and arena results are lost on restart or sleep. For persistent data,
upgrade to Render's paid persistent disk ($1/month/GB) or add a hosted database
(Neon, Supabase free tier).
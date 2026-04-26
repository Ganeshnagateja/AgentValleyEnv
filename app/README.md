# Agent Valley Frontend

React/Vite frontend for Agent Valley.

The active backend is the Python FastAPI server in `../server/app.py`. The old
Express server under `app/server/` is legacy-only and is not used for real
training.

## Run

Terminal 1, from the repository root:

```bash
python -m server.app
```

Terminal 2, from this `app/` directory:

```bash
npm install
npm run dev
```

The frontend runs on `http://localhost:3000`. Vite proxies `/api/*` to
`http://localhost:7860`.

## Training Dashboard

The Backend RL Training screen calls the Python backend:

```text
GET  /api/training/status
GET  /api/training/metrics
POST /api/training/start
POST /api/training/stop
POST /api/policy/evaluate
```

Learning curves are rendered from backend JSONL metric fields only. The visual
canvas remains a separate local simulation.

## Scripts

```bash
npm run dev          # frontend dev server
npm run build        # type check and production build
npm run backend      # start Python backend from app/
npm run legacy:server
```

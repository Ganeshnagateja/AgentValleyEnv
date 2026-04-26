# Legacy Node Server

This Express server is retained only as a historical visual-demo backend. It is
not part of the active development flow and does not provide real reinforcement
learning.

Use the Python FastAPI backend from the repository root instead:

```bash
python -m server.app
```

The React app proxies `/api/*` to `http://localhost:7860`, where the Python
backend exposes environment, training, metrics, and policy endpoints.

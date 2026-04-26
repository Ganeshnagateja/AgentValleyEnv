FROM node:20-slim AS frontend

WORKDIR /frontend/app
COPY app/package*.json ./
RUN npm ci
COPY app ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY env ./env
COPY datasets ./datasets
COPY server ./server
COPY training ./training
COPY artifacts ./artifacts
COPY assets ./assets
COPY scripts ./scripts
COPY docs ./docs
COPY notebooks ./notebooks
COPY baseline_eval.py inference.py openenv.yaml README.md pyproject.toml ./
COPY --from=frontend /frontend/app/dist ./app/dist

EXPOSE 7860
CMD ["python", "-m", "server.app"]

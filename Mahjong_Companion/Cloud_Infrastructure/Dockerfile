# Showcase copy of services/api/Dockerfile from the private mahjong-companion repo. Here to be read, not run.

# Build from the repo root: docker build -f services/api/Dockerfile .
# One image serves the API under /api and the web app at / (ADR 6).

FROM node:22-slim AS web
RUN npm install -g pnpm@10
WORKDIR /web
COPY apps/web/package.json apps/web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY apps/web ./
RUN pnpm build

FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY services/api ./services/api
RUN uv sync --frozen --no-dev --package mahjong-api
COPY --from=web /web/dist ./web
ENV WEB_DIR=/app/web
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "mahjong_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

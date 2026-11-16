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
# The app scales to zero between games, and this base image ships no compiled Python. Compile
# Python's own library and the installed packages here, once, instead of at every cold start
# (in a container held to 0.5 CPU: about 4.5 s from start to the first answer without this,
# about 1.7 s with it).
ENV UV_COMPILE_BYTECODE=1
RUN python -m compileall -q -x site-packages /usr/local/lib
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY services/api ./services/api
RUN uv sync --frozen --no-dev --package mahjong-api
COPY --from=web /web/dist ./web
ENV WEB_DIR=/app/web
# The commit this image was built from and where it runs. Langfuse records them on every trace
# as the release and the environment, so a change in quality or cost can be tied to a deploy and
# real games stay apart from test runs. The deploy workflow passes both; a local build passes
# neither and traces as "development". Last, so it doesn't undo the layers above.
ARG GIT_SHA=""
ARG TRACING_ENVIRONMENT=""
ENV LANGFUSE_RELEASE=$GIT_SHA LANGFUSE_TRACING_ENVIRONMENT=$TRACING_ENVIRONMENT
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "mahjong_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

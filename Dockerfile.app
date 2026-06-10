# Light app-tier image for the API / review-UI / console (the navigation + query
# surface). Deliberately torch-free — these services only read Postgres and serve
# HTTP/JSON, so the image stays small and builds fast on aarch64. The AI workers
# (cv/vlm/embedding) keep their own heavy inference stack and run separately.
#
# The repo is bind-mounted at /app by docker-compose.platform.yml, so code edits
# don't need a rebuild — only a dependency change does.
FROM python:3.12-slim

WORKDIR /app

# Only the light deps the read/query/console tier needs (psycopg v3 binary build
# avoids needing libpq-dev). pyyaml/numpy cover the config + embedding-search imports.
RUN pip install --no-cache-dir \
      "fastapi" "uvicorn[standard]" "httpx" "psycopg[binary]" "pyyaml" "numpy"

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000 8010 8080

# Default command is overridden per service in docker-compose.platform.yml.
CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

#!/usr/bin/env bash
# DB (Postgres): client + migrate + seed. Needs the postgres service up (make up).
set -euo pipefail; cd "$(dirname "$0")/../.."
.venv/bin/pip install -q -r requirements/db.txt
.venv/bin/python scripts/db_migrate.py up
.venv/bin/python scripts/db_seed.py
echo "setup/db: OK"

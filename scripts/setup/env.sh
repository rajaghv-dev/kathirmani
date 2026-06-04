#!/usr/bin/env bash
# Base env: venv + shared libs + .env. Frameworks: spec/12-frameworks.md.
set -euo pipefail; cd "$(dirname "$0")/../.."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements/base.txt -r requirements/dev.txt
[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example — edit secrets"; }
echo "setup/env: OK"

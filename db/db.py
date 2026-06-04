"""Tiny DB helper: DSN resolution + a psycopg connection.

Postgres is the platform's single structured store (metadata, events, vectors,
model runs, and the job queue). DSN from DATABASE_URL, defaulting to the base
docker-compose Postgres. psycopg (v3) is the driver.
"""
from __future__ import annotations

import os


def dsn() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://kathir:change_me_in_env@localhost:5432/kathirmani",
    )


def connect():
    import psycopg
    return psycopg.connect(dsn())

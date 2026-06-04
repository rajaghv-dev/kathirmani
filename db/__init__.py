"""Platform DB package (Postgres = the single structured store)."""
from .db import connect, dsn

__all__ = ["connect", "dsn"]

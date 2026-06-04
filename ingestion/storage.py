"""Object storage abstraction (master plan §5.2: "MinIO or local NVMe path").

LocalFSStorage is the hermetic default (tests, no service). MinioStorage uploads to
the running MinIO and returns an s3:// URI. Factory picks by `backend`/env, and
falls back to local if the MinIO client/endpoint is unavailable.
"""
from __future__ import annotations

import abc
import os
import shutil
from pathlib import Path


class StorageBackend(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def put_clip(self, local_path: Path, key: str) -> str:
        """Store the clip under `key`; return its canonical path/URI."""

    @abc.abstractmethod
    def put_thumb(self, local_path: Path, key: str) -> str:
        ...


class LocalFSStorage(StorageBackend):
    name = "local"

    def __init__(self, base_dir: Path):
        self.base = Path(base_dir)

    def _put(self, local_path: Path, key: str) -> str:
        dest = self.base / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(local_path).resolve() != dest.resolve():
            shutil.move(str(local_path), str(dest))
        return str(dest)

    put_clip = _put
    put_thumb = _put


class MinioStorage(StorageBackend):
    name = "minio"

    def __init__(self, endpoint: str, access: str, secret: str, bucket: str, secure: bool = False):
        from minio import Minio
        self.bucket = bucket
        self.client = Minio(endpoint, access_key=access, secret_key=secret, secure=secure)
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def _put(self, local_path: Path, key: str) -> str:
        self.client.fput_object(self.bucket, key, str(local_path))
        return f"s3://{self.bucket}/{key}"

    put_clip = _put
    put_thumb = _put


def make_storage(backend: str | None, base_dir: Path) -> StorageBackend:
    backend = backend or os.environ.get("INGEST_STORAGE", "local")
    if backend == "minio":
        try:
            return MinioStorage(
                endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
                access=os.environ.get("MINIO_ROOT_USER", "kathir"),
                secret=os.environ.get("MINIO_ROOT_PASSWORD", "change_me"),
                bucket=os.environ.get("MINIO_BUCKET", "clips"),
            )
        except Exception as e:                       # graceful fallback — never block ingest
            print(f"[storage] MinIO unavailable ({e}); falling back to local")
    return LocalFSStorage(base_dir)

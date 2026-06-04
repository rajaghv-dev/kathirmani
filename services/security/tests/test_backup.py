"""backup: command-build + clips manifest without running pg_dump. No DB needed."""
from __future__ import annotations

from services.security import backup


def test_pg_dump_command_shape(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:secretpw@h:5432/db")
    cmd = backup.pg_dump_command("/tmp/out.dump")
    assert cmd[0] == "pg_dump"
    assert "--format=custom" in cmd
    assert "--file=/tmp/out.dump" in cmd
    assert cmd[-1] == "--dbname=postgresql://u:secretpw@h:5432/db"


def test_clips_manifest_checksums(tmp_path):
    (tmp_path / "a.mkv").write_bytes(b"hello-clip")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.mp4").write_bytes(b"another")
    (tmp_path / "notes.txt").write_text("ignore me")

    man = backup.clips_manifest(tmp_path)
    names = {entry["path"].split("/")[-1] for entry in man}
    assert names == {"a.mkv", "b.mp4"}  # only clip patterns
    for e in man:
        assert len(e["sha256"]) == 64 and e["size"] > 0


def test_clips_manifest_missing_dir():
    assert backup.clips_manifest("/no/such/dir") == []


def test_run_backup_dryplan_redacts_dsn(tmp_path):
    res = backup.run_backup(tmp_path / "bk", tmp_path, dsn="postgresql://u:pw@h/db")
    assert res["executed"] is False
    assert res["pg_dump_cmd_redacted"][-1] == "--dbname=***REDACTED***"
    # the real DSN must not appear anywhere in the returned plan
    assert "pw@h" not in repr(res)
    assert res["pg_dump_path"].endswith(".dump")
    assert res["manifest_path"].endswith(".json")


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    import hashlib
    assert backup.sha256_file(f) == hashlib.sha256(b"abc").hexdigest()

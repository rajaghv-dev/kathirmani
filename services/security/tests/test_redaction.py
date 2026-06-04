"""redaction: secrets masked in text + dicts. No DB needed."""
from __future__ import annotations

from services.security.redaction import MASK, redact


def test_rtsp_credentials_masked():
    out = redact("rtsp://admin:hunter2@10.0.0.5:554/stream")
    assert "hunter2" not in out
    assert "rtsp://admin:" in out and MASK in out


def test_postgres_dsn_masked():
    out = redact("postgresql://kathir:change_me_in_env@localhost:5432/kathirmani")
    assert "change_me_in_env" not in out


def test_bearer_token_masked():
    out = redact("Authorization: Bearer abc.def.ghijkl123")
    assert "abc.def.ghijkl123" not in out and "Bearer" in out


def test_hf_token_masked():
    out = redact("export HF_TOKEN=hf_abcdEFGH1234567890")
    assert "hf_abcdEFGH1234567890" not in out


def test_keyvalue_masked():
    assert "topsecret" not in redact("password=topsecret")
    assert "myapikey99" not in redact('api_key: "myapikey99"')


def test_dict_secret_keys_masked():
    d = {"user": "alice", "password": "p@ss", "rtsp_url": "rtsp://u:pw@h/s",
         "nested": {"api_key": "deadbeef", "ok": 5}}
    r = redact(d)
    assert r["user"] == "alice"
    assert r["password"] == MASK
    assert "pw" not in r["rtsp_url"]
    assert r["nested"]["api_key"] == MASK
    assert r["nested"]["ok"] == 5  # non-secret scalar untouched


def test_list_recursed():
    r = redact(["ok", "token=abc123def456", {"secret": "x"}])
    assert r[0] == "ok"
    assert "abc123def456" not in r[1]
    assert r[2]["secret"] == MASK


def test_non_secret_text_untouched():
    assert redact("camera bill_counter at 10:00") == "camera bill_counter at 10:00"

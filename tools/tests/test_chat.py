"""Tests for tools/chat.py. Pure functions and backends with fakes; no serial port needed."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chat  # noqa: E402

LOCAL_FOOTER = (
    "model   : stub\n"
    "\n--\n4 tokens in 0.100 s = 40.00 tok/s (first token 0.010 s)\n"
    "weight bytes read: 1.0 MB total, 0.25 MB/token\n"
)

SERIAL_FOOTER = (
    "64 tokens in 16104 ms = 3.97 tok/s (first token 262 ms)\n"
    "weights read 545259 KB total, 8520 KB/token, effective 33.86 MB/s\n"
)


def test_parse_stats_local():
    st = chat.parse_stats_local(LOCAL_FOOTER)
    assert st.tokens == 4
    assert st.seconds == 0.1
    assert st.tok_s == 40.0
    assert st.first_s == 0.01
    assert st.mb_per_token == 0.25
    assert st.eff_mbs is None


def test_parse_stats_local_absent():
    assert chat.parse_stats_local("open: No such file or directory\n") is None


def test_parse_stats_serial():
    st = chat.parse_stats_serial(SERIAL_FOOTER)
    assert st.tokens == 64
    assert st.seconds == 16.104
    assert st.tok_s == 3.97
    assert st.first_s == 0.262
    assert abs(st.mb_per_token - 8520 / 1024.0) < 1e-9
    assert st.eff_mbs == 33.86

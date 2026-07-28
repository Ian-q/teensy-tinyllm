#!/usr/bin/env python3
"""Terminal chat REPL for tinyllm.

Backends:
  local   spawn host/build/tq_run per turn — works with no hardware at all
  serial  drive the Zephyr shell on the Teensy over USB CDC

Usage:
  python3 tools/chat.py local  host/build/test_q4.etq
  python3 tools/chat.py serial /dev/cu.usbmodem12345

Design: docs/superpowers/specs/2026-07-27-chat-cli-design.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class BackendError(Exception):
    """Anything the backend cannot recover from mid-command."""


@dataclass
class GenOpts:
    n: int = 128
    temp: float = 0.9
    topp: float = 0.9
    seed: int | None = None  # None = fresh random seed every turn


@dataclass
class GenStats:
    tokens: int
    seconds: float
    tok_s: float
    first_s: float
    mb_per_token: float | None = None
    eff_mbs: float | None = None


_LOCAL_STATS_RE = re.compile(
    r"(\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \(first token ([\d.]+) s\)"
)
_LOCAL_BYTES_RE = re.compile(r"weight bytes read: [\d.]+ MB total, ([\d.]+) MB/token")

_SERIAL_STATS_RE = re.compile(
    r"(\d+) tokens in (\d+) ms = (\d+\.\d+) tok/s \(first token (\d+) ms\)"
)
_SERIAL_BYTES_RE = re.compile(
    r"weights read \d+ KB total, (\d+) KB/token, effective (\d+\.\d+) MB/s"
)


def parse_stats_local(text: str) -> GenStats | None:
    m = _LOCAL_STATS_RE.search(text)
    if not m:
        return None
    st = GenStats(int(m[1]), float(m[2]), float(m[3]), float(m[4]))
    b = _LOCAL_BYTES_RE.search(text)
    if b:
        st.mb_per_token = float(b[1])
    return st


def parse_stats_serial(text: str) -> GenStats | None:
    m = _SERIAL_STATS_RE.search(text)
    if not m:
        return None
    st = GenStats(int(m[1]), int(m[2]) / 1000.0, float(m[3]), int(m[4]) / 1000.0)
    b = _SERIAL_BYTES_RE.search(text)
    if b:
        st.mb_per_token = int(b[1]) / 1024.0
        st.eff_mbs = float(b[2])
    return st

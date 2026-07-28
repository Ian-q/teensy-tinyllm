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


SHELL_CMD_MAX = 255  # CONFIG_SHELL_CMD_BUFF_SIZE=256, minus the NUL
# cmd_gen loop: accepts word only when used + l + 2 < 192; worst case is single
# word l + 2 < 192 => l <= 189. So 189-byte prompt is safe, 190+ is dropped.
PROMPT_BUF_MAX = 189


def build_gen_command(prompt: str, opts: GenOpts, seed: int) -> str:
    """Build a `tinyllm gen` line. The firmware re-joins argv words with single
    spaces and SILENTLY DROPS words that overflow its 192-byte prompt buffer,
    so the caps live here, as errors. Also guards against reserved flag tokens
    that the firmware would parse as options."""
    prompt = " ".join(prompt.split())
    if '"' in prompt:
        raise BackendError(
            "the Zephyr shell parses double quotes; remove them from the prompt"
        )
    words = prompt.split()
    for word in words:
        if word in ("-n", "-t", "-p", "-s"):
            raise BackendError(
                f"prompt contains reserved flag token '{word}'; "
                "the firmware would read it as an option"
            )
    nbytes = len(prompt.encode())
    if nbytes > PROMPT_BUF_MAX:
        raise BackendError(
            f"prompt is {nbytes} bytes; the firmware caps it at {PROMPT_BUF_MAX}"
        )
    cmd = f"tinyllm gen -n {opts.n} -t {opts.temp:g} -p {opts.topp:g} -s {seed} {prompt}"
    nbytes = len(cmd.encode())
    if nbytes > SHELL_CMD_MAX:
        raise BackendError(
            f"command is {nbytes} bytes; the shell buffer caps it at {SHELL_CMD_MAX}"
        )
    return cmd


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# A CSI sequence can arrive split across reads (e.g. one byte per feed()); hold
# back a trailing partial sequence so ANSI_RE only ever sees complete ones.
_ANSI_PARTIAL_RE = re.compile(r"\x1b(\[[0-9;]*)?\Z")
PROMPT = "tinyllm> "
FOOTER = "--"
# Best-effort: the strings cmd_load/cmd_gen/cmd_stream emit via shell_error.
ERROR_PREFIXES = (
    "usage:",
    "no model loaded",
    "no PSRAM",
    "encode:",
    "forward at",
    "open:",
    "load:",
    "read:",
)

_LOG_TPL = "[NN:NN:NN.NNN,NNN] <"


def _log_match_len(s: str) -> int:
    """Leading chars of s consistent with the Zephyr log prefix; -1 on mismatch."""
    for i, ch in enumerate(s[: len(_LOG_TPL)]):
        want = _LOG_TPL[i]
        if want == "N":
            if not ch.isdigit():
                return -1
        elif ch != want:
            return -1
    return min(len(s), len(_LOG_TPL))


def _is_log_line(line: str) -> bool:
    return _log_match_len(line) == len(_LOG_TPL)


def _must_hold(partial: str) -> bool:
    """True while a partial line could still turn into a control line."""
    if not partial:
        return False
    if _log_match_len(partial) >= 0:
        return True
    if FOOTER.startswith(partial) or partial == FOOTER:
        return True
    if PROMPT.startswith(partial):
        return True
    return any(e.startswith(partial) or partial.startswith(e) for e in ERROR_PREFIXES)


class ShellFramer:
    """Turns the Zephyr shell's byte stream into ("text"|"error"|"stats"|"done", value) events.

    States: echo (discard through the command echo's newline) -> stream ->
    footer (two stats lines after `--`) -> tail (discard until the prompt).
    """

    def __init__(self, expect_footer: bool = True):
        self.expect_footer = expect_footer
        self.state = "echo"
        self.buf = ""
        self.raw = ""  # unstripped bytes withheld until a partial ANSI code resolves
        self.line_is_text = False  # head of the current line was already emitted
        self.footer: list[str] = []

    def feed(self, data: str) -> list[tuple]:
        out: list[tuple] = []
        self.raw += data
        m = _ANSI_PARTIAL_RE.search(self.raw)
        cut = m.start() if m else len(self.raw)
        cleaned, self.raw = self.raw[:cut], self.raw[cut:]
        self.buf += ANSI_RE.sub("", cleaned).replace("\r", "")
        while True:
            if self.state == "echo":
                nl = self.buf.find("\n")
                if nl < 0:
                    return out
                self.buf = self.buf[nl + 1 :]
                self.state = "stream"
                continue
            nl = self.buf.find("\n")
            if nl >= 0:
                line, self.buf = self.buf[:nl], self.buf[nl + 1 :]
                out.extend(self._complete_line(line))
                continue
            if not self.line_is_text and self.buf.startswith(PROMPT):
                self.buf = self.buf[len(PROMPT) :]
                self.state = "tail"
                out.append(("done", None))
                continue
            if self.state == "stream" and self.buf and (
                self.line_is_text or not _must_hold(self.buf)
            ):
                out.append(("text", self.buf))
                self.buf = ""
                self.line_is_text = True
            return out

    def _complete_line(self, bare: str) -> list[tuple]:
        head_emitted, self.line_is_text = self.line_is_text, False
        if head_emitted:
            # The line's head already streamed out; it cannot be a control line.
            return [("text", bare + "\n")] if self.state == "stream" else []
        if _is_log_line(bare):
            return []
        if self.state == "footer":
            self.footer.append(bare)
            if len(self.footer) == 2:
                self.state = "tail"
                st = parse_stats_serial("\n".join(self.footer))
                return [("stats", st)] if st else []
            return []
        if self.state == "tail":
            return []
        if bare == FOOTER and self.expect_footer:
            self.state = "footer"
            return []
        for e in ERROR_PREFIXES:
            if bare.startswith(e):
                return [("error", bare)]
        return [("text", bare + "\n")]

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

import argparse
import random
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


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


class LocalBackend:
    """One tq_run subprocess per turn. stdout is the token stream, stderr the stats."""

    def __init__(self, tq_run: str, model: str):
        self.tq_run = tq_run
        self.model = model
        self.proc: subprocess.Popen | None = None
        self._stats: GenStats | None = None

    def generate(self, prompt: str, opts: GenOpts) -> Iterator[str]:
        seed = (
            opts.seed if opts.seed is not None else random.randrange(1, 1 << 31)
        )
        argv = [
            self.tq_run, self.model, "-i", prompt,
            "-n", str(opts.n), "-t", f"{opts.temp:g}", "-p", f"{opts.topp:g}",
            "-s", str(seed),
        ]
        self._stats = None
        try:
            self.proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except OSError as e:
            raise BackendError(f"cannot run {self.tq_run}: {e}") from e
        try:
            while True:
                chunk = self.proc.stdout.read1(4096)
                if not chunk:
                    break
                yield chunk.decode("utf-8", "replace")
            err = self.proc.stderr.read().decode("utf-8", "replace")
            rc = self.proc.wait()
        finally:
            if self.proc.poll() is None:
                # generator abandoned mid-stream (Ctrl-C / GeneratorExit)
                self.proc.kill()
                self.proc.wait()
            self.proc.stdout.close()
            self.proc.stderr.close()
        self._stats = parse_stats_local(err)
        if rc != 0:
            tail = "\n".join(err.strip().splitlines()[-3:])
            raise BackendError(f"tq_run exited {rc}: {tail}")

    def stats(self) -> GenStats | None:
        return self._stats

    def cancel(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def load(self, path: str) -> str:
        self.model = path
        return f"model set to {path} for subsequent turns\n"

    def info(self) -> str:
        # Trailing newline: the REPL prints backend text with end="", because
        # the serial backend's lines already carry their newlines.
        return f"local backend: {self.tq_run} {self.model}\n"

    def close(self) -> None:
        self.cancel()


FIRST_BYTE_TIMEOUT = 120.0  # prefill on an SD-streamed model can run ~40 s
INTER_CHUNK_TIMEOUT = 30.0
LOAD_TIMEOUT = 120.0
RESYNC_TIMEOUT = 300.0
DRAIN_TIMEOUT = 2.0


class SerialBackend:
    """Drives the firmware's Zephyr shell. Zero firmware changes: it types what
    you would type and parses what you would read."""

    def __init__(self, port=None, baud=115200, transport=None):
        if transport is None:
            try:
                import serial
            except ImportError as e:
                raise BackendError("serial backend needs pyserial: pip install pyserial") from e
            try:
                transport = serial.Serial(port, baud, timeout=0.1)
            except serial.SerialException as e:
                raise BackendError(f"cannot open {port}: {e}") from e
        self.ser = transport
        self.dirty = False  # an interrupted command may still be producing output
        self._stats: GenStats | None = None

    def transportless_written(self) -> bytes:
        return getattr(self.ser, "written", b"")

    def _drain(self) -> None:
        deadline = time.monotonic() + DRAIN_TIMEOUT
        while self.ser.read(4096):
            if time.monotonic() > deadline:
                raise BackendError("device is flooding the line; power-cycle it?")

    def _resync(self) -> None:
        """Wait out an interrupted command until the prompt reappears."""
        fr = ShellFramer(expect_footer=False)
        fr.state = "stream"  # no echo is coming
        deadline = time.monotonic() + RESYNC_TIMEOUT
        while time.monotonic() < deadline:
            chunk = self.ser.read(4096)
            if not chunk:
                continue
            for ev, _ in fr.feed(chunk.decode("utf-8", "replace")):
                if ev == "done":
                    self.dirty = False
                    return
        raise BackendError("device never returned to its prompt; power-cycle it?")

    def _run(self, cmdline: str, expect_footer: bool, first_timeout: float) -> Iterator[str]:
        if self.dirty:
            self._resync()
        self._drain()
        self.ser.write((cmdline + "\n").encode())
        fr = ShellFramer(expect_footer=expect_footer)
        deadline = time.monotonic() + first_timeout
        # Deferred until "done" (or the deadline): the framer can emit "error"
        # and "done" in the same feed() batch (the common
        # "...error...\r\ntinyllm> " shape). Raising immediately on "error"
        # would discard that bundled "done" and mark the backend dirty even
        # though the device is provably idle at its prompt, sending the next
        # command into a RESYNC_TIMEOUT wait for a prompt that already came.
        error: str | None = None
        while True:
            chunk = self.ser.read(4096)
            if not chunk:
                if time.monotonic() > deadline:
                    self.dirty = True
                    raise BackendError(
                        error or f"timed out waiting for the device after `{cmdline}`"
                    )
                continue
            deadline = time.monotonic() + INTER_CHUNK_TIMEOUT
            for ev, val in fr.feed(chunk.decode("utf-8", "replace")):
                if ev == "text":
                    if error is None:
                        yield val
                elif ev == "stats":
                    self._stats = val
                elif ev == "error":
                    error = error or val
                elif ev == "done":
                    if error is not None:
                        raise BackendError(error)
                    return

    def generate(self, prompt: str, opts: GenOpts) -> Iterator[str]:
        seed = opts.seed if opts.seed is not None else random.randrange(1, 1 << 31)
        self._stats = None
        yield from self._run(build_gen_command(prompt, opts, seed), True, FIRST_BYTE_TIMEOUT)

    def stats(self) -> GenStats | None:
        return self._stats

    def cancel(self) -> None:
        # cmd_gen cannot be aborted; the shell buffers our next line until it returns.
        self.dirty = True

    def load(self, path: str) -> str:
        return "".join(self._run(f"tinyllm load {path}", False, LOAD_TIMEOUT))

    def info(self) -> str:
        return "".join(self._run("tinyllm info", False, 10.0))

    def close(self) -> None:
        self.ser.close()


HELP = """\
anything else   generate a completion of what you typed
/n N            tokens per turn          /t X    temperature
/p X            top-p                    /s N    pin the seed (/s auto to unpin)
/load FILE      load a model (serial: from SD; local: path for the next spawn)
/info           backend/device info      /q      quit"""

DIM, RESET = "\x1b[2m", "\x1b[0m"


def parse_slash(line: str, opts: GenOpts) -> tuple[str, str | None]:
    if not line.startswith("/"):
        return ("gen", line)
    cmd, _, arg = line.partition(" ")
    arg = arg.strip()
    try:
        if cmd == "/q":
            return ("quit", None)
        if cmd == "/help":
            return ("ok", HELP)
        if cmd == "/info":
            return ("info", None)
        if cmd == "/load":
            return ("load", arg) if arg else ("error", "usage: /load <file.etq>")
        if cmd == "/n":
            opts.n = int(arg)
            return ("ok", f"n = {opts.n}")
        if cmd == "/t":
            opts.temp = float(arg)
            return ("ok", f"temp = {opts.temp:g}")
        if cmd == "/p":
            opts.topp = float(arg)
            return ("ok", f"top-p = {opts.topp:g}")
        if cmd == "/s":
            if arg == "auto":
                opts.seed = None
                return ("ok", "seed = fresh each turn")
            opts.seed = int(arg)
            return ("ok", f"seed = {opts.seed}")
    except ValueError:
        return ("error", f"bad value for {cmd}: {arg!r}")
    return ("error", f"unknown command {cmd} — try /help")


def repl(backend, opts: GenOpts) -> None:
    try:
        import readline  # noqa: F401  (line editing + history for input())
    except ImportError:
        pass
    print("tinyllm chat — /help for commands, /q to quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        action, val = parse_slash(line, opts)
        if action == "quit":
            return
        if action in ("ok", "error"):
            print(val)
            continue
        try:
            if action == "info":
                print(backend.info(), end="")
                continue
            if action == "load":
                print(backend.load(val), end="")
                continue
            try:
                for chunk in backend.generate(val, opts):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                backend.cancel()
                print("\n(interrupted — a busy device finishes on its own; "
                      "the next command waits for it)")
                continue
            st = backend.stats()
            if st:
                extra = f", {st.eff_mbs:.1f} MB/s" if st.eff_mbs else ""
                print(f"\n{DIM}{st.tokens} tok, {st.tok_s:.2f} tok/s, "
                      f"first {st.first_s:.2f}s{extra}{RESET}")
            else:
                print()
        except BackendError as e:
            print(f"error: {e}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="chat.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode", required=True)
    lp = sub.add_parser("local", help="run against the native tq_run binary")
    lp.add_argument("model", help="path to a .etq model")
    lp.add_argument(
        "--tq-run",
        default=str(Path(__file__).resolve().parents[1] / "host" / "build" / "tq_run"),
        help="tq_run binary (default: host/build/tq_run)",
    )
    sp = sub.add_parser("serial", help="drive the Teensy over USB CDC")
    sp.add_argument("port", help="serial device, e.g. /dev/cu.usbmodem12345")
    sp.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args(argv)
    try:
        backend = (
            LocalBackend(args.tq_run, args.model)
            if args.mode == "local"
            else SerialBackend(args.port, args.baud)
        )
        try:
            repl(backend, GenOpts())
        finally:
            backend.close()
    except BackendError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

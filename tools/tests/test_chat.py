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


def test_build_gen_command():
    cmd = chat.build_gen_command("Once  upon a time", chat.GenOpts(n=64, temp=0.8), seed=7)
    assert cmd == "tinyllm gen -n 64 -t 0.8 -p 0.9 -s 7 Once upon a time"


def test_build_gen_command_rejects_long_prompt():
    import pytest

    with pytest.raises(chat.BackendError, match="189"):
        chat.build_gen_command("x" * 300, chat.GenOpts(), seed=1)


def test_build_gen_command_rejects_quotes():
    import pytest

    with pytest.raises(chat.BackendError, match="quote"):
        chat.build_gen_command('say "hi"', chat.GenOpts(), seed=1)


def test_build_gen_command_accepts_189_byte_prompt():
    # Exactly 189 bytes is safe
    prompt = "x" * 189
    cmd = chat.build_gen_command(prompt, chat.GenOpts(), seed=1)
    assert "x" * 189 in cmd


def test_build_gen_command_rejects_190_byte_prompt():
    import pytest

    with pytest.raises(chat.BackendError, match="190"):
        chat.build_gen_command("x" * 190, chat.GenOpts(), seed=1)


def test_build_gen_command_rejects_191_byte_prompt():
    import pytest

    with pytest.raises(chat.BackendError, match="191"):
        chat.build_gen_command("x" * 191, chat.GenOpts(), seed=1)


def test_build_gen_command_rejects_flag_token():
    import pytest

    with pytest.raises(chat.BackendError, match="reserved flag token"):
        chat.build_gen_command("hello -t 0.5 world", chat.GenOpts(), seed=1)


def test_build_gen_command_allows_flag_like_word():
    # Substring of flag like "t-shirts" is allowed (only exact matches rejected)
    cmd = chat.build_gen_command("twenty -shirt t-shirts", chat.GenOpts(), seed=1)
    assert "twenty" in cmd and "t-shirts" in cmd


GEN_BYTES = (
    "tinyllm gen -n 4 -t 0.9 -p 0.9 -s 7 Once\r\n"  # echo
    "Once upon a time there was a pony.\n"
    "[00:00:01.123,456] <inf> tinyllm: something\r\n"
    "\r\n--\r\n" + SERIAL_FOOTER.replace("\n", "\r\n") + "tinyllm> "
)


def run_framer(data: str, expect_footer: bool = True, chunk: int = 1):
    fr = chat.ShellFramer(expect_footer=expect_footer)
    events = []
    for i in range(0, len(data), chunk):
        events.extend(fr.feed(data[i : i + chunk]))
    return events


def collect(events):
    text = "".join(v for e, v in events if e == "text")
    stats = [v for e, v in events if e == "stats"]
    errors = [v for e, v in events if e == "error"]
    done = any(e == "done" for e, _ in events)
    return text, stats, errors, done


def test_framer_gen_byte_at_a_time():
    text, stats, errors, done = collect(run_framer(GEN_BYTES, chunk=1))
    assert text == "Once upon a time there was a pony.\n\n"
    assert len(stats) == 1 and stats[0].tokens == 64
    assert errors == [] and done


def test_framer_gen_large_chunks():
    text, stats, errors, done = collect(run_framer(GEN_BYTES, chunk=4096))
    assert text == "Once upon a time there was a pony.\n\n"
    assert len(stats) == 1 and done


def test_framer_ansi_stripped_error():
    data = (
        "tinyllm gen x\r\n\x1b[1;31mno model loaded — `tinyllm load <file.etq>`"
        "\x1b[0m\r\ntinyllm> "
    )
    text, stats, errors, done = collect(run_framer(data))
    assert text == "" and stats == []
    assert errors and errors[0].startswith("no model loaded")
    assert done


def test_framer_dashes_are_text_without_footer():
    data = "tinyllm info\r\n--\r\nkernel: smlad\r\ntinyllm> "
    text, stats, errors, done = collect(run_framer(data, expect_footer=False))
    assert text == "--\nkernel: smlad\n"
    assert stats == [] and done


def test_framer_dash_prefixed_token_text():
    data = "cmd\r\n--not a footer\r\n\r\n--\r\n" + SERIAL_FOOTER.replace("\n", "\r\n") + "tinyllm> "
    text, stats, errors, done = collect(run_framer(data, chunk=1))
    assert text == "--not a footer\n\n"
    assert len(stats) == 1 and done


import pytest  # noqa: E402

STUB_OK = """#!/usr/bin/env python3
import sys
sys.stdout.write(" ".join(sys.argv[1:]))
sys.stdout.flush()
sys.stderr.write(
    "model   : stub\\n"
    "\\n--\\n4 tokens in 0.100 s = 40.00 tok/s (first token 0.010 s)\\n"
    "weight bytes read: 1.0 MB total, 0.25 MB/token\\n")
"""

STUB_FAIL = """#!/usr/bin/env python3
import sys
sys.stderr.write("open: No such file or directory\\n")
sys.exit(1)
"""

STUB_SLOW = """#!/usr/bin/env python3
import sys, time
while True:
    sys.stdout.write("x")
    sys.stdout.flush()
    time.sleep(0.01)
"""


def make_stub(tmp_path, body):
    p = tmp_path / "tq_run"
    p.write_text(body)
    p.chmod(0o755)
    return str(p)


def test_local_backend_streams_and_stats(tmp_path):
    be = chat.LocalBackend(make_stub(tmp_path, STUB_OK), "m.etq")
    text = "".join(be.generate("Once upon", chat.GenOpts(n=4, seed=7)))
    assert text == "m.etq -i Once upon -n 4 -t 0.9 -p 0.9 -s 7"
    st = be.stats()
    assert st and st.tokens == 4 and st.mb_per_token == 0.25


def test_local_backend_random_seed_when_unpinned(tmp_path):
    be = chat.LocalBackend(make_stub(tmp_path, STUB_OK), "m.etq")
    a = "".join(be.generate("hi", chat.GenOpts()))
    b = "".join(be.generate("hi", chat.GenOpts()))
    assert a != b  # -s differs between unpinned turns


def test_local_backend_error_exit(tmp_path):
    be = chat.LocalBackend(make_stub(tmp_path, STUB_FAIL), "m.etq")
    with pytest.raises(chat.BackendError, match="No such file"):
        list(be.generate("hi", chat.GenOpts()))


def test_local_backend_load_swaps_model(tmp_path):
    be = chat.LocalBackend(make_stub(tmp_path, STUB_OK), "old.etq")
    be.load("new.etq")
    text = "".join(be.generate("hi", chat.GenOpts(seed=1)))
    assert text.startswith("new.etq ")


def test_local_backend_abandoned_generator_kills_child(tmp_path):
    be = chat.LocalBackend(make_stub(tmp_path, STUB_SLOW), "m.etq")
    g = be.generate("hi", chat.GenOpts())
    next(g)  # get first chunk
    g.close()  # abandon generator
    assert be.proc.poll() is not None  # child reaped
    assert be.proc.stdout.closed
    assert be.proc.stderr.closed


def test_local_backend_pipes_closed_after_normal_turn(tmp_path):
    be = chat.LocalBackend(make_stub(tmp_path, STUB_OK), "m.etq")
    "".join(be.generate("hi", chat.GenOpts(seed=1)))
    assert be.proc.stdout.closed
    assert be.proc.stderr.closed
    assert be.stats() is not None


class FakeSerial:
    """Scripted transport. Like real hardware, it has nothing to say until a
    command is written — otherwise the backend's pre-command _drain() would
    swallow the scripted reply. read() pops chunks after write() arms it, and
    each pop disarms again so a chunk queued for a later command can't be
    drained early by *that* command's own pre-write _drain()."""

    def __init__(self, chunks):
        self.chunks = [c.encode() if isinstance(c, str) else c for c in chunks]
        self.written = b""
        self.armed = False

    def read(self, n=1):
        if not self.armed or not self.chunks:
            return b""
        self.armed = False
        return self.chunks.pop(0)

    def write(self, data):
        self.written += data
        self.armed = True

    def close(self):
        pass


def test_serial_backend_gen_roundtrip():
    be = chat.SerialBackend(transport=FakeSerial([GEN_BYTES]))
    text = "".join(be.generate("Once", chat.GenOpts(n=4, seed=7)))
    assert be.transportless_written().startswith(b"tinyllm gen -n 4 ")
    assert "pony" in text
    st = be.stats()
    assert st and st.tokens == 64 and st.eff_mbs == 33.86


def test_serial_backend_error_with_prompt_stays_clean():
    # Error text and the prompt arrive in the same reply: the device is
    # provably idle at "tinyllm> " by the time we raise, so no resync needed.
    data = "echo\r\n\x1b[1;31mno model loaded — x\x1b[0m\r\ntinyllm> "
    be = chat.SerialBackend(transport=FakeSerial([data]))
    with pytest.raises(chat.BackendError, match="no model loaded"):
        list(be.generate("hi", chat.GenOpts()))
    assert be.dirty is False


def test_serial_backend_error_without_prompt_marks_dirty(monkeypatch):
    # Error text arrives but the line goes quiet before the prompt does: we
    # cannot prove the device is idle, so the deadline path must mark dirty.
    monkeypatch.setattr(chat, "FIRST_BYTE_TIMEOUT", 0.05)
    monkeypatch.setattr(chat, "INTER_CHUNK_TIMEOUT", 0.05)
    data = "echo\r\nno model loaded — x\r\n"
    be = chat.SerialBackend(transport=FakeSerial([data]))
    with pytest.raises(chat.BackendError, match="no model loaded"):
        list(be.generate("hi", chat.GenOpts()))
    assert be.dirty is True


def test_serial_backend_usable_after_prompt_bundled_error():
    data = "echo\r\n\x1b[1;31mno model loaded — x\x1b[0m\r\ntinyllm> "
    be = chat.SerialBackend(transport=FakeSerial([data, GEN_BYTES]))
    with pytest.raises(chat.BackendError, match="no model loaded"):
        list(be.generate("hi", chat.GenOpts()))
    assert be.dirty is False
    # No resync delay: the second command's reply must be waiting already.
    text = "".join(be.generate("Once", chat.GenOpts(n=4, seed=7)))
    assert "pony" in text
    st = be.stats()
    assert st and st.tokens == 64


def test_serial_backend_first_byte_timeout(monkeypatch):
    monkeypatch.setattr(chat, "FIRST_BYTE_TIMEOUT", 0.05)
    be = chat.SerialBackend(transport=FakeSerial([]))
    with pytest.raises(chat.BackendError, match="timed out"):
        list(be.generate("hi", chat.GenOpts()))


def test_serial_backend_load():
    data = "echo\r\nloading stories15M.etq\r\ndone  (8721 KB in 998 ms, 8734 KB/s)\r\ntinyllm> "
    be = chat.SerialBackend(transport=FakeSerial([data]))
    out = be.load("stories15M.etq")
    assert "done" in out
    assert be.transportless_written().startswith(b"tinyllm load stories15M.etq")

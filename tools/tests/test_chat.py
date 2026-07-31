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
    assert cmd == 'tinyllm gen -n 64 -t 0.8 -p 0.9 -s 7 "Once upon a time"'


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


def test_build_gen_command_allows_11plus_word_prompt():
    # No word-count ceiling now that the prompt is a single quoted argv token.
    prompt = " ".join(f"word{i}" for i in range(15))
    cmd = chat.build_gen_command(prompt, chat.GenOpts(), seed=1)
    assert f'"{prompt}"' in cmd


def test_build_gen_command_keeps_apostrophe():
    cmd = chat.build_gen_command("don't stop", chat.GenOpts(), seed=1)
    assert '"don\'t stop"' in cmd


def test_build_gen_command_rejects_backslash():
    import pytest

    with pytest.raises(chat.BackendError, match="backslash"):
        chat.build_gen_command("a\\b", chat.GenOpts(), seed=1)


def test_build_gen_command_rejects_empty_prompt():
    import pytest

    with pytest.raises(chat.BackendError, match="empty prompt"):
        chat.build_gen_command("   ", chat.GenOpts(), seed=1)


def test_build_gen_command_flag_like_words_stay_inside_quotes():
    cmd = chat.build_gen_command("hello -t 0.5 world", chat.GenOpts(), seed=1)
    assert cmd.endswith('"hello -t 0.5 world"')


GEN_CMD = 'tinyllm gen -n 4 -t 0.9 -p 0.9 -s 7 "Once"'

GEN_BYTES = (
    GEN_CMD + "\r\n"  # echo, exactly as the shell sends it back
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


def test_framer_load_failed_is_error_not_text():
    data = "tinyllm load bogus.etq\r\nload failed: -5\r\ntinyllm> "
    text, stats, errors, done = collect(run_framer(data, expect_footer=False))
    assert text == ""
    assert errors == ["load failed: -5"]
    assert done


def test_scrub_strips_terminal_control_bytes():
    assert chat._scrub("a\x1b]0;t\x07b\x08\ncd\te") == "a]0;tb\ncd\te"


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
    """Scripted transport. Each element of `script` is one command's reply,
    given as a single chunk (str/bytes) or a list of chunks. read() serves a
    chunk only once its command has been issued via write(), so a later
    command's reply can never be drained early, while a single command's
    multi-chunk reply flows without re-arming."""

    def __init__(self, script):
        self.queue = []
        for i, burst in enumerate(script):
            chunks = burst if isinstance(burst, list) else [burst]
            for c in chunks:
                self.queue.append((i, c.encode() if isinstance(c, str) else c))
        self.written = b""
        self.cmds = 0

    def read(self, n=1):
        if self.queue and self.queue[0][0] < self.cmds:
            return self.queue.pop(0)[1]
        return b""

    def write(self, data):
        self.written += data
        self.cmds += 1

    def close(self):
        pass


def test_serial_backend_gen_roundtrip():
    be = chat.SerialBackend(transport=FakeSerial([GEN_BYTES]))
    text = "".join(be.generate("Once", chat.GenOpts(n=4, seed=7)))
    assert be.transportless_written().startswith(b"tinyllm gen -n 4 ")
    assert "pony" in text
    st = be.stats()
    assert st and st.tokens == 64 and st.eff_mbs == 33.86


def test_serial_backend_multichunk_single_command():
    head, tail = GEN_BYTES[:40], GEN_BYTES[40:]
    be = chat.SerialBackend(transport=FakeSerial([[head, tail]]))
    text = "".join(be.generate("Once", chat.GenOpts(n=4, seed=7)))
    assert "pony" in text
    st = be.stats()
    assert st is not None and st.tokens == 64
    assert be.dirty is False


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


def test_serial_backend_load_reports_model_open_error():
    data = "echo\r\nmodel open: I/O error\r\ntinyllm> "
    be = chat.SerialBackend(transport=FakeSerial([data]))
    with pytest.raises(chat.BackendError, match="model open"):
        be.load("bogus.etq")


def test_parse_slash_gen_passthrough():
    assert chat.parse_slash("Once upon a time", chat.GenOpts()) == ("gen", "Once upon a time")


def test_parse_slash_sets_options():
    opts = chat.GenOpts()
    assert chat.parse_slash("/n 32", opts)[0] == "ok"
    assert chat.parse_slash("/t 1.1", opts)[0] == "ok"
    assert chat.parse_slash("/s 99", opts)[0] == "ok"
    assert (opts.n, opts.temp, opts.seed) == (32, 1.1, 99)
    assert chat.parse_slash("/s auto", opts)[0] == "ok"
    assert opts.seed is None


def test_parse_slash_bad_value_and_unknown():
    opts = chat.GenOpts()
    assert chat.parse_slash("/n pony", opts)[0] == "error"
    assert opts.n == 128
    assert chat.parse_slash("/frobnicate", opts)[0] == "error"


def test_parse_slash_load_and_quit():
    opts = chat.GenOpts()
    assert chat.parse_slash("/load a.etq", opts) == ("load", "a.etq")
    assert chat.parse_slash("/load", opts)[0] == "error"
    assert chat.parse_slash("/q", opts) == ("quit", None)


def test_main_serial_open_failure_returns_1(capsys):
    rc = chat.main(["serial", "/nonexistent-port-xyz"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err

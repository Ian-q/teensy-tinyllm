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

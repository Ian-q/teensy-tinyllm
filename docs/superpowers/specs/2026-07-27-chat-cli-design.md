# Chat CLI design — `tools/chat.py`

Date: 2026-07-27
Status: approved (pending spec review)

## Purpose

A terminal chat REPL on the host that talks to the tinyllm engine, with a
pluggable backend so the same tool drives either the native `host/tq_run`
binary (works today, no hardware) or the Teensy's Zephyr shell over USB CDC
serial (works at bring-up time, zero firmware changes).

This was chosen over device-side interactivity (mic/STT, TTS, screen,
keyboard) for v1 because none of those meaningfully touch the real
bottleneck — FlexSPI2 weight bandwidth — but all of them sit downstream of a
first hardware bring-up that has not happened yet. A host CLI is upstream of
bring-up, testable immediately, and is the seam voice or a web UI would plug
into later.

## Goals

- Type a prompt, watch the completion stream token by token, see tok/s.
- Identical UX against `tq_run` (local subprocess) and the device (serial).
- Usable before the firmware has ever run; no firmware changes required.

## Non-goals (v1)

- No firmware-side machine protocol. The Zephyr shell is scraped as-is; a
  clean framed protocol remains a later swap if scraping proves brittle.
- No voice, no web UI, no Ethernet. The backend seam is where they attach.
- No mid-generation abort on serial (`cmd_gen` blocks the shell thread; the
  CLI documents this and resyncs at the next prompt).

## Shape

- One file: `tools/chat.py`, Python 3, ruff-clean (`ruff check tools/`).
- Stdlib only, except `pyserial`, imported lazily inside `SerialBackend` so
  the local backend runs without it installed.

## Architecture

```
REPL loop ──> Backend.generate(prompt, opts) -> Iterator[str]   # text chunks
              Backend.stats() -> GenStats | None                # after a turn
              Backend.command(name, args)                       # /load, /info

LocalBackend   subprocess: host/tq_run MODEL.etq -i <prompt> -n ... ;
               stdout = token stream (tq_run flushes per token),
               stderr = stats footer.
SerialBackend  pyserial to the CDC ACM port; writes
               `tinyllm gen -n N "<prompt>"`, discards the command echo,
               yields bytes as text until the `--` footer, parses the
               stats line, resyncs on the `tinyllm> ` prompt. The prompt is
               sent as a single double-quoted argv token: the pinned Zephyr
               shell caps argv at SHELL_ARGC_MAX=20 tokens and its tokenizer
               consumes bare `'` and `\`, so an unquoted multi-word prompt
               can overflow argc or lose characters.
```

Framing/parsing logic (echo stripping, footer detection, stats parsing,
log-line filtering) lives in pure functions on `str`/`bytes` so it is
testable without a serial port.

## REPL UX

- Bare text → a generation. Streamed output, then a dim one-line stats
  summary (`64 tok, 3.97 tok/s`).
- Slash commands: `/n`, `/t`, `/p`, `/s` set steps, temperature, top-p, seed
  (verified: `cmd_gen` and `tq_run` accept the identical flags); `/load
  <file>` sets the model path for subsequent spawns (local) or issues
  `tinyllm load` (serial); `/info` passes through on serial and prints a
  one-line backend description locally; `/q` quits.
- Serial prompts are capped client-side to fit the tighter of the firmware's
  192-byte `cmd_gen` prompt buffer and the 256-byte `SHELL_CMD_BUFF_SIZE`,
  with an explicit error — the firmware silently drops overflowing words, so
  the cap must live in the CLI.

## Error handling

- Serial uses two timeouts: a generous first-byte timeout (a `load` runs
  ~1 s per 20 MB from SD) and a shorter inter-byte timeout once streaming.
- `shell_error` output (e.g. "no model loaded") is surfaced as an error, not
  echoed as token text.
- Ctrl-C: locally, kills the `tq_run` subprocess; on serial, prints that the
  device cannot abort mid-generation and resyncs to the next `tinyllm> `
  prompt.
- Deferred-log lines interleaved in serial output are filtered by the
  framing layer, not shown as tokens.

## Testing

- pytest in `tools/tests/test_chat.py`: framing/parsing against canned
  transcripts (normal gen, interleaved deferred-log line, `shell_error`,
  truncated stream), plus a `LocalBackend` test against a stub executable
  standing in for `tq_run`.
- Wired into the existing CI `python` job alongside `ruff check tools/`.

## Future directions (explicitly deferred)

- Voice loop on the host (mic → STT → generate → TTS) behind the same
  backend seam.
- Ethernet transport once the magjack is fitted and Zephyr networking is up.
- Firmware-side framed protocol, only if shell scraping demonstrates real
  brittleness on hardware.

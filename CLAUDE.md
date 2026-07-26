# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A quantized transformer language model running on a Teensy 4.1 with
hand-soldered PSRAM. Conventions are inherited from the `ET-embed` repo
(Zephyr + west, C99, tab indentation, `snake_case` functions, `PascalCase`
typedefs, `UPPER_SNAKE_CASE` constants).

**Read [`docs/STATUS.md`](docs/STATUS.md) first.** It records exactly which
claims are verified by execution and which are untested because no hardware
was present when this was built. Do not assert that something works if that
page says it is unverified.

## The one fact that governs every decision

Decoding is **memory-bandwidth bound by ~12×** — the Cortex-M7 is idle about
92% of every token, waiting on FlexSPI2. Before proposing any optimization,
check whether it reduces *bytes read per token*. If it does not, it almost
certainly does not matter. See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Layout

| Path | Notes |
|---|---|
| `core/` | `libtq`. Portable C99, no malloc, no stdio. Shared verbatim between host and firmware — there is exactly one copy of the numerics. |
| `firmware/teensy41-tinyllm/` | Zephyr app. `psram_flexspi2.c` owns the FlexSPI2 peripheral end to end (see decision D4 in `docs/RESEARCH.md`). |
| `host/` | Native build + the test suite. |
| `tools/etq/` | Converter, quantizers, and the NumPy reference oracle. |

## Testing contract

`host/tests/test_all.c` is the safety net for the device. It runs three ways
and all three must pass:

```bash
make -C host test        # native, generic C99 kernels
make -C host test-arm    # SMLAD kernels executed under qemu-arm
make -C host size-m7     # bare-metal Cortex-M7 build + size
```

- The full forward pass is checked against `tools/etq/reference.py`, a NumPy
  oracle that mirrors the C operation for operation. Worst relative logit
  error must stay under 2e-3 (it is currently 7.9e-06).
- `test-arm` is not optional. The `SSUB8`/`SXTB16`/`SMLAD` path cannot be
  exercised natively, and it is exactly where a nibble-order bug would hide.
  The Makefile greps the disassembly to confirm `smlad` is really present.
- `host/tests/golden.h` is **generated and committed**. After any change to
  the quantizers, the container, or the reference forward pass, run
  `make -C host golden` and commit the result. CI regenerates and diffs, so a
  stale golden file fails the build rather than silently weakening the test.

## Things that will bite you

- **Q4_0 blocks are 18 bytes**, so half of them are only 2-byte aligned. All
  32-bit loads of block payloads go through `memcpy` (`tq_ld32`). Do not
  "simplify" that to a pointer cast.
- **Nibble packing is split, not sequential.** Low nibble of `qs[j]` is
  element `j`; high nibble is element `j+16`.
- **The scale is rounded to fp16 before quantizing against it.** Both the C
  and the Python must do this in the same order or every block picks up a
  systematic bias.
- **`tq_sample()` consumes its `logits` argument** — the softmax is in place.
- **`tq_view()` on a streaming store returns a pointer into a shared tile**
  that the next call invalidates. Copy anything you need to keep.
- **PSRAM works without an MPU region only because Zephyr enables
  `PRIVDEFENA`.** Turning on `CONFIG_USERSPACE` breaks this.
- **Never regenerate `golden.h` to make a failing test pass.** That inverts
  the entire point of it.

## Conventions

- C99, tabs, Zephyr style. `-Werror` is scoped to our own targets, not global.
- Python is linted with `ruff check tools/`.
- Comments explain *why*, especially where a choice looks odd (the split
  nibbles, the bespoke PSRAM driver, the two-arena API). Do not narrate what
  the code already says.

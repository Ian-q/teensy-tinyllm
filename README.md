# teensy41-tinyllm

A quantized transformer language model running on a **Teensy 4.1** — 600 MHz
Cortex-M7, no operating system, no network, and up to **32 MB of PSRAM**
hand-soldered to the two QSPI pads on the underside of the board.

The ESP32 demos that made the rounds run a 260K-parameter model in 2 MB of
PSRAM. A fully-populated Teensy 4.1 has **16× the memory**, **2.5× the clock**,
a hardware FPU, DSP MAC instructions, and a 4-bit SDIO card slot. This project
finds out what that actually buys you.

Short answer: a **42-million-parameter** model resident in RAM at ~1.5–2.3
tok/s, a **15M** model at ~4–6 tok/s, and — by streaming weights off the SD
card instead of holding them — a **110M** model at ~0.5 tok/s, which is
about 4× larger than the board's entire physical memory.

> **Status: everything except the soldering is done.** The inference engine is
> written, tested, and verified against a NumPy oracle; the Cortex-M7 DSP
> kernels have been executed under emulation and produce bit-identical results;
> the FlexSPI2 PSRAM driver is written against the RT1062 reference manual with
> its command encodings machine-checked against PJRC's reference
> implementation. What has **not** happened is anyone soldering chips to a
> board and running it. See [docs/STATUS.md](docs/STATUS.md) for exactly what
> is proven and what is not.

---

## What's here

| Path | What it is |
|---|---|
| `core/` | `libtq` — the inference engine. Portable C99, no malloc, no libc beyond `memcpy` and `libm`. 7.9 KB of Cortex-M7 code, zero static RAM. |
| `firmware/teensy41-tinyllm/` | Zephyr application: FlexSPI2 PSRAM driver, SD model loader, shell. |
| `host/` | The same engine built natively, plus the test suite. |
| `tools/etq/` | Model converter and quantizers (`llama2.c` and HuggingFace inputs). |
| `docs/` | Hardware BOM, soldering guide, bring-up procedure, performance model. |

## Start here

- **[docs/HARDWARE.md](docs/HARDWARE.md)** — what to buy, and which of the
  three possible memory configurations to build.
- **[docs/SOLDERING.md](docs/SOLDERING.md)** — the actual fiddly bit.
- **[docs/BRINGUP.md](docs/BRINGUP.md)** — first power-on to first token.
- **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** — why this is memory-bound by
  a factor of twelve, and what that predicts.
- **[docs/RESEARCH.md](docs/RESEARCH.md)** — prior art, sources, and the
  reasoning behind each design decision.

## Quickstart (no hardware needed)

The engine runs natively. This is the fastest way to see it work:

```bash
# build a tiny synthetic model and run the full test suite
make -C host test

# generate from it
make -C host run
```

To run the Cortex-M7 DSP kernels — the real ones, `SSUB8`/`SXTB16`/`SMLAD` —
without a Teensy:

```bash
sudo apt install gcc-arm-linux-gnueabihf libc6-dev-armhf-cross qemu-user-static
make -C host test-arm
```

Code size on the actual target:

```bash
sudo apt install gcc-arm-none-eabi libnewlib-arm-none-eabi
make -C host size-m7
```

## Converting a real model

```bash
pip install numpy
# llama2.c checkpoints (this is the path for the stories* models)
python3 -m etq.convert llama2c stories15M.bin tokenizer.bin stories15M.etq --q4 -v

# or a HuggingFace Llama-architecture directory (safetensors read directly;
# no torch, no safetensors package required)
python3 -m etq.convert hf ./some-llama-model out.etq --q4
```

Copy the `.etq` to a FAT32 microSD, put it in the Teensy, and:

```
tinyllm> tinyllm psram sweep
tinyllm> tinyllm load stories15M.etq
tinyllm> tinyllm gen Once upon a time
```

## Design in one paragraph

Token generation reads every weight exactly once and reuses none of them, so
it is a pure memory-bandwidth problem — on this hardware, compute is idle
about 92% of the time. Every decision follows from that: 4-bit weights (half
the bytes of int8), scales interleaved into the blocks rather than kept in a
parallel array (one sequential read stream instead of two), the FlexSPI2 clock
pushed as high as the soldered parts will tolerate (a measured per-board
property, so the firmware sweeps for it), activations in DTCM and weights in
PSRAM, and `SMLAD` on the Cortex-M7 so the arithmetic stays comfortably ahead
of the bus. The result is that predicted token rate is just
`PSRAM MB/s ÷ model MB` — which is a satisfying place to end up, because it
means the only thing left to optimize is bytes.

## Licence

MIT. See [LICENSE](LICENSE).

The `.etq` container, quantizers and inference structure are original, but the
transformer is Llama-2 and the debt to Andrej Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c) is obvious and gratefully
acknowledged; the block-quantization layout follows the conventions
[ggml](https://github.com/ggml-org/llama.cpp) established. The FlexSPI2
bring-up sequence follows PJRC's `cores/teensy4/startup.c`, which is the
reference this hardware is known to work with.

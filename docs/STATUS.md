# What is proven, and what is not

This project was written without a Teensy 4.1 present and first met real
hardware on 2026-07-31. It now runs end to end: a quantized Llama generates
TinyStories prose from hand-soldered PSRAM. This page records what is proven
by execution and what still is not, so nobody debugs the wrong layer.

## Verified by execution

| Claim | How |
|---|---|
| The inference engine computes the right thing | 602 assertions in `host/tests/test_all.c`. The full forward pass matches an independent NumPy reference (`tools/etq/reference.py`) to **7.9e-06 worst relative logit error** over 8 decode steps. Covers Q4_0 dequantization, grouped-query attention, RoPE, SwiGLU, RMSNorm and the shared classifier. |
| The Cortex-M7 DSP kernels are correct | The `SSUB8`/`SXTB16`/`SMLAD` path is cross-compiled for ARMv7 with `__ARM_FEATURE_DSP` and **executed under `qemu-arm`**, producing the identical 7.9e-06 figure. The build asserts `smlad` instructions are actually present in the binary. Same instructions the M7 issues, same semantics. |
| Q4/Q8 kernels match an independent reference | 200 randomised trials per dtype against a deliberately differently-written reference (full dequantize, then dot in `double`), across three magnitude regimes plus an all-zero block. |
| The SD-streaming path is arithmetically identical to the resident path | A simulated tiled read backend reproduces the mapped-store logits exactly (same 7.9e-06). Tiling changes no arithmetic. |
| int8 KV cache is an acceptable approximation | Worst error **4.9% of logit RMS** versus the fp32 golden. |
| The tokenizer round-trips | Encode/decode over 9 cases including multi-byte UTF-8 byte-fallback and BOS/EOS bracketing. |
| Malformed model files are rejected | Bad magic, bad version, truncation, and undersized arena all return the right error instead of executing. |
| The engine fits | **10,550 bytes** of Cortex-M7 `.text` at `-Os`, **0 bytes** of `.data` and `.bss`. Built with real `arm-none-eabi-gcc`. |
| The Python exporter and the C parser agree | The C test suite parses a container the Python writer produced; struct layouts are pinned by `static_assert` on the C side and `struct.calcsize` on the Python side. |
| The FlexSPI2 LUT encodings are right | All 13 command words machine-compared against the macros in PJRC's `imxrt.h`. Register addresses and bitfield positions taken from the same source. |
| **The Zephyr firmware compiles and links** | `west build -b teensy41` passes in CI (first green: 2026-07-29) and uploads `zephyr.hex`/`zephyr.elf` as the `zephyr-hex` artifact, with `smlad` confirmed present in the image. The first builds surfaced and fixed: a wrong FatFS Kconfig symbol, an unsatisfiable `EXCEPTION_STACK_TRACE`, base-address macro collisions with the MCUX SDK header, a missing USBD-next device context (the CDC-ACM console had no instance at all), and the activation arena overflowing `zephyr,sram` — it now lives in the otherwise-unused 512 KB OCRAM2 (256 KB arena + 32 KB tile = 56% used; kernel RAM at 45% of 256 KB). |
| **Both PSRAM chips enumerate — under PJRC's code** | 2× APS6404L-3SQR hand-soldered; PJRC's `teensy41_psram_memtest` reports 16 Mbyte and passes repeated pseudo-random sweeps at 88 MHz (2026-07-28). This validates the solder joints and chips, not this repo's driver. |
| **This repo's driver runs the chips, measured** | On the real board (2026-07-31): 16 MB across both chips, ids `0x32535d0d`. First contact found two driver bugs — MCR0's ATDFEN/ARDFEN reset to 1 and routed the IP RX FIFO to DMA (every ID read returned zeros), and the clock table assumed PJRC's PLL-PFD programming while Zephyr's differs (rows labeled 166 MHz really ran at ~65). Frequencies are now computed from the live CCM_ANALOG registers. Sweep result: clean linear scaling 13.2→31.3 MB/s from 41→105.6 MHz, **settled at 105.6 MHz / 31.4 MB/s sustained read** (the model predicted 34 — 92% achieved). CS1 drops out at ~120 MHz real; the sweep now counts that as failure, not an 8 MB "pass". Boot default (index 3) is already this board's optimum. |
| **THE WHOLE THING RUNS** *(2026-07-31)* | `stories15M.etq` loads from SD into PSRAM and generates coherent TinyStories prose on the Teensy: **2.98 tok/s, 309 ms to first token, 8.3 MB read/token, 25.5 MB/s effective**. Bring-up found five real defects — MCR0 ATDFEN/ARDFEN (PSRAM read as absent), a clock table assuming PJRC's PFD programming, a missing USBD device context, SD DMA into cacheable buffers, and a `tq_carve` alignment bug that faulted the parser. |
| **A real checkpoint converts correctly** *(2026-07-31)* | `convert.py llama2c` run on the real `stories15M.bin` + llama2.c `tokenizer.bin`: 9.12 MB Q4 file, and `tq_run` generates coherent TinyStories English from it on the host. The HuggingFace/safetensors path remains exercised only against the synthetic fixture. |
| **The Semaphore codec round-trips** *(2026-08-01)* | `core/src/tq_ac.c` codes a message against the model's own predictions. Exact round trip on all 18 messages of both corpora in `tools/corpora/`, at **1.100 bits/char** in the TinyStories register and **3.293** on out-of-domain operational messages — 1.79x smaller than dictionary-primed deflate, the strongest classical baseline. That baseline independently reproduces `tools/semaphore_probe.py`'s figure to three digits (5.886 vs 5.89). The coder is exercised standalone against explicit count tables (uniform, 65520/65536-skewed to force carry propagation, and ragged) as well as against real logits. 1562 bytes of `.text` on Cortex-M7, no RAM. |
| **Logits are bit-identical across architectures** *(2026-08-01)* | `make -C host test-bitexact` hashes the bit pattern of every golden logit and requires the native and qemu-arm runs to match. They do: **`5dcac5e05353b45c`** across macOS arm64 with the generic C99 kernels and armv7-linux under QEMU with the `SSUB8`/`SXTB16`/`SMLAD` kernels — different OS, libc, compiler, architecture, word size and kernel implementation. This is what lets a laptop be the far end of a Semaphore decode. It required replacing `expf`/`sinf`/`cosf`/`powf` with `core/src/tq_math.c` and compiling with `-ffp-contract=off`: libm's own results for those four functions differ between the two platforms on **1.21% of the values the forward pass uses** (958 of 78924, all last-ulp), which at ~8000 `expf` calls per token would desynchronise a decoder almost immediately. `sqrtf` is exempt — IEEE-754 requires it to be correctly rounded. |
| **Semaphore runs on the board, and interoperates with the host** *(2026-08-02)* | `tinyllm sem encode/decode/hash` on the real Teensy. `tinyllm sem hash` and `tq_sem stories15M.etq hash` both print **`4aba58abe4d1c3fa`** — 8 forward passes x 32000 logits, 256,000 floats, identical to the bit between a Cortex-M7 running the `SMLAD` kernels and macOS arm64 running the generic C99 ones. So either end can be either side, and it was: the host compressed 70 characters to **5 bytes** (`11536c21ff`) and the board recovered the text exactly; the board compressed 59 characters to 10 bytes and the host recovered those. On three separate messages the board's wire bytes were byte-identical to the host's own encoding of the same string. Cost is **~325 ms per token each way** — a 70-character message is about 5.5 s to encode and 5.5 s to decode. |
| **USB CDC-ACM console comes up** *(verified)* | Enumerates on macOS as `/dev/cu.usbmodem*`; the shell, `psram`, `load`, and `gen` all drive over it. `src/usb_console.c` supplies the device context the USBD-next stack requires and the original firmware lacked. `src/usb_reboot.c` adds PJRC's baud-134 convention, so reflashing needs no button — except after a hang, when nothing services the poll. |

## Not verified — needs the hardware

| Claim | Risk |
|---|---|
| **The clock sweep finds a high clock** | Entirely a property of your solder joints and your specific chips. |
| **Predicted token rates** | Every number in [PERFORMANCE.md](PERFORMANCE.md) is a bandwidth model with one free parameter. `tinyllm psram bench` measures it and `tinyllm gen` reports achieved throughput, so the prediction is falsifiable in thirty seconds — but it has not been falsified or confirmed. |

## Known gaps in the implementation

- **No batched prefill.** Prompt tokens cost a full forward pass each. This is
  the one place batching would genuinely help, since it raises arithmetic
  intensity, and it is not implemented.
- **Streaming mode has no tokenizer.** BPE encode does thousands of tiny random
  reads; over SD that is pathological. Streaming emits raw token ids.
- **Q4_0 only, plus Q8_0.** No sub-4-bit, no mixed precision, no importance
  matrices. The classifier — 61% of the bytes on a small model — is quantized
  the same as everything else, which is leaving the largest single win on the
  table. See [PERFORMANCE.md](PERFORMANCE.md).
- **No eDMA double-buffering.** Argued against in PERFORMANCE.md on the grounds
  that the CPU is already 92% idle, but the argument deserves a measurement.

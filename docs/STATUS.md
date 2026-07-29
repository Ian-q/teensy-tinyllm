# What is proven, and what is not

This project was built and verified without a Teensy 4.1 physically present.
That is a real limitation and this page states exactly where the line falls,
so nobody debugs the wrong layer.

## Verified by execution

| Claim | How |
|---|---|
| The inference engine computes the right thing | 554 assertions in `host/tests/test_all.c`. The full forward pass matches an independent NumPy reference (`tools/etq/reference.py`) to **7.9e-06 worst relative logit error** over 8 decode steps. Covers Q4_0 dequantization, grouped-query attention, RoPE, SwiGLU, RMSNorm and the shared classifier. |
| The Cortex-M7 DSP kernels are correct | The `SSUB8`/`SXTB16`/`SMLAD` path is cross-compiled for ARMv7 with `__ARM_FEATURE_DSP` and **executed under `qemu-arm`**, producing the identical 7.9e-06 figure. The build asserts `smlad` instructions are actually present in the binary. Same instructions the M7 issues, same semantics. |
| Q4/Q8 kernels match an independent reference | 200 randomised trials per dtype against a deliberately differently-written reference (full dequantize, then dot in `double`), across three magnitude regimes plus an all-zero block. |
| The SD-streaming path is arithmetically identical to the resident path | A simulated tiled read backend reproduces the mapped-store logits exactly (same 7.9e-06). Tiling changes no arithmetic. |
| int8 KV cache is an acceptable approximation | Worst error **4.9% of logit RMS** versus the fp32 golden. |
| The tokenizer round-trips | Encode/decode over 9 cases including multi-byte UTF-8 byte-fallback and BOS/EOS bracketing. |
| Malformed model files are rejected | Bad magic, bad version, truncation, and undersized arena all return the right error instead of executing. |
| The engine fits | **7,946 bytes** of Cortex-M7 `.text` at `-Os`, **0 bytes** of `.data` and `.bss`. Built with real `arm-none-eabi-gcc`. |
| The Python exporter and the C parser agree | The C test suite parses a container the Python writer produced; struct layouts are pinned by `static_assert` on the C side and `struct.calcsize` on the Python side. |
| The FlexSPI2 LUT encodings are right | All 13 command words machine-compared against the macros in PJRC's `imxrt.h`. Register addresses and bitfield positions taken from the same source. |
| **The Zephyr firmware compiles and links** | `west build -b teensy41` passes in CI (first green: 2026-07-29) and uploads `zephyr.hex`/`zephyr.elf` as the `zephyr-hex` artifact, with `smlad` confirmed present in the image. The first builds surfaced and fixed: a wrong FatFS Kconfig symbol, an unsatisfiable `EXCEPTION_STACK_TRACE`, base-address macro collisions with the MCUX SDK header, a missing USBD-next device context (the CDC-ACM console had no instance at all), and the activation arena overflowing `zephyr,sram` — it now lives in the otherwise-unused 512 KB OCRAM2 (256 KB arena + 32 KB tile = 56% used; kernel RAM at 45% of 256 KB). |
| **Both PSRAM chips enumerate — under PJRC's code** | 2× APS6404L-3SQR hand-soldered; PJRC's `teensy41_psram_memtest` reports 16 Mbyte and passes repeated pseudo-random sweeps at 88 MHz (2026-07-28). This validates the solder joints and chips, not this repo's driver. |

## Not verified — needs the hardware

| Claim | Risk |
|---|---|
| **PSRAM enumerates under this repo's driver** | The chips and joints are proven good by PJRC's memtest (see above), and the LUT words are verified, but `psram_flexspi2.c` itself has not driven the pins yet. |
| **The clock sweep finds a high clock** | Entirely a property of your solder joints and your specific chips. |
| **Predicted token rates** | Every number in [PERFORMANCE.md](PERFORMANCE.md) is a bandwidth model with one free parameter. `tinyllm psram bench` measures it and `tinyllm gen` reports achieved throughput, so the prediction is falsifiable in thirty seconds — but it has not been falsified or confirmed. |
| **A real checkpoint converts correctly** | `tools/etq/convert.py` was written against the documented `llama2.c` and HuggingFace formats but never run on one — the sandbox could not reach HuggingFace. It is exercised end to end against a synthetic model with the same code path. Verify with step 10 of [BRINGUP.md](BRINGUP.md) (run the converted model on the host) before blaming the board. |
| **USB CDC-ACM console comes up** | The USBD-next stack requires an application-defined device context, which the original firmware lacked entirely — `src/usb_console.c` now provides it (found at the first link: the chosen console had no device instance). Compiles and links; enumeration on real hardware is unexercised. |

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

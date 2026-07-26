# Research notes and design decisions

## Prior art

**The ESP32 wave.** Dave Bennett's [`esp32-llm`](https://github.com/DaveBben/esp32-llm)
runs Karpathy's 260K-parameter `tinyllamas` checkpoint on an ESP32-S3 with 2 MB
of embedded PSRAM at **19.13 tok/s**, using both cores and the ESP-DSP dot
products. Later work reports a 28.9M-parameter model at ~9.5 tok/s by keeping
per-layer embeddings in flash, and an 8M variant at ~12 tok/s. There is also a
[demo](https://github.com/bertaye/wifi-llm) that streams a 16 MB quantized
llama2 over WiFi into an ESP32 with 300 KB of RAM, which is a nice statement of
how far the streaming idea can be pushed.

**Where the Teensy 4.1 sits.** Against the ESP32-S3: 2.5× the clock (600 vs
240 MHz), a hardware FPU, ARM DSP MAC instructions, 1 MB of on-chip SRAM
instead of ~512 KB, up to 16× the external RAM (32 MB vs 2 MB), and a 4-bit
SDIO slot at ~22 MB/s. Against that, the ESP32-S3 has two cores and this has
one, and its PSRAM sits on a similar-width QSPI bus — so the Teensy's advantage
in *bandwidth* is real but modest (roughly 34–53 MB/s against the ESP32-S3's
~40 MB/s at 80 MHz octal), while its advantage in *capacity* is enormous.

That shapes the goal: this is not a project about being faster per parameter.
It is about running a model an order of magnitude larger.

## Sources

Hardware and bandwidth:

- [PJRC — PSRAM chip for Teensy 4.1](https://www.pjrc.com/store/psram.html)
- [ProtoSupplies — Working with Teensy 4.1 memory](https://protosupplies.com/learn/prototyping-system-for-teensy-4-1-working-with-teensy-4-1-memory/) — the 88 MHz / 44 MB/s theoretical figure, the 105.6 MHz Teensyduino 1.60 default, and the 132 MHz ceiling
- [ProtoSupplies — 16MB PSRAM for Teensy 4.1](https://protosupplies.com/product/psram_16mb/) — the ISSI IS66WVS16M8FBLL, two 8 MB dies per package, reliable at 120 MHz
- [PJRC forum — Teensy 4.1 PSRAM memtest](https://forum.pjrc.com/threads/71975-Teensy-4-1-psram-memtest) — the measured ~28 MB/s `memcpy` read at 88 MHz that anchors the efficiency estimate
- [PJRC forum — SdFat SDIO for Teensy 4](https://forum.pjrc.com/threads/57669-SdFat-SDIO-for-Teensy-4-0) — 22.9 MB/s sequential read over 4-bit SDIO
- [PJRC `cores/teensy4/startup.c`](https://github.com/PaulStoffregen/cores/blob/master/teensy4/startup.c) — the reference FlexSPI2 bring-up: pad muxing, clock table, LUT contents, JEDEC ID decode
- [NXP AN13028 — Advanced HyperRAM/PSRAM usage on i.MX RT](https://www.nxp.com/docs/en/application-note/AN13028.pdf)

Software:

- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) — the architecture, the `.bin` checkpoint format, the tokenizer export, and `runq.c`'s int8 approach
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) — the Q4_0/Q8_0 block layout and split-nibble packing
- [ARM CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) — the `SXTB16` + `SMLAD` int8 dot-product pattern
- [zephyr#83244](https://github.com/zephyrproject-rtos/zephyr/issues/83244) — the in-tree APS6404L driver does not build for `teensy41` (fixed upstream in #97380, after our pinned v4.0.0)

## Decisions

### D1 — Q4_0 as the default, not int8

Decoding is memory-bound by ~12× ([PERFORMANCE.md](PERFORMANCE.md)), so bits
per weight converts almost perfectly into token rate. Q4_0 is 1.9× fewer bytes
than Q8_0 and therefore 1.9× the speed. Q8_0 is kept because it is a useful
accuracy reference when a Q4 model misbehaves, and because at these model
sizes Q4 is visibly lossy.

Sub-4-bit was considered and rejected *for now*: it keeps paying at the same
rate, but doing it well needs importance matrices or per-channel scales rather
than round-to-nearest, and that is a research project rather than an
implementation detail.

### D2 — Scales interleaved into blocks, not in a parallel array

`llama2.c`'s `runq.c` stores each tensor's int8 payload and its fp32 scales as
two separate runs. That is fine when you have a memory controller with many
outstanding requests. On FlexSPI2 it means two concurrent sequential streams
contending for the same AHB RX buffers, which defeats the prefetcher on the
one access pattern this workload consists entirely of. ggml's layout — a
2-byte fp16 scale immediately followed by its 16 or 32 payload bytes — makes a
whole matrix row one unbroken forward read.

### D3 — Split-nibble packing for Q4

Byte `j` of a block holds element `j` in its low nibble and element `j+16` in
its high nibble. Sequential packing would be easier to read; split packing
means one 32-bit load plus a mask and a shift yields two independent runs of
four int8 values, which is exactly what `SSUB8`/`SXTB16`/`SMLAD` want. Same
reason ggml does it.

### D4 — A bespoke FlexSPI2 driver instead of Zephyr's

Three reasons, any one of which would be sufficient: Zephyr's
`memc_mcux_flexspi_aps6404l` does not compile for `teensy41` on our pinned
v4.0.0 (zephyr#83244); it assumes a single 8 MB APS6404L and cannot describe
two chips or the 16 MB ISSI part's ID encoding; and it fixes the clock at
devicetree-configure time, when the single most valuable thing we can do is
sweep the clock at runtime and memtest each step.

The driver declares its registers from the RT1062 reference manual rather than
including MCUX SDK headers, so it does not break when the HAL revision in the
workspace changes. Every LUT word is machine-compared against PJRC's macros in
`host`-side tooling.

### D5 — Zephyr, not Teensyduino

Teensyduino has mature, well-tested PSRAM support (`EXTMEM`, 16 MB parts since
1.60) and would have been the lower-risk choice for the PSRAM specifically.
Zephyr wins on everything else: a real shell over CDC-ACM, a FAT filesystem on
the SDIO slot, `west` pinning the whole toolchain for reproducible CI, and a
build that matches the conventions of the repo this grew out of. Since the
PSRAM driver was going to be bespoke either way (D4), Teensyduino's main
advantage evaporated.

### D6 — Two caller-owned arenas, no allocator

`libtq` never calls `malloc`. The caller passes a "fast" arena for activations
and a "cache" arena for the KV cache, and the firmware points those at DTCM
and PSRAM respectively. An allocator would have to be told about the split
anyway, and this way the host build and the device build share the exact same
code with different pointers.

### D7 — int8 KV cache

fp32 KV for `stories42M` at a 1024-token context is 33.5 MB — more than the
entire board. int8 with a per-head scale is 8.9 MB and costs 4.9% of logit RMS
(measured, `host/tests/test_all.c`). Per-head rather than per-vector because
attention heads routinely differ by an order of magnitude in scale and a
shared scale crushes the quiet ones.

### D8 — Verify the DSP kernels under QEMU

The `SMLAD` path is the one piece of the engine that cannot be exercised by a
native x86 test, and it is exactly where a nibble-order or byte-pairing
mistake would hide. `armv7-a` defines `__ARM_FEATURE_DSP` identically to the
Cortex-M7 and `qemu-arm` executes those instructions faithfully, so
`make -C host test-arm` runs the real kernels and asserts they produce the same
answers as the portable path. The Makefile also greps the disassembly to
confirm `smlad` is actually present, because a silently-disabled intrinsic that
falls back to portable C would otherwise pass.

## Model shopping list

What actually fits, at Q4_0, on 32 MB:

| Model | Params | Q4 | Fits resident? | Notes |
|---|---|---|---|---|
| `stories260K` | 0.26M | 0.2 MB | trivially | The ESP32 demo model. Runs at hundreds of tok/s here; not interesting. |
| `stories15M` | 15.2M | 8.5 MB | yes, even on 8 MB+ | **Recommended first model.** Coherent children's stories at ~4–6 tok/s. |
| `stories42M` | 41.7M | 23.4 MB | yes, needs 32 MB | **The flagship.** ~1.5–2.3 tok/s with a 512-token int8 KV cache. |
| `stories110M` | 109.5M | 61.6 MB | no — stream from SD | ~0.5 tok/s. 4× the board's physical memory. |
| SmolLM2-135M | 135M | 76 MB | no — stream | Untested; a real instruction-ish model rather than a story generator. |
| Qwen3-0.6B and up | 600M+ | 340 MB+ | no | Streaming at ~22 MB/s puts this at ~0.06 tok/s. A token every 16 seconds is a stunt, not a demo. |

The ceiling for a *resident* model on this board is about **50M parameters at
Q4_0**, and `stories42M` sits right under it. That is the number this whole
exercise was trying to find.

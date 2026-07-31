# Performance

## The one fact that determines everything

Autoregressive decoding of one token reads **every weight in the model exactly
once** and reuses none of them. There is no batch dimension to amortise
against — a matrix-vector product does one multiply-accumulate per weight
loaded. So the arithmetic intensity is fixed at roughly 1 MAC per byte, and
the only question is which side of the machine gives out first.

On a Teensy 4.1 it is not close:

| | Rate | Time for a 42M-param Q4 model |
|---|---|---|
| Compute — Cortex-M7 @ 600 MHz, `SMLAD`, derated 60% | ~720 MMAC/s | 58 ms |
| Memory — PSRAM @ 105.6 MHz | ~34 MB/s | **690 ms** |

**Memory loses by 11.9×.** The core spends about 92% of every token waiting on
the FlexSPI2 bus. That ratio is essentially constant across model sizes,
because both terms scale linearly with parameter count.

Two consequences that drove the whole design:

1. **Bits per weight is the only lever that matters.** Q4_0 versus Q8_0 is a
   straight 1.9× on token rate. Nothing else available comes close.
2. **CPU overclocking is pointless.** Running the Teensy at 816 MHz would take
   compute from 58 ms to 43 ms against a 690 ms memory bill. Spend the effort
   on the FlexSPI2 clock instead, where it converts 1:1 into tokens.

## Predicted token rate

`tok/s = PSRAM read MB/s ÷ bytes read per token`

Q4_0 costs 18 bytes per 32 weights = **0.5625 bytes/weight**. Bytes read per
token is *all* layer weights plus the classifier matmul — but **not** the token
embedding table, which is only row-indexed (one row per token, ~1 KB).
For a shared-classifier model the embedding table is read once as the
classifier, so it appears exactly once in the total.

| Model | Params | Q4 file | Read/token | 88 MHz<br>28.4 MB/s | 105.6 MHz<br>34.0 MB/s | 132 MHz<br>42.5 MB/s | 166 MHz<br>53.5 MB/s |
|---|---|---|---|---|---|---|---|
| stories15M | 15.2M | 8.5 MB | 8.5 MB | 3.3 | **4.0** | 5.0 | 6.3 |
| TinyStories-27M class | 27.4M | 15.4 MB | 15.4 MB | 1.8 | **2.2** | 2.8 | 3.5 |
| stories42M | 41.7M | 23.4 MB | 23.4 MB | 1.2 | **1.5** | 1.8 | 2.3 |
| stories110M *(SD-streamed)* | 109.5M | 61.6 MB | 61.6 MB | — | — | — | ~0.5† |

† Streamed from SD at ~22 MB/s, not PSRAM. Does not fit in 32 MB at any
quantization this project implements.

These are predictions from a bandwidth model, not measurements. The model has
one free parameter — sustained PSRAM read bandwidth — and `tinyllm psram bench`
measures it directly on your board, so the first thing bring-up does is
replace the estimate with the real number. `tinyllm gen` then reports achieved
effective MB/s next to the token rate, so the prediction is falsifiable in
about thirty seconds.

**Measured (first board, 2× APS6404L-3SQR, 2026-07-31):** 31.4 MB/s
sustained read at 105.6 MHz — 92% of the 34 MB/s the table assumes for that
clock, scaling linearly from 13.2 MB/s at 41 MHz. The higher columns are out
of reach on this board: CS1 drops off the bus around 120 MHz real, and
nothing enumerates at 132+. Predicted `stories15M` rate at the measured
bandwidth: ~3.7 tok/s.

## Where the bytes actually go

For `stories15M` (dim 288, 6 layers, vocab 32000):

| Component | Params | Share of each token |
|---|---|---|
| Classifier / embedding matmul (32000 × 288) | 9.2M | **61%** |
| 6 × attention (wq, wk, wv, wo) | 2.0M | 13% |
| 6 × FFN (w1, w2, w3) | 4.0M | 26% |

**The classifier is the majority of the work**, and it is that big only
because the Llama tokenizer has a 32000-token vocabulary — enormous next to a
288-dimension model. The single highest-leverage optimization left in this
project is not a faster kernel; it is a smaller vocabulary. A 4096-token
tokenizer retrained on the same corpus would cut `stories15M`'s per-token read
from 8.5 MB to 3.9 MB and roughly **double** the token rate.

Two cheaper variants of the same idea, neither implemented:

- **Quantize the classifier harder than the rest.** It is the most redundant
  matrix in the model. Q3 or Q2 on the output head alone would cut 25–50% off
  its share for a quality hit concentrated in tokens the model was never going
  to pick.
- **Hierarchical / clustered softmax.** Read a coarse head to pick a bucket,
  then only the rows in that bucket. Turns an `O(vocab)` read into
  `O(√vocab)`, at the cost of exactness.

## Time to first token

Prompt processing runs the same code path per token, so a 20-token prompt
costs 20 forward passes — about 5 s on `stories15M` at 4 tok/s. There is no
batched prefill: batching would raise arithmetic intensity and genuinely help
here, but it needs a matrix-matrix kernel and enough DTCM to hold the whole
prompt's activations. Worth doing; not done.

Model load from SD is about **1 s per 20 MB** (SDIO 4-bit, ~22 MB/s
sequential), so a 23 MB `stories42M` is resident in a little over a second.

## The optimizations that are in

| | Why it pays |
|---|---|
| **Q4_0 weights** | 1.9× fewer bytes than int8. The whole ballgame. |
| **Interleaved scales** | Scales live inside each 18-byte block, ggml-style, instead of in a parallel array (llama2.c's `runq` layout). A split layout forces two concurrent read streams through one FlexSPI2 AHB RX buffer and defeats the prefetcher. |
| **FlexSPI2 AHB prefetch** | Buffers 0 and 1 get the full 512 B with `PREFETCHEN`. Weight reads are perfectly sequential, so the prefetcher works ahead almost ideally. |
| **Runtime clock sweep** | Up to 1.9× between the stock 88 MHz and a part that holds 166 MHz. Per-board, so it must be measured, not assumed. |
| **`SMLAD` kernels** | 4 int8 MACs per two instructions. Not the bottleneck, but it keeps compute far enough back that it never becomes one. |
| **Activations in on-chip SRAM (OCRAM2)** | Fast AXI access for the vectors touched thousands of times per token, while weights stream from the slow memory they only need once. True single-cycle DTCM is only 128 KB in the default FlexRAM split — smaller than a 32k-vocab logit buffer — so moving the hot vectors there means repartitioning FlexRAM: unmeasured future work, bounded by the ~8% of each token the CPU isn't stalled on FlexSPI2. |
| **One activation quantization per fan-out** | `xq` is computed once and consumed by wq/wk/wv; likewise for w1/w3. Free. |
| **RoPE tables per token, not per head** | The rotation angles depend only on position, so 32 `sinf`/`cosf` calls per token replace a few thousand. |
| **int8 KV cache** | 4× smaller than fp32 for ~5% of logit RMS in error. It is what lets a 1024-token context coexist with 24 MB of weights in 32 MB. |

## The optimizations that are out, and why

- **eDMA double-buffering of weight tiles.** The obvious next idea: DMA the
  next tile into OCRAM while the CPU works on the current one. But the CPU is
  already idle 92% of the time — the gain is bounded by the ~8% it spends
  computing, and only if the FlexSPI2 AHB prefetcher is leaving bandwidth on
  the table, which it mostly is not. Measure `psram bench` against the
  achieved MB/s in `tinyllm gen` before spending a weekend on this.
- **Both FlexSPI controllers in parallel.** FlexSPI1 holds the program flash.
  Splitting weights across both would nearly double bandwidth and would also
  mean the board cannot boot.
- **CPU overclock.** See above. Wrong bottleneck.
- **Sub-4-bit weights.** Q2/Q3 would keep paying at the same 1:1 rate, and are
  the most promising unexplored direction. Not implemented because Q4_0 at
  this model scale is already visibly lossy and going lower needs
  quantization-aware methods (importance matrices, per-channel scales) rather
  than plain round-to-nearest.

## Measuring on your board

```
tinyllm> tinyllm psram bench      # raw sequential read/write MB/s
tinyllm> tinyllm psram sweep      # fastest stable clock, memtested
tinyllm> tinyllm load stories15M.etq
tinyllm> tinyllm gen -n 64 Once upon a time
...
64 tokens in 16104 ms = 3.97 tok/s (first token 262 ms)
weights read 545259 KB total, 8520 KB/token, effective 33.86 MB/s
```

That last line is the falsification test. If `effective MB/s` is close to what
`psram bench` reported, the engine is bandwidth-limited exactly as designed and
there is no overhead left to find. If it is much lower, something is stalling —
start with log level and shell output inside the decode loop.

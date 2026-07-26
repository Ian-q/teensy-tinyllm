# The `.etq` container

Defined by `core/include/tq/tq_format.h`; written by `tools/etq/format.py`.
The C structs are pinned with `static_assert` and the Python struct format
strings are asserted against the same sizes, so the two cannot drift silently.

## Layout

```
0x0000  TqHeader          256 bytes
0x0100  TqTensorEntry[]   64 bytes each
        (padding to a 64-byte boundary)
        tensor payloads, each starting 64-byte aligned
```

64-byte alignment on payloads so that no eDMA burst or Cortex-M7 cache line
(32 B) straddles a tensor boundary.

## Header (96 bytes used, padded to 256)

| Field | Type | Notes |
|---|---|---|
| `magic` | u32 | `0x31515445` — "ETQ1" little-endian |
| `version` | u32 | 1 |
| `header_bytes` | u32 | 256 |
| `arch` | u32 | 0 = llama2 |
| `tensor_count` | u32 | |
| `tensor_table_offset` | u32 | |
| `data_offset` | u32 | |
| `flags` | u32 | reserved |
| `file_bytes` | u64 | total size, checked against the store |
| `dim` … `seq_len` | 7 × i32 | architecture |
| `qtype` | u32 | `TQ_DT_Q4_0` or `TQ_DT_Q8_0` |
| `group_size` | u32 | must be 32 |
| `shared_classifier` | u32 | 1 ⇒ the output head reuses `tok_emb` |
| `rope_theta`, `norm_eps` | 2 × f32 | |
| `bos_token`, `eos_token` | 2 × u32 | |

## Tensor entry (exactly 64 bytes)

`char name[24]`, `u32 dtype`, `u32 ndim`, `u32 shape[4]`, `u64 offset`,
`u64 nbytes`.

## Tensor names

| Name | dtype | Shape |
|---|---|---|
| `tok_emb` | quantized | `[vocab, dim]` |
| `rms_final` | f32 | `[dim]` |
| `cls` | quantized | `[vocab, dim]` — absent when `shared_classifier` |
| `lNNN.rms_att`, `lNNN.rms_ffn` | f32 | `[dim]` |
| `lNNN.wq`, `lNNN.wo` | quantized | `[dim, dim]` |
| `lNNN.wk`, `lNNN.wv` | quantized | `[kv_dim, dim]` |
| `lNNN.w1`, `lNNN.w3` | quantized | `[hidden, dim]` |
| `lNNN.w2` | quantized | `[dim, hidden]` |
| `tok.scores` | f32 | `[vocab]` |
| `tok.offs` | u32 | `[vocab+1]` — byte offsets into the blob |
| `tok.blob` | u8 | packed piece strings |
| `tok.sorted` | u32 | `[vocab]` — token ids in lexicographic order |

Matrices are row-major `[out_features, in_features]`, so one output row is one
unbroken sequential read.

`tok.sorted` is the tokenizer's index: string → id is a binary search over a
permutation computed at export time. Building that order on the device would
mean sorting 32000 strings in PSRAM before the first token.

## Block formats

**Q8_0** — 34 bytes per 32 weights:
```c
struct { uint16_t d; int8_t qs[32]; };     /* value = qs[i] * d */
```

**Q4_0** — 18 bytes per 32 weights:
```c
struct { uint16_t d; uint8_t qs[16]; };    /* value = (nibble - 8) * d */
```

`d` is IEEE-754 binary16. Nibble order is **split**: the low nibble of `qs[j]`
is element `j`, the high nibble is element `j+16`.

Q4 scales come from the *signed* extreme (`d = signed_max / -8`), not the
magnitude, so `d` may be negative. That lets the 16-level codebook cover
`[-8, +7]` without wasting a level on symmetry, and costs the decoder nothing —
dequant is `(nibble - 8) * d` either way.

### The rule that makes host and device agree

**The scale is rounded to fp16 before the values are quantized against it.**
Quantizing against an fp32 scale and then storing a rounded one puts a
systematic bias into every block. Both `tools/etq/quantize.py` and
`tq_quantize_q8()` do it in this order, and rounding is half-away-from-zero
spelled `trunc(v + copysign(0.5, v))` on both sides.

## RoPE convention

Interleaved pairs — element `2i` rotates against `2i+1` within each head,
matching `llama2.c`. HuggingFace checkpoints use the split-half convention, so
`convert.py` permutes `wq` and `wk` at export time. The device only ever sees
one convention.

## Adding a tensor

1. Emit it from `tools/etq/convert.py`.
2. Add it to the want-list in `tq_model_open()` (`core/src/tq_model.c`).
3. Regenerate the golden vectors: `make -C host golden`.
4. `make -C host test`.

CI regenerates `golden.h` into a temp directory and diffs, so a stale golden
file fails the build rather than quietly weakening the test.

#!/usr/bin/env python3
# Copyright (c) 2026 Ian Adelman
# SPDX-License-Identifier: MIT
"""Block quantizers for the .etq container.

These must agree with core/src/tq_kernels.c to the last bit. Two rules make
that possible:

  * The scale is rounded to fp16 BEFORE the values are quantized against it.
    Quantizing against an fp32 scale and then storing a rounded one injects a
    systematic bias into every single block.
  * Rounding is half-away-from-zero, spelled trunc(v + copysign(0.5, v)),
    which is what the C code does without pulling in libm's roundf.
"""

from __future__ import annotations

import numpy as np

from etq.format import DT_Q4_0, DT_Q8_0, GROUP_SIZE


def _round_half_away(v: np.ndarray) -> np.ndarray:
    return np.trunc(v + np.copysign(0.5, v))


def _blocks(w: np.ndarray) -> np.ndarray:
    w = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    if w.size % GROUP_SIZE:
        raise ValueError(f"{w.size} values is not a whole number of "
                         f"{GROUP_SIZE}-value blocks")
    return w.reshape(-1, GROUP_SIZE)


def quantize_q8_0(w: np.ndarray) -> bytes:
    """fp16 scale + 32 int8 per block, 34 bytes."""
    b = _blocks(w)
    amax = np.abs(b).max(axis=1)
    d16 = (amax / 127.0).astype(np.float16)
    d = d16.astype(np.float32)
    inv = np.where(d != 0.0, 1.0 / np.where(d != 0.0, d, 1.0), 0.0)

    q = _round_half_away(b * inv[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)

    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def quantize_q4_0(w: np.ndarray) -> bytes:
    """fp16 scale + 32 nibbles per block, 18 bytes.

    The scale is derived from the SIGNED extreme rather than the magnitude, so
    d can be negative. That is deliberate: it lets the codebook use all 16
    levels [-8, +7] instead of wasting one to stay symmetric, and it costs the
    decoder nothing — dequant is (nibble - 8) * d either way.

    Nibbles are packed SPLIT, not sequential: byte j holds element j in its low
    half and element j+16 in its high half. One 32-bit load then yields two
    independent runs of four values, which is exactly the shape the Cortex-M7
    SSUB8/SXTB16/SMLAD sequence consumes.
    """
    b = _blocks(w)
    imax = np.abs(b).argmax(axis=1)
    signed_max = b[np.arange(b.shape[0]), imax]

    d16 = (signed_max / -8.0).astype(np.float16)
    d = d16.astype(np.float32)
    inv = np.where(d != 0.0, 1.0 / np.where(d != 0.0, d, 1.0), 0.0)

    q = _round_half_away(b * inv[:, None]) + 8.0
    q = np.clip(q, 0, 15).astype(np.uint8)

    lo = q[:, :16]
    hi = q[:, 16:]
    packed = (lo | (hi << 4)).astype(np.uint8)

    out = np.empty((b.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.tobytes()


def quantize(w: np.ndarray, dtype: int) -> bytes:
    if dtype == DT_Q8_0:
        return quantize_q8_0(w)
    if dtype == DT_Q4_0:
        return quantize_q4_0(w)
    raise ValueError(f"not a block dtype: {dtype}")


def dequantize(payload: bytes, dtype: int, n: int) -> np.ndarray:
    """Inverse of the above; the reference the C dequantizer is tested against."""
    nb = n // GROUP_SIZE
    if dtype == DT_Q8_0:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(nb, 34)
        d = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
        q = raw[:, 2:].view(np.int8).astype(np.float32)
        return (q * d[:, None]).reshape(-1)
    if dtype == DT_Q4_0:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(nb, 18)
        d = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
        p = raw[:, 2:]
        lo = (p & 0x0F).astype(np.float32) - 8.0
        hi = (p >> 4).astype(np.float32) - 8.0
        vals = np.concatenate([lo, hi], axis=1)
        return (vals * d[:, None]).reshape(-1)
    raise ValueError(f"not a block dtype: {dtype}")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def report(w: np.ndarray, dtype: int) -> str:
    """Round-trip error for a tensor, used by convert.py's --verbose mode."""
    w = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    back = dequantize(quantize(w, dtype), dtype, w.size)
    denom = float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) or 1.0
    return f"rmse={rmse(w, back):.6g} rel={rmse(w, back) / denom:.4%}"

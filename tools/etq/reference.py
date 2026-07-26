#!/usr/bin/env python3
# Copyright (c) 2026 Ian Adelman
# SPDX-License-Identifier: MIT
"""NumPy reference implementation of the .etq forward pass.

This is the oracle. It deliberately mirrors core/src/tq_forward.c operation for
operation — including quantizing activations to Q8_0 before every matmul and
accumulating int32 per 32-value block — so a disagreement means a real bug in
the C, not a modelling difference.

The one intentional divergence is accumulation precision: this sums in float64
where the device sums in float32. Comparisons therefore use a relative
tolerance rather than bit equality. Bit equality IS asserted, separately,
between the two C kernel backends (host/tests/test_kernels.c), which is where
the SMLAD path can actually go wrong.
"""

from __future__ import annotations

import numpy as np

from etq.format import DT_Q4_0, DT_Q8_0, GROUP_SIZE, Reader


def quantize_act(x: np.ndarray):
    """Q8_0-quantize an activation vector; returns (int8 blocks, fp32 scales)."""
    b = x.reshape(-1, GROUP_SIZE).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d16 = (amax / 127.0).astype(np.float16)
    d = d16.astype(np.float32)
    inv = np.where(d != 0.0, 1.0 / np.where(d != 0.0, d, 1.0), 0.0)
    v = b * inv[:, None]
    # Exactly what the C does: (int)(v < 0 ? v - 0.5f : v + 0.5f)
    q = np.trunc(np.where(v < 0.0, v - 0.5, v + 0.5))
    q = np.clip(q, -127, 127).astype(np.int32)
    return q, d.astype(np.float64)


def unpack_weight(payload: bytes, dtype: int, n: int, d: int):
    """Return (int codes [d, nb, 32], scales [d, nb]) for a quantized matrix."""
    nb = n // GROUP_SIZE
    if dtype == DT_Q8_0:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(d, nb, 34)
        s = np.ascontiguousarray(raw[:, :, :2]).view(np.float16).astype(np.float64)
        q = np.ascontiguousarray(raw[:, :, 2:]).view(np.int8).astype(np.int32)
        return q, s.reshape(d, nb)
    if dtype == DT_Q4_0:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(d, nb, 18)
        s = np.ascontiguousarray(raw[:, :, :2]).view(np.float16).astype(np.float64)
        p = raw[:, :, 2:]
        lo = (p & 0x0F).astype(np.int32) - 8
        hi = (p >> 4).astype(np.int32) - 8
        return np.concatenate([lo, hi], axis=2), s.reshape(d, nb)
    raise ValueError(f"not a block dtype: {dtype}")


class Model:
    def __init__(self, path: str):
        self.r = Reader(open(path, "rb").read())
        self.c = self.r.cfg
        self._cache = {}

    def _w(self, name: str, n: int, d: int):
        if name not in self._cache:
            t = self.r.tensors[name]
            self._cache[name] = unpack_weight(t.payload, t.dtype, n, d)
        return self._cache[name]

    def matvec(self, name: str, x: np.ndarray, n: int, d: int) -> np.ndarray:
        wq, ws = self._w(name, n, d)
        xq, xs = quantize_act(x)
        acc = np.einsum("dbk,bk->db", wq, xq, optimize=True).astype(np.float64)
        return (acc * ws * xs[None, :]).sum(axis=1)

    def f32(self, name: str) -> np.ndarray:
        return self.r.array(name).astype(np.float64)

    def embed(self, token: int) -> np.ndarray:
        c = self.c
        t = self.r.tensors["tok_emb"]
        row_n = c.dim
        nb = row_n // GROUP_SIZE
        stride = nb * (34 if t.dtype == DT_Q8_0 else 18)
        sl = t.payload[token * stride:(token + 1) * stride]
        q, s = unpack_weight(sl, t.dtype, row_n, 1)
        return (q[0].astype(np.float64) * s[0][:, None]).reshape(-1)


def rmsnorm(x, w, eps):
    return w * (x / np.sqrt(np.mean(x * x) + eps))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def generate_logits(path: str, tokens: list[int]) -> list[np.ndarray]:
    """Run the model over `tokens` and return the logits after each step."""
    m = Model(path)
    c = m.c
    hs, kv_dim = c.head_size, c.kv_dim
    kmul = c.n_heads // c.n_kv_heads

    kc = np.zeros((c.n_layers, len(tokens), kv_dim))
    vc = np.zeros((c.n_layers, len(tokens), kv_dim))
    out = []

    for pos, tok in enumerate(tokens):
        x = m.embed(tok)

        half = hs // 2
        freq = 1.0 / (c.rope_theta ** (np.arange(half) * 2.0 / hs))
        ang = pos * freq
        cos, sin = np.cos(ang), np.sin(ang)

        def rope(v, cos=cos, sin=sin):
            # cos/sin bound as defaults: they are per-position, and binding
            # them explicitly keeps the closure from depending on loop state.
            v = v.reshape(-1, hs).copy()
            a, b = v[:, 0::2].copy(), v[:, 1::2].copy()
            v[:, 0::2] = a * cos - b * sin
            v[:, 1::2] = a * sin + b * cos
            return v.reshape(-1)

        for l in range(c.n_layers):
            p = f"l{l:03d}."
            xb = rmsnorm(x, m.f32(p + "rms_att"), c.norm_eps)
            q = rope(m.matvec(p + "wq", xb, c.dim, c.dim))
            k = rope(m.matvec(p + "wk", xb, c.dim, kv_dim))
            v = m.matvec(p + "wv", xb, c.dim, kv_dim)
            kc[l, pos], vc[l, pos] = k, v

            att_out = np.zeros(c.dim)
            for h in range(c.n_heads):
                kvh = h // kmul
                qh = q[h * hs:(h + 1) * hs]
                ks = kc[l, :pos + 1, kvh * hs:(kvh + 1) * hs]
                sc = softmax(ks @ qh / np.sqrt(hs))
                vs = vc[l, :pos + 1, kvh * hs:(kvh + 1) * hs]
                att_out[h * hs:(h + 1) * hs] = sc @ vs

            x = x + m.matvec(p + "wo", att_out, c.dim, c.dim)

            xb = rmsnorm(x, m.f32(p + "rms_ffn"), c.norm_eps)
            h1 = m.matvec(p + "w1", xb, c.dim, c.hidden_dim)
            h3 = m.matvec(p + "w3", xb, c.dim, c.hidden_dim)
            hh = (h1 / (1.0 + np.exp(-h1))) * h3
            x = x + m.matvec(p + "w2", hh, c.hidden_dim, c.dim)

        xf = rmsnorm(x, m.f32("rms_final"), c.norm_eps)
        cls = "tok_emb" if c.shared_classifier else "cls"
        out.append(m.matvec(cls, xf, c.dim, c.vocab_size))

    return out

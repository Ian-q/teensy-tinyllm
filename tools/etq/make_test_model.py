#!/usr/bin/env python3
# Copyright (c) 2026 Ian Adelman
# SPDX-License-Identifier: MIT
"""Synthesize a deterministic tiny model for the test suite.

Real checkpoints are tens of megabytes and live behind a network fetch, which
makes them useless as a CI fixture. This builds a ~200 KB model with seeded
pseudo-random weights and a toy vocabulary that still exercises every code
path that matters:

  * grouped-query attention (n_kv_heads < n_heads)
  * a shared classifier
  * both Q4_0 and Q8_0
  * byte-fallback tokenization and multi-byte BPE merges

Being random, it generates gibberish — but gibberish that the NumPy reference
and the C engine must agree on to six digits.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from etq.convert import write_model
from etq.format import DT_Q4_0, DT_Q8_0, Config

DIM = 64
HIDDEN = 128
LAYERS = 2
HEADS = 4
KV_HEADS = 2          # exercises the GQA path
SEQ = 32


def build_vocab():
    """3 specials + 256 byte fallbacks + 61 merge pieces = 320 tokens."""
    pieces = [b"<unk>", b"\n<s>\n", b"\n</s>\n"]
    pieces += [f"<0x{b:02X}>".encode() for b in range(256)]

    merges = [b" "]
    merges += [bytes([ord("a") + i]) for i in range(26)]
    merges += [b" " + bytes([ord("a") + i]) for i in range(26)]
    merges += [b" the", b" and", b"ing", b" to", b" of", b"er", b"in", b"on"]
    assert len(merges) == 61, len(merges)
    pieces += merges

    assert len(pieces) == len(set(pieces)), "vocabulary must be unique"

    # Longer pieces score higher so BPE prefers them; the tiny id term keeps
    # ties deterministic.
    scores = np.asarray(
        [len(p) - 0.001 * i for i, p in enumerate(pieces)], dtype=np.float32)
    return pieces, scores


def build_weights(seed: int):
    rng = np.random.default_rng(seed)
    hs = DIM // HEADS
    kv_dim = hs * KV_HEADS
    vocab = 320

    def n(*shape):
        # Small magnitudes keep the residual stream from exploding across
        # layers, which would make the reference/device comparison dominated
        # by float range rather than by the quantizer.
        return rng.standard_normal(shape).astype(np.float32) * 0.05

    W = {"tok_emb": n(vocab, DIM), "rms_final": np.abs(n(DIM)) + 1.0}
    for l in range(LAYERS):
        W[f"l{l}.rms_att"] = np.abs(n(DIM)) + 1.0
        W[f"l{l}.rms_ffn"] = np.abs(n(DIM)) + 1.0
        W[f"l{l}.wq"] = n(DIM, DIM)
        W[f"l{l}.wk"] = n(kv_dim, DIM)
        W[f"l{l}.wv"] = n(kv_dim, DIM)
        W[f"l{l}.wo"] = n(DIM, DIM)
        W[f"l{l}.w1"] = n(HIDDEN, DIM)
        W[f"l{l}.w2"] = n(DIM, HIDDEN)
        W[f"l{l}.w3"] = n(HIDDEN, DIM)
    return W, vocab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--q8", action="store_true", help="Q8_0 instead of Q4_0")
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args(argv)

    W, vocab = build_weights(args.seed)
    pieces, scores = build_vocab()
    cfg = Config(DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, vocab, SEQ,
                 DT_Q8_0 if args.q8 else DT_Q4_0, shared_classifier=1)
    write_model(args.out, cfg, W, pieces, scores, verbose=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

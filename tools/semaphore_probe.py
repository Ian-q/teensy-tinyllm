#!/usr/bin/env python3
# Copyright (c) 2026 Ian Adelman
# SPDX-License-Identifier: MIT
"""Is the model a better compressor than classical codecs on short messages?

The "Semaphore" idea is to use the model as the probability source for an
arithmetic coder: two devices holding identical weights transmit only the
residual surprise of a message. That is worth building only if the model's
cross-entropy on realistic short messages beats what gzip/bzip2/lzma achieve
on the same strings — including a dictionary-primed deflate, which is the
strongest classical baseline for payloads too short to build a dictionary
from scratch.

This measures that, on the host, with no hardware in the loop. An arithmetic
coder reaches within ~0.01 bits/symbol of the model's cross-entropy, so
bits/char here is achievable in practice, not a bound.

    python3 tools/semaphore_probe.py --model stories15M.etq \\
        --tokenizer ~/Downloads/tokenizer.bin
"""

from __future__ import annotations

import argparse
import bz2
import lzma
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from etq import reference  # noqa: E402
from etq.convert import read_tokenizer_bin  # noqa: E402
from etq.format import Reader  # noqa: E402

# In-domain for a TinyStories model: this is the ceiling the model can reach
# on text it was actually trained for.
STORIES = [
    "Once upon a time there was a little girl named Lily who loved to play.",
    "Tom saw a big red ball in the park and he wanted to play with it.",
    "The cat was very happy because she found a warm place to sleep.",
    "Sara and her mom went to the store to buy some fresh apples.",
    "The little dog barked at the bird sitting in the tall tree.",
    "Ben was sad when his balloon flew away into the blue sky.",
    "Mia helped her dad wash the car on a sunny Saturday morning.",
    "The old man smiled and gave the children some candy from his pocket.",
]

# The actual target: terse operational messages of the kind that go over a
# very slow link. Out of domain for a TinyStories model, so these numbers are
# a pessimistic floor — a model fine-tuned on this register would do better.
MESSAGES = [
    "Meet me at the north trailhead at sunset, bring the spare battery.",
    "Camp 2 all fine, water resupply done, moving to camp 3 tomorrow AM.",
    "Need pickup at mile marker 42 on the county road, truck wont start.",
    "Weather turning, wind 25 gusting 40, we are staying put tonight.",
    "Arrived safe at the cabin, no injuries, radio check at 0800 daily.",
    "Running two hours behind schedule, do not send the search party.",
    "Found the trail junction, taking the east fork toward the lake.",
    "Low on fuel and food, request resupply at the usual drop point.",
    "Bridge is out at the creek crossing, detour adds about six miles.",
    "All clear here, packing up at first light, see you Sunday evening.",
]

# The corpus the dictionary baseline may train on. It must be the same
# REGISTER as the test messages without sharing their phrasing: a dictionary
# built from paraphrases of the test set is test-set leakage, and it flatters
# deflate enormously (it scored 2.19 bits/char that way, beating the model —
# an artifact, not a result). Held-out text of the same genre is the fair
# analogue of "both endpoints share a model trained on this domain".
DICT_CORPUS = (
    b"Base station copies your last transmission, signal strength good. "
    b"Vehicle three departed the staging area, ETA ninety minutes. "
    b"Snow line has dropped to about six thousand feet overnight. "
    b"Generator serviced, oil changed, hours logged in the maintenance book. "
    b"River level rising after the storm, ford is no longer passable. "
    b"Two hikers reported overdue from the ridge route, initiating callout. "
    b"Batteries at forty percent, switching to solar charge in the morning. "
    b"Antenna mast guyed and tested, SWR reads one point four. "
    b"Supplies cached at the col: rope, stove fuel, and three days rations. "
    b"Visibility under one hundred meters, holding position until it lifts. "
    b"Helicopter cannot fly in these conditions, ground team proceeding. "
    b"Checked the relay, replaced a blown fuse, back on the air now. "
)


def encode_bpe(text: str, pieces: list[bytes], scores: np.ndarray) -> list[int]:
    """llama2.c's greedy BPE: seed with single bytes, then merge the
    best-scoring adjacent pair until nothing improves."""
    lookup = {}
    for i, p in enumerate(pieces):
        lookup.setdefault(p, i)

    toks: list[int] = []
    for ch in text.encode():
        b = bytes([ch])
        if b in lookup:
            toks.append(lookup[b])
        else:
            toks.append(ch + 3)   # llama2.c byte-fallback: byte b is token b+3

    while True:
        best_score, best_at, best_id = -1e10, -1, -1
        for i in range(len(toks) - 1):
            merged = pieces[toks[i]] + pieces[toks[i + 1]]
            j = lookup.get(merged)
            if j is not None and scores[j] > best_score:
                best_score, best_at, best_id = scores[j], i, j
        if best_at < 0:
            break
        toks[best_at] = best_id
        del toks[best_at + 1]
    return toks


def model_bits(path: str, toks: list[int]) -> float:
    """Total bits an ideal arithmetic coder needs for `toks` under the model,
    counting every token after the BOS that seeds it."""
    logits = reference.generate_logits(path, toks)
    total = 0.0
    for i in range(len(toks) - 1):
        lg = logits[i].astype(np.float64)
        lg -= lg.max()
        p = np.exp(lg)
        p /= p.sum()
        total += -np.log2(max(p[toks[i + 1]], 1e-30))
    return total


def classical_bits(text: str) -> dict[str, float]:
    raw = text.encode()

    # Raw deflate, no gzip/zlib header — the honest framing for a payload
    # that rides inside a radio protocol that already has its own framing.
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    deflate = co.compress(raw) + co.flush()

    # Same, but primed with a domain dictionary both endpoints already hold.
    # This is the classical analogue of "both ends share the model".
    cd = zlib.compressobj(9, zlib.DEFLATED, -15, zdict=DICT_CORPUS[-32768:])
    deflate_dict = cd.compress(raw) + cd.flush()

    return {
        "raw": len(raw) * 8.0,
        "deflate": len(deflate) * 8.0,
        "deflate+dict": len(deflate_dict) * 8.0,
        "bzip2": len(bz2.compress(raw, 9)) * 8.0,
        "lzma": len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)) * 8.0,
    }


def run(name: str, corpus: list[str], path: str, pieces, scores, bos: int) -> None:
    print(f"\n=== {name} ({len(corpus)} messages) ===")
    tot_chars = 0
    tot = {k: 0.0 for k in ("raw", "deflate", "deflate+dict", "bzip2", "lzma")}
    tot_model = 0.0

    for text in corpus:
        toks = [bos] + encode_bpe(text, pieces, scores)
        mb = model_bits(path, toks)
        cb = classical_bits(text)
        tot_chars += len(text)
        tot_model += mb
        for k in tot:
            tot[k] += cb[k]

    print(f"  {'method':<16} {'bits/char':>10} {'bytes/msg':>11} {'vs raw':>8}")
    print(f"  {'-' * 47}")
    order = ["raw", "deflate", "deflate+dict", "bzip2", "lzma"]
    for k in order:
        bpc = tot[k] / tot_chars
        print(f"  {k:<16} {bpc:>10.3f} {tot[k] / 8 / len(corpus):>11.1f} "
              f"{tot['raw'] / tot[k]:>7.2f}x")
    bpc = tot_model / tot_chars
    print(f"  {'MODEL':<16} {bpc:>10.3f} {tot_model / 8 / len(corpus):>11.1f} "
          f"{tot['raw'] / tot_model:>7.2f}x")

    best_classical = min(tot[k] for k in order if k != "raw")
    winner = min((k for k in order if k != "raw"), key=lambda k: tot[k])
    ratio = best_classical / tot_model
    print(f"\n  best classical: {winner} at {best_classical / tot_chars:.3f} bits/char")
    print(f"  model is {ratio:.2f}x {'BETTER' if ratio > 1 else 'WORSE'} "
          f"than the best classical codec")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()

    r = Reader(open(args.model, "rb").read())
    vocab = r.cfg.vocab_size
    bos = r.cfg.bos_token if hasattr(r.cfg, "bos_token") else 1
    pieces, scores = read_tokenizer_bin(os.path.expanduser(args.tokenizer), vocab)
    print(f"model: {args.model}  vocab {vocab}  dim {r.cfg.dim}  "
          f"layers {r.cfg.n_layers}")

    run("IN-DOMAIN (TinyStories register)", STORIES, args.model, pieces, scores, bos)
    run("TARGET (terse operational messages)", MESSAGES, args.model, pieces,
        scores, bos)
    return 0


if __name__ == "__main__":
    sys.exit(main())

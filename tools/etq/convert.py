#!/usr/bin/env python3
# Copyright (c) 2026 Ian Adelman
# SPDX-License-Identifier: MIT
"""Convert a trained checkpoint into an .etq file the Teensy can run.

Two input paths:

  llama2c   Andrej Karpathy's llama2.c .bin (v0 legacy fp32, or v1/v2) plus a
            tokenizer.bin. This is the path for the stories* models, which are
            the only publicly available checkpoints small enough to be
            interesting on a 32 MB board.

  hf        A HuggingFace Llama-architecture directory (config.json +
            *.safetensors + tokenizer.model). Reads safetensors directly, so
            no torch and no safetensors package is required.

The one non-obvious transform is the RoPE permutation. HF stores wq/wk laid
out for the "split half" rotary convention; llama2.c and this engine use the
"interleaved pair" convention. Permuting once here means the device never has
to know there are two conventions.

Usage:
    python3 -m etq.convert llama2c  stories15M.bin tokenizer.bin out.etq --q4
    python3 -m etq.convert hf       ./TinyLlama-1.1B out.etq --q8
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

from etq import quantize as Q
from etq.format import DT_F32, DT_Q4_0, DT_Q8_0, DT_U8, DT_U32, Config, Writer

# ------------------------------------------------------------------ helpers


def permute_reverse(w: np.ndarray, n_heads: int, dim1: int, dim2: int) -> np.ndarray:
    """HF split-half RoPE layout -> llama2.c interleaved-pair layout."""
    return (w.reshape(n_heads, 2, dim1 // n_heads // 2, dim2)
             .transpose(0, 2, 1, 3)
             .reshape(dim1, dim2))


def _build_sorted(pieces: list[bytes]) -> np.ndarray:
    """Lexicographic permutation of token ids, so the device can binary-search.

    Sorting is by raw bytes with shorter-is-smaller on a tie, matching tq_cmp()
    in core/src/tq_tokenizer.c exactly.
    """
    order = sorted(range(len(pieces)), key=lambda i: pieces[i])
    return np.asarray(order, dtype=np.uint32)


def add_tokenizer(w: Writer, pieces: list[bytes], scores: np.ndarray) -> None:
    blob = b"".join(pieces)
    offs = np.zeros(len(pieces) + 1, dtype=np.uint32)
    np.cumsum([len(p) for p in pieces], out=offs[1:], dtype=np.uint32)

    w.add("tok.scores", DT_F32, (len(pieces),),
          scores.astype(np.float32).tobytes())
    w.add("tok.offs", DT_U32, (len(pieces) + 1,), offs.tobytes())
    w.add("tok.blob", DT_U8, (len(blob),), blob)
    w.add("tok.sorted", DT_U32, (len(pieces),), _build_sorted(pieces).tobytes())


def emit(w: Writer, name: str, arr: np.ndarray, qtype: int, verbose: bool) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    w.add(name, qtype, arr.shape, Q.quantize(arr, qtype))
    if verbose:
        print(f"  {name:<16} {str(list(arr.shape)):<16} {Q.report(arr, qtype)}",
              file=sys.stderr)


def emit_f32(w: Writer, name: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    w.add(name, DT_F32, arr.shape, arr.tobytes())


def write_model(path: str, cfg: Config, weights: dict, pieces, scores,
                verbose: bool) -> None:
    """weights: dict of name -> ndarray, already in device layout."""
    w = Writer(cfg)
    qt = cfg.qtype

    emit(w, "tok_emb", weights["tok_emb"], qt, verbose)
    emit_f32(w, "rms_final", weights["rms_final"])
    if not cfg.shared_classifier:
        emit(w, "cls", weights["cls"], qt, verbose)

    for l in range(cfg.n_layers):
        emit_f32(w, f"l{l:03d}.rms_att", weights[f"l{l}.rms_att"])
        emit_f32(w, f"l{l:03d}.rms_ffn", weights[f"l{l}.rms_ffn"])
        for t in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            emit(w, f"l{l:03d}.{t}", weights[f"l{l}.{t}"], qt, verbose)

    if pieces is not None:
        add_tokenizer(w, pieces, scores)

    with open(path, "wb") as fp:
        total = w.write(fp)
    print(f"wrote {path}: {total / 1e6:.2f} MB, {len(w.tensors)} tensors",
          file=sys.stderr)


# ------------------------------------------------------------ llama2.c input


def read_tokenizer_bin(path: str, vocab_size: int):
    """llama2.c tokenizer.bin: max_len, then (score, len, bytes) per token."""
    with open(path, "rb") as f:
        buf = f.read()
    off = 4  # skip max_token_length
    pieces, scores = [], np.zeros(vocab_size, dtype=np.float32)
    for i in range(vocab_size):
        (score, ln) = struct.unpack_from("<fi", buf, off)
        off += 8
        pieces.append(buf[off:off + ln])
        off += ln
        scores[i] = score
    return pieces, scores


def load_llama2c(bin_path: str):
    with open(bin_path, "rb") as f:
        raw = f.read()

    magic, = struct.unpack_from("<I", raw, 0)
    if magic == 0x616B3432:  # "ak42" -> already-quantized v2, not supported
        raise SystemExit(
            "this is a llama2.c v2 (int8) export; convert from the fp32 "
            "checkpoint instead — requantizing an already-quantized model "
            "compounds the error for no benefit")

    dim, hidden, nl, nh, nkv, vocab, seq = struct.unpack_from("<7i", raw, 0)
    shared = 1 if vocab > 0 else 0
    vocab = abs(vocab)
    head_size = dim // nh
    off = 28

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        a = np.frombuffer(raw, dtype=np.float32, count=n, offset=off).reshape(shape)
        off += n * 4
        return a

    W = {}
    W["tok_emb"] = take(vocab, dim)
    rms_att = take(nl, dim)
    wq = take(nl, nh * head_size, dim)
    wk = take(nl, nkv * head_size, dim)
    wv = take(nl, nkv * head_size, dim)
    wo = take(nl, dim, nh * head_size)
    rms_ffn = take(nl, dim)
    w1 = take(nl, hidden, dim)
    w2 = take(nl, dim, hidden)
    w3 = take(nl, hidden, dim)
    W["rms_final"] = take(dim)
    # v0 files carry precomputed RoPE tables here; we recompute on device.
    take(seq, head_size // 2)
    take(seq, head_size // 2)
    if not shared:
        W["cls"] = take(vocab, dim)

    for l in range(nl):
        W[f"l{l}.rms_att"] = rms_att[l]
        W[f"l{l}.rms_ffn"] = rms_ffn[l]
        W[f"l{l}.wq"] = wq[l]
        W[f"l{l}.wk"] = wk[l]
        W[f"l{l}.wv"] = wv[l]
        W[f"l{l}.wo"] = wo[l]
        W[f"l{l}.w1"] = w1[l]
        W[f"l{l}.w2"] = w2[l]
        W[f"l{l}.w3"] = w3[l]

    cfg = Config(dim, hidden, nl, nh, nkv, vocab, seq, DT_Q4_0, shared)
    return cfg, W


# --------------------------------------------------------- safetensors input


def read_safetensors(path: str) -> dict:
    """Minimal safetensors reader. Handles F32, F16 and BF16."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(n))
        body = f.read()
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        raw = body[a:b]
        if v["dtype"] == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif v["dtype"] == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif v["dtype"] == "BF16":
            # bf16 is the top 16 bits of an fp32; widen by shifting back.
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        else:
            raise SystemExit(f"{k}: unsupported dtype {v['dtype']}")
        out[k] = arr.reshape(v["shape"])
    return out


def load_hf(d: str):
    cfgj = json.load(open(os.path.join(d, "config.json")))
    tensors = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".safetensors"):
            tensors.update(read_safetensors(os.path.join(d, fn)))

    dim = cfgj["hidden_size"]
    hidden = cfgj["intermediate_size"]
    nl = cfgj["num_hidden_layers"]
    nh = cfgj["num_attention_heads"]
    nkv = cfgj.get("num_key_value_heads", nh)
    vocab = cfgj["vocab_size"]
    seq = cfgj.get("max_position_embeddings", 2048)
    head_size = dim // nh

    def g(name):
        for pre in ("", "model."):
            if pre + name in tensors:
                return tensors[pre + name]
        raise SystemExit(f"missing tensor: {name}")

    W = {"tok_emb": g("embed_tokens.weight"), "rms_final": g("norm.weight")}
    shared = "lm_head.weight" not in tensors
    if not shared:
        W["cls"] = tensors["lm_head.weight"]

    for l in range(nl):
        p = f"layers.{l}."
        W[f"l{l}.rms_att"] = g(p + "input_layernorm.weight")
        W[f"l{l}.rms_ffn"] = g(p + "post_attention_layernorm.weight")
        W[f"l{l}.wq"] = permute_reverse(g(p + "self_attn.q_proj.weight"),
                                        nh, nh * head_size, dim)
        W[f"l{l}.wk"] = permute_reverse(g(p + "self_attn.k_proj.weight"),
                                        nkv, nkv * head_size, dim)
        W[f"l{l}.wv"] = g(p + "self_attn.v_proj.weight")
        W[f"l{l}.wo"] = g(p + "self_attn.o_proj.weight")
        W[f"l{l}.w1"] = g(p + "mlp.gate_proj.weight")
        W[f"l{l}.w2"] = g(p + "mlp.down_proj.weight")
        W[f"l{l}.w3"] = g(p + "mlp.up_proj.weight")

    cfg = Config(dim, hidden, nl, nh, nkv, vocab, seq, DT_Q4_0, int(shared),
                 rope_theta=float(cfgj.get("rope_theta", 10000.0)),
                 norm_eps=float(cfgj.get("rms_norm_eps", 1e-5)),
                 bos_token=int(cfgj.get("bos_token_id", 1)),
                 eos_token=int(cfgj.get("eos_token_id", 2)))
    return cfg, W


def load_sentencepiece(path: str, vocab_size: int):
    """Read tokenizer.model without the sentencepiece package.

    tokenizer.model is a protobuf; we only need the repeated `pieces` field
    (field 1, each with piece=field 1 string and score=field 2 float). Parsing
    those two fields by hand avoids a heavyweight dependency for what amounts
    to a flat list.
    """
    buf = open(path, "rb").read()
    pieces, scores = [], []
    i = 0

    def varint(b, i):
        v, s = 0, 0
        while True:
            c = b[i]
            i += 1
            v |= (c & 0x7F) << s
            if not (c & 0x80):
                return v, i
            s += 7

    while i < len(buf):
        key, i = varint(buf, i)
        fld, wt = key >> 3, key & 7
        if fld == 1 and wt == 2:
            ln, i = varint(buf, i)
            sub, j, piece, score = buf[i:i + ln], 0, None, 0.0
            i += ln
            while j < len(sub):
                k2, j = varint(sub, j)
                f2, w2 = k2 >> 3, k2 & 7
                if f2 == 1 and w2 == 2:
                    l2, j = varint(sub, j)
                    piece = sub[j:j + l2]
                    j += l2
                elif f2 == 2 and w2 == 5:
                    score = struct.unpack_from("<f", sub, j)[0]
                    j += 4
                elif w2 == 0:
                    _, j = varint(sub, j)
                elif w2 == 2:
                    l2, j = varint(sub, j)
                    j += l2
                elif w2 == 5:
                    j += 4
                else:
                    break
            if piece is not None:
                pieces.append(piece.replace("▁".encode(), b" "))
                scores.append(score)
        elif wt == 2:
            ln, i = varint(buf, i)
            i += ln
        elif wt == 0:
            _, i = varint(buf, i)
        elif wt == 5:
            i += 4
        else:
            break

    if len(pieces) < vocab_size:
        # Pad out unused ids (some configs declare a larger vocab than the
        # tokenizer actually defines).
        pieces += [f"<unused{k}>".encode() for k in range(len(pieces), vocab_size)]
        scores += [0.0] * (vocab_size - len(scores))
    return pieces[:vocab_size], np.asarray(scores[:vocab_size], dtype=np.float32)


# --------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("llama2c")
    a.add_argument("checkpoint")
    a.add_argument("tokenizer")
    a.add_argument("out")

    b = sub.add_parser("hf")
    b.add_argument("dir")
    b.add_argument("out")

    for p in (a, b):
        g = p.add_mutually_exclusive_group()
        g.add_argument("--q4", action="store_const", const=DT_Q4_0, dest="qtype")
        g.add_argument("--q8", action="store_const", const=DT_Q8_0, dest="qtype")
        p.add_argument("--seq-len", type=int, default=0,
                       help="clamp the declared context length")
        p.add_argument("-v", "--verbose", action="store_true",
                       help="print per-tensor quantization error")

    args = ap.parse_args(argv)

    if args.mode == "llama2c":
        cfg, W = load_llama2c(args.checkpoint)
        pieces, scores = read_tokenizer_bin(args.tokenizer, cfg.vocab_size)
    else:
        cfg, W = load_hf(args.dir)
        tk = os.path.join(args.dir, "tokenizer.model")
        if os.path.exists(tk):
            pieces, scores = load_sentencepiece(tk, cfg.vocab_size)
        else:
            print("warning: no tokenizer.model; model will have no tokenizer",
                  file=sys.stderr)
            pieces, scores = None, None

    cfg.qtype = args.qtype or DT_Q4_0
    if args.seq_len:
        cfg.seq_len = args.seq_len

    write_model(args.out, cfg, W, pieces, scores, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())

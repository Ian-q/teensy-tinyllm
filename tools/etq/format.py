#!/usr/bin/env python3
# Copyright (c) 2026 Ian Adelman
# SPDX-License-Identifier: MIT
"""Reader/writer for the .etq container.

The byte layout here is the mirror image of core/include/tq/tq_format.h. The C
side pins its structs with static asserts and host/tests/test_format.c reparses
a file this module wrote, so the two cannot drift silently.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO

import numpy as np

MAGIC = 0x31515445  # "ETQ1"
VERSION = 1
HEADER_BYTES = 256
ALIGN = 64
NAME_BYTES = 24
GROUP_SIZE = 32

DT_F32, DT_Q8_0, DT_Q4_0, DT_U8, DT_U32 = 0, 1, 2, 3, 4
ARCH_LLAMA2 = 0

DTYPE_NAMES = {DT_F32: "f32", DT_Q8_0: "q8_0", DT_Q4_0: "q4_0",
               DT_U8: "u8", DT_U32: "u32"}

# 8 x u32, u64, 7 x i32, 3 x u32, 2 x f32, 2 x u32  -> 96 bytes, then padding.
_HDR = "<8I Q 7i 3I 2f 2I"
_ENT = f"<{NAME_BYTES}s 2I 4I 2Q"

assert struct.calcsize(_HDR) == 96, struct.calcsize(_HDR)
assert struct.calcsize(_ENT) == 64, struct.calcsize(_ENT)


def dtype_nbytes(dtype: int, n: int) -> int:
    if dtype in (DT_F32, DT_U32):
        return n * 4
    if dtype == DT_U8:
        return n
    if dtype == DT_Q8_0:
        assert n % GROUP_SIZE == 0, f"{n} is not a whole number of blocks"
        return (n // GROUP_SIZE) * 34
    if dtype == DT_Q4_0:
        assert n % GROUP_SIZE == 0, f"{n} is not a whole number of blocks"
        return (n // GROUP_SIZE) * 18
    raise ValueError(f"bad dtype {dtype}")


@dataclass
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    qtype: int
    shared_classifier: int = 1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    bos_token: int = 1
    eos_token: int = 2

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.head_size * self.n_kv_heads


@dataclass
class Tensor:
    name: str
    dtype: int
    shape: tuple
    payload: bytes

    def __post_init__(self):
        if len(self.name.encode()) >= NAME_BYTES:
            raise ValueError(f"tensor name too long: {self.name!r}")
        if len(self.shape) > 4:
            raise ValueError(f"{self.name}: rank {len(self.shape)} > 4")


@dataclass
class Writer:
    cfg: Config
    tensors: list = field(default_factory=list)

    def add(self, name: str, dtype: int, shape: tuple, payload: bytes) -> None:
        n = 1
        for s in shape:
            n *= s
        want = dtype_nbytes(dtype, n)
        if len(payload) != want:
            raise ValueError(
                f"{name}: payload {len(payload)} B, expected {want} B "
                f"for {DTYPE_NAMES[dtype]}{list(shape)}")
        self.tensors.append(Tensor(name, dtype, tuple(shape), payload))

    def write(self, fp: BinaryIO) -> int:
        table_off = HEADER_BYTES
        table_bytes = 64 * len(self.tensors)
        data_off = _align(table_off + table_bytes)

        entries, off = [], data_off
        for t in self.tensors:
            shape = list(t.shape) + [0] * (4 - len(t.shape))
            entries.append(struct.pack(
                _ENT, t.name.encode(), t.dtype, len(t.shape),
                shape[0], shape[1], shape[2], shape[3], off, len(t.payload)))
            off = _align(off + len(t.payload))
        total = off

        c = self.cfg
        fp.write(struct.pack(
            _HDR,
            MAGIC, VERSION, HEADER_BYTES, ARCH_LLAMA2,
            len(self.tensors), table_off, data_off, 0,
            total,
            c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads,
            c.vocab_size, c.seq_len,
            c.qtype, GROUP_SIZE, c.shared_classifier,
            c.rope_theta, c.norm_eps,
            c.bos_token, c.eos_token))
        fp.write(b"\0" * (HEADER_BYTES - 96))
        for e in entries:
            fp.write(e)
        _pad_to(fp, data_off)
        for t, e in zip(self.tensors, entries):
            want = struct.unpack(_ENT, e)[7]   # payload offset
            _pad_to(fp, want)
            fp.write(t.payload)
        _pad_to(fp, total)
        return total


def _align(v: int) -> int:
    return (v + ALIGN - 1) & ~(ALIGN - 1)


def _pad_to(fp: BinaryIO, target: int) -> None:
    cur = fp.tell()
    if cur > target:
        raise AssertionError(f"overshot: at {cur}, wanted {target}")
    if cur < target:
        fp.write(b"\0" * (target - cur))


@dataclass
class Reader:
    """Minimal reader, used by etq-info and by the round-trip tests."""

    buf: bytes

    def __post_init__(self):
        f = struct.unpack_from(_HDR, self.buf, 0)
        if f[0] != MAGIC:
            raise ValueError("not an .etq file")
        if f[1] != VERSION:
            raise ValueError(f"unsupported version {f[1]}")
        (_, _, _, self.arch, self.tensor_count, self.table_off,
         self.data_off, self.flags, self.file_bytes,
         dim, hidden, nl, nh, nkv, vocab, seq,
         qtype, gs, shared, theta, eps, bos, eos) = f
        self.cfg = Config(dim, hidden, nl, nh, nkv, vocab, seq, qtype,
                          shared, theta, eps, bos, eos)
        self.group_size = gs
        self.tensors = {}
        self.order = []
        for i in range(self.tensor_count):
            e = struct.unpack_from(_ENT, self.buf, self.table_off + 64 * i)
            name = e[0].split(b"\0")[0].decode()
            off, nbytes = e[7], e[8]
            t = Tensor(name, e[1], tuple(e[3:3 + e[2]]),
                       self.buf[off:off + nbytes])
            self.tensors[name] = t
            self.order.append(name)

    def array(self, name: str) -> np.ndarray:
        """Dequantize a tensor back to float32 (or its raw numeric form)."""
        t = self.tensors[name]
        n = int(np.prod(t.shape)) if t.shape else 0
        if t.dtype == DT_F32:
            return np.frombuffer(t.payload, dtype=np.float32).reshape(t.shape)
        if t.dtype == DT_U32:
            return np.frombuffer(t.payload, dtype=np.uint32).reshape(t.shape)
        if t.dtype == DT_U8:
            return np.frombuffer(t.payload, dtype=np.uint8)
        from etq import quantize
        return quantize.dequantize(t.payload, t.dtype, n).reshape(t.shape)

/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * .etq container format — the on-disk / in-PSRAM layout of a quantized model.
 *
 * Design constraints that shaped this format (see docs/MODEL-FORMAT.md):
 *
 *  1. Token generation on a Teensy 4.1 is *memory-bandwidth bound*, not compute
 *     bound, by roughly an order of magnitude. Every byte of a weight matrix is
 *     read exactly once per token and never reused. So the format is optimised
 *     for one thing: a single, perfectly sequential read per matrix.
 *
 *  2. Scales are INTERLEAVED with their quantized values (ggml block layout),
 *     not stored in a parallel array (llama2.c runq layout). A split layout
 *     forces two concurrent read streams through one FlexSPI2 AHB RX buffer,
 *     which thrashes the prefetcher. Interleaved keeps it to one stream.
 *
 *  3. Every tensor payload is 64-byte aligned so eDMA bursts and Cortex-M7
 *     cache lines (32 B) never straddle a tensor boundary.
 *
 *  4. The file is directly usable in place. Load it into PSRAM at 0x70000000
 *     (or XIP it from QSPI NOR) and the tensor payloads are already in their
 *     final form — no unpacking pass, no second copy. On a 32 MB part there is
 *     no room for a second copy.
 */

#ifndef TQ_FORMAT_H_
#define TQ_FORMAT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* "ETQ1" as a little-endian uint32. */
#define TQ_MAGIC          0x31515445u
#define TQ_VERSION        1u
#define TQ_HEADER_BYTES   256u
#define TQ_ALIGN          64u
#define TQ_NAME_BYTES     24u

/* Quantization / storage types. */
enum {
	TQ_DT_F32   = 0, /* plain float32                                    */
	TQ_DT_Q8_0  = 1, /* 32-value blocks, fp16 scale + 32 int8            */
	TQ_DT_Q4_0  = 2, /* 32-value blocks, fp16 scale + 16 packed nibbles  */
	TQ_DT_U8    = 3, /* raw bytes (tokenizer string blob)                */
	TQ_DT_U32   = 4, /* raw uint32 (tokenizer offsets)                   */
};

/* Model architecture family. Only llama2-style is implemented today; the
 * field exists so a future GPT-2/Qwen decoder can share the container. */
enum {
	TQ_ARCH_LLAMA2 = 0,
};

/* Every quantized block covers exactly this many weights. Fixed rather than
 * configurable: the Cortex-M7 kernels unroll against it, and 32 is the point
 * where Q4 scale overhead (2 B per 16 B of payload) stops mattering. */
#define TQ_GROUP_SIZE 32

/* Q8_0: fp16 scale followed by 32 int8 values. 34 bytes, no padding. */
typedef struct {
	uint16_t d;
	int8_t   qs[TQ_GROUP_SIZE];
} TqBlockQ80;

/*
 * Q4_0: fp16 scale followed by 32 nibbles packed 2-per-byte, 18 bytes.
 *
 * Nibble order is SPLIT, not sequential: the low nibble of qs[j] holds
 * element j, the high nibble holds element j+16. Sequential packing would be
 * easier to read but it costs an extra shuffle on every unpack. With split
 * packing, one 32-bit load yields two independent groups of four int8 values
 * via a mask and a shift, which is exactly what __SSUB8 / __SXTB16 / __SMLAD
 * want on Cortex-M7. Value = nibble - 8, giving the range [-8, +7].
 */
typedef struct {
	uint16_t d;
	uint8_t  qs[TQ_GROUP_SIZE / 2];
} TqBlockQ40;

/* File header. Exactly TQ_HEADER_BYTES, zero-padded. */
typedef struct {
	uint32_t magic;               /* TQ_MAGIC                             */
	uint32_t version;             /* TQ_VERSION                           */
	uint32_t header_bytes;        /* TQ_HEADER_BYTES                      */
	uint32_t arch;                /* TQ_ARCH_*                            */

	uint32_t tensor_count;
	uint32_t tensor_table_offset; /* byte offset of the TqTensorEntry[]   */
	uint32_t data_offset;         /* byte offset of the first payload     */
	uint32_t flags;               /* reserved, 0                          */

	uint64_t file_bytes;          /* total size, for integrity checking   */

	/* --- llama2 architecture parameters --- */
	int32_t  dim;
	int32_t  hidden_dim;
	int32_t  n_layers;
	int32_t  n_heads;
	int32_t  n_kv_heads;          /* < n_heads means grouped-query attn   */
	int32_t  vocab_size;
	int32_t  seq_len;             /* max context the weights were trained for */

	uint32_t qtype;               /* TQ_DT_Q4_0 or TQ_DT_Q8_0             */
	uint32_t group_size;          /* must equal TQ_GROUP_SIZE             */
	uint32_t shared_classifier;   /* 1 => output head reuses tok_emb      */

	float    rope_theta;          /* usually 10000.0                      */
	float    norm_eps;            /* usually 1e-5                         */

	uint32_t bos_token;
	uint32_t eos_token;

	/* Fields above occupy 96 bytes; pad the struct out to TQ_HEADER_BYTES. */
	uint8_t  reserved[TQ_HEADER_BYTES - 96];
} TqHeader;

/* Tensor directory entry. Exactly 64 bytes. */
typedef struct {
	char     name[TQ_NAME_BYTES];
	uint32_t dtype;               /* TQ_DT_*                              */
	uint32_t ndim;
	uint32_t shape[4];
	uint64_t offset;              /* from file start, TQ_ALIGN-aligned    */
	uint64_t nbytes;
} TqTensorEntry;

/* Bytes needed to store `n` values of `dtype`. Returns 0 for a bad dtype or
 * for a quantized length that is not a whole number of blocks. */
uint64_t tq_dtype_nbytes(uint32_t dtype, uint64_t n);

#ifdef __cplusplus
}
#endif

#endif /* TQ_FORMAT_H_ */

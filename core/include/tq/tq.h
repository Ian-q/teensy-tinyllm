/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * libtq — a portable C99 quantized-transformer inference core.
 *
 * The whole engine is freestanding: no malloc, no stdio, no float printf, no
 * OS calls. The caller supplies every byte of working memory, which is what
 * lets the firmware place runtime state in DTCM (fast, 512 KB) and the KV
 * cache in PSRAM (slow, up to 32 MB) without the core knowing or caring.
 *
 * The same source builds and runs on the host, where it is checked against a
 * NumPy reference (tools/gen_golden.py -> host/tests/test_golden.c). Anything
 * the Teensy computes wrong, the host test should have caught first.
 */

#ifndef TQ_H_
#define TQ_H_

#include <stddef.h>
#include <stdint.h>

#include "tq/tq_format.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ errors */

typedef enum {
	TQ_OK = 0,
	TQ_ERR_MAGIC       = -1,  /* not an .etq file                          */
	TQ_ERR_VERSION     = -2,  /* container version we do not speak         */
	TQ_ERR_TRUNCATED   = -3,  /* header/table/payload runs past end of file */
	TQ_ERR_UNSUPPORTED = -4,  /* arch or dtype this build cannot run       */
	TQ_ERR_MISSING     = -5,  /* a required tensor is not in the directory */
	TQ_ERR_SHAPE       = -6,  /* a tensor has the wrong shape              */
	TQ_ERR_NOMEM       = -7,  /* caller arena too small                    */
	TQ_ERR_IO          = -8,  /* backing store read failed                 */
	TQ_ERR_RANGE       = -9,  /* argument out of range                     */
} TqStatus;

const char *tq_strerror(int status);

/* ------------------------------------------------------------- weight store
 *
 * Three storage tiers are supported behind one interface:
 *
 *   PSRAM  / XIP NOR : memory-mapped. `base` is set, reads are pointer maths.
 *   microSD          : streamed. `read` is set and the engine tiles matrices
 *                      through `tile`, trading throughput for capacity.
 *   host file        : mmap'd, so it looks exactly like the PSRAM case.
 *
 * Streaming mode exists so the board can run a model larger than its PSRAM
 * (e.g. a 62 MB Q4 stories110M on a 32 MB board). It is roughly 2x slower —
 * SDIO tops out near 22 MB/s against PSRAM's 34-55 MB/s — but it is the
 * difference between "will not load" and "slow".
 */

typedef struct TqStore {
	const uint8_t *base;       /* non-NULL => memory-mapped, `read` unused  */
	uint64_t       bytes;      /* total size of the backing object          */

	/* Streaming backend. Must return 0 on success. */
	int  (*read)(void *ctx, uint64_t offset, void *dst, uint32_t nbytes);
	void  *ctx;

	uint8_t *tile;             /* bounce buffer, >= one matrix row          */
	uint32_t tile_bytes;
} TqStore;

/* Point a store at a contiguous memory image (PSRAM, XIP flash, host mmap). */
void tq_store_init_mapped(TqStore *st, const void *base, uint64_t bytes);

/* Point a store at a streaming backend plus a bounce buffer. */
void tq_store_init_stream(TqStore *st, uint64_t bytes,
			  int (*read)(void *, uint64_t, void *, uint32_t),
			  void *ctx, uint8_t *tile, uint32_t tile_bytes);

/* --------------------------------------------------------------- the model */

typedef struct {
	uint64_t rms_att;   /* f32[dim]                */
	uint64_t rms_ffn;   /* f32[dim]                */
	uint64_t wq, wk, wv, wo;
	uint64_t w1, w2, w3;
} TqLayerOffsets;

typedef struct {
	TqHeader        hdr;
	const TqStore  *store;

	int dim, hidden_dim, n_layers, n_heads, n_kv_heads;
	int vocab_size, seq_len, head_size, kv_dim, kv_mul;
	uint32_t qtype;

	uint64_t tok_emb;      /* quantized [vocab, dim]   */
	uint64_t rms_final;    /* f32[dim]                 */
	uint64_t classifier;   /* quantized [vocab, dim]   */
	TqLayerOffsets *layers;  /* n_layers entries, from the caller arena */

	/* Tokenizer blobs (optional; zero if the model carries no tokenizer). */
	uint64_t tok_scores, tok_offs, tok_blob, tok_sorted;
	uint32_t tok_blob_bytes;
} TqModel;

/*
 * Parse the header and tensor directory.
 *
 * `arena`/`arena_bytes` supply the per-layer offset table; it needs
 * n_layers * sizeof(TqLayerOffsets) bytes. Nothing else is allocated.
 *
 * For a streaming store this reads only the header and directory, so a 62 MB
 * model on an SD card opens in milliseconds.
 */
int tq_model_open(TqModel *m, const TqStore *store, void *arena, size_t arena_bytes);

/* Bytes of scratch that tq_runtime_init needs, given a model and context. */
size_t tq_runtime_bytes(const TqModel *m, int max_seq);

/* Bytes of KV cache for `max_seq` tokens at the given cache dtype. */
size_t tq_kv_bytes(const TqModel *m, int max_seq, int kv_dtype);

/* ------------------------------------------------------------- KV cache dtype
 *
 * fp32 is exact and simple; int8 costs ~4x less memory for a barely
 * measurable quality change, which is what makes a 1024-token context fit
 * alongside 24 MB of weights on a 32 MB board.
 */
enum {
	TQ_KV_F32 = 0,
	TQ_KV_Q8  = 1,
};

typedef struct {
	const TqModel *m;
	int max_seq;
	int kv_dtype;

	/* activations, all f32, all in the fast arena */
	float *x, *xb, *xb2, *hb, *hb3, *q, *att, *logits;
	float *kb, *vb;   /* kv_dim staging for the current position */

	/*
	 * RoPE cos/sin for the current position, head_size/2 pairs.
	 *
	 * Position is constant across one forward() call and the rotation
	 * angles depend only on (pos, index-within-head), not on the layer or
	 * the head. So these are computed once per token and reused
	 * n_layers * n_heads times — which turns a few thousand sinf/cosf
	 * calls per token into thirty-two.
	 */
	float *rope_cos, *rope_sin;

	/* quantized activation staging */
	TqBlockQ80 *xq;   /* dim/32 blocks    */
	TqBlockQ80 *hq;   /* hidden/32 blocks */

	/* KV cache, in the (possibly slow) cache arena */
	void  *k_cache, *v_cache;
	float *k_scale, *v_scale;   /* only used when kv_dtype == TQ_KV_Q8 */

	uint64_t bytes_read;  /* weight bytes touched, for the bandwidth counter */
} TqRuntime;

/*
 * Wire up a runtime over two caller-owned arenas.
 *
 *   fast / fast_bytes  — activations. ~150 KB for a 42M model with a 32k
 *                        vocab (the logits vector alone is 128 KB). Put this
 *                        in DTCM; it is touched thousands of times per token.
 *   cache / cache_bytes — KV cache. Grows with context; put it in PSRAM.
 */
int tq_runtime_init(TqRuntime *rt, const TqModel *m, int max_seq, int kv_dtype,
		    void *fast, size_t fast_bytes,
		    void *cache, size_t cache_bytes);

/*
 * Run one decode step. Writes vocab_size logits into rt->logits.
 * `pos` must be in [0, max_seq).
 */
int tq_forward(TqRuntime *rt, int token, int pos);

/* ------------------------------------------------------------------ sampler */

typedef struct {
	uint64_t rng;       /* xorshift64* state; any non-zero seed */
	float    temperature;
	float    topp;      /* nucleus threshold; <= 0 or >= 1 disables it */
	int      vocab_size;
	/* Scratch for top-p: vocab_size entries, caller-owned. */
	int32_t *idx;
	float   *prob;
} TqSampler;

void tq_sampler_init(TqSampler *s, int vocab_size, float temperature, float topp,
		     uint64_t seed, int32_t *idx_scratch, float *prob_scratch);

/*
 * Draw a token. NOTE: this CONSUMES `logits` — the softmax is done in place to
 * avoid a second vocab_size buffer (128 KB at a 32k vocab, which is a quarter
 * of the Teensy's DTCM). Copy them first if you still need the raw values.
 */
int  tq_sample(TqSampler *s, float *logits);

/* ---------------------------------------------------------------- tokenizer */

typedef struct {
	const TqModel *m;
	const float    *scores;   /* [vocab]                                  */
	const uint32_t *offs;     /* [vocab+1] byte offsets into blob          */
	const uint32_t *sorted;   /* [vocab] token ids in lexicographic order  */
	const char     *blob;
	int             vocab_size;
	/* Scratch used by encode: two token-string buffers. */
	char *buf;
	size_t buf_bytes;
} TqTokenizer;

/*
 * Bind a tokenizer to a model. Requires a memory-mapped store — the tokenizer
 * does thousands of tiny random reads and streaming them off SD would be
 * pathological. `buf` needs 2*max_token_len+2 bytes; 64 is plenty for Llama.
 */
int tq_tokenizer_init(TqTokenizer *t, const TqModel *m, char *buf, size_t buf_bytes);

/*
 * BPE-encode `text` into `tokens`. Writes at most `max_tokens`; returns the
 * count, or a negative TqStatus. Set `add_bos`/`add_eos` to bracket the
 * sequence. A leading space is prepended like SentencePiece does, unless the
 * text is empty.
 */
int tq_encode(TqTokenizer *t, const char *text, int add_bos, int add_eos,
	      int32_t *tokens, int max_tokens);

/*
 * Decode one token to a NUL-terminated string in `out`. Handles the
 * SentencePiece "▁" space marker and <0xXX> byte fallbacks. `prev` is the
 * preceding token (or -1); Llama strips the leading space after BOS.
 */
int tq_decode(TqTokenizer *t, int prev, int token, char *out, size_t out_bytes);

/* ------------------------------------------------------- kernels (exposed for
 * benchmarking and for the host test suite; not needed by normal callers) */

/* Quantize `n` floats into n/32 Q8_0 blocks. n must be a multiple of 32. */
void tq_quantize_q8(const float *x, TqBlockQ80 *out, int n);

/* Dequantize one row of `n` values from a Q4_0/Q8_0 block array. */
void tq_dequantize(const void *blocks, uint32_t dtype, float *out, int n);

/* out[0..d) = W[d, n] @ xq, where xq holds n/32 Q8_0 blocks. */
int  tq_matvec(TqRuntime *rt, float *out, uint64_t w_off, uint32_t dtype,
	       int n, int d, const TqBlockQ80 *xq);

/* Single-row dot products. `nb` = number of 32-value blocks. */
float tq_dot_q40_q80(const TqBlockQ40 *w, const TqBlockQ80 *x, int nb);
float tq_dot_q80_q80(const TqBlockQ80 *w, const TqBlockQ80 *x, int nb);

/* IEEE-754 half <-> float. Exact, branch-light, no FP16 hardware needed. */
float    tq_half_to_float(uint16_t h);
uint16_t tq_float_to_half(float f);

/* Name of the active kernel backend, e.g. "cortex-m7-dsp" or "generic-c99". */
const char *tq_kernel_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* TQ_H_ */

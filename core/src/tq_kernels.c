/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Quantization primitives and the matrix-vector kernels.
 *
 * Two backends live here and they are required to agree BIT-FOR-BIT:
 *
 *   generic-c99    portable reference, used on the host and on any core
 *                  without the ARM DSP extension.
 *   cortex-m7-dsp  SSUB8 / SXTB16 / SMLAD, ~4 int8 MACs per two instructions.
 *
 * Bit-exactness is not an accident, it is the test strategy. Both backends
 * accumulate int32 per 32-value block and only then fold in the two fp16
 * scales, so the integer sums are identical (integer addition is associative)
 * and the float operations happen in the same order. host/tests/test_kernels.c
 * asserts equality, which means a host test failure is a device bug.
 */

#include <string.h>

#include "tq/tq.h"

/*
 * The Python exporter in tools/etq/format.py writes these structures byte for
 * byte. If a compiler ever pads them differently the model file silently
 * misparses, so pin the layout at compile time.
 */
#define TQ_STATIC_ASSERT(cond, tag) typedef char tq_sa_##tag[(cond) ? 1 : -1]
TQ_STATIC_ASSERT(sizeof(TqHeader) == TQ_HEADER_BYTES, header_is_256);
TQ_STATIC_ASSERT(sizeof(TqTensorEntry) == 64, entry_is_64);
TQ_STATIC_ASSERT(sizeof(TqBlockQ80) == 34, q80_block_is_34);
TQ_STATIC_ASSERT(sizeof(TqBlockQ40) == 18, q40_block_is_18);

/* --------------------------------------------------------- half precision */

float tq_half_to_float(uint16_t h)
{
	uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
	uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
	uint32_t mant = (uint32_t)h & 0x3FFu;
	uint32_t bits;
	float    f;

	if (exp == 0u) {
		if (mant == 0u) {
			bits = sign;
		} else {
			/* Subnormal half -> normal float. Shift the mantissa up
			 * until the implicit bit appears, paying for it in the
			 * exponent. */
			uint32_t e = 127u - 15u + 1u;

			while ((mant & 0x400u) == 0u) {
				mant <<= 1;
				e--;
			}
			mant &= 0x3FFu;
			bits = sign | (e << 23) | (mant << 13);
		}
	} else if (exp == 31u) {
		bits = sign | 0x7F800000u | (mant << 13);
	} else {
		bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
	}

	memcpy(&f, &bits, sizeof(f));
	return f;
}

uint16_t tq_float_to_half(float f)
{
	uint32_t x;
	uint32_t sign, mant;
	int32_t  exp;

	memcpy(&x, &f, sizeof(x));
	sign = (x >> 16) & 0x8000u;
	mant = x & 0x7FFFFFu;
	exp  = (int32_t)((x >> 23) & 0xFFu);

	if (exp == 0xFF) {                       /* inf / NaN */
		return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
	}

	exp = exp - 127 + 15;

	if (exp >= 31) {                         /* overflows the half range */
		return (uint16_t)(sign | 0x7C00u);
	}

	if (exp <= 0) {                          /* subnormal half, or zero */
		uint32_t shift, half, rem, halfway;

		if (exp < -10) {
			return (uint16_t)sign;
		}
		mant |= 0x800000u;                    /* restore implicit bit */
		shift   = (uint32_t)(14 - exp);       /* 14..24 */
		half    = mant >> shift;
		rem     = mant & ((1u << shift) - 1u);
		halfway = 1u << (shift - 1u);
		/* round half to even */
		if (rem > halfway || (rem == halfway && (half & 1u))) {
			half++;
		}
		return (uint16_t)(sign | half);
	}

	{
		uint32_t h   = ((uint32_t)exp << 10) | (mant >> 13);
		uint32_t rem = mant & 0x1FFFu;

		/* A carry out of the mantissa lands in the exponent field, which
		 * is exactly the right answer because the fields are adjacent. */
		if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
			h++;
		}
		return (uint16_t)(sign | h);
	}
}

/* ------------------------------------------------------- DSP intrinsics */

#if defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1)
#define TQ_HAVE_DSP 1

/* ------------------------------------------------------------ byte loading
 *
 * Q4_0 blocks are 18 bytes, so every other block starts on a 2-byte boundary
 * and the nibble payload is never 4-byte aligned. memcpy is the portable
 * spelling of an unaligned 32-bit load; GCC turns it into a single LDR on
 * Cortex-M7, which handles unaligned access in Normal memory. (This is also
 * why the PSRAM MPU region must be Normal, not Device — see
 * firmware/.../psram_flexspi2.c.)
 * Only the DSP kernels do 32-bit loads, so it lives inside this guard —
 * outside it, clang's -Wunused-function (which, unlike GCC's, fires for
 * unused static inline functions in C) breaks the -Werror native build.
 */
static inline uint32_t tq_ld32(const void *p)
{
	uint32_t v;

	memcpy(&v, p, sizeof(v));
	return v;
}

static inline uint32_t tq_ssub8(uint32_t a, uint32_t b)
{
	uint32_t r;

	__asm volatile("ssub8 %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
	return r;
}

static inline uint32_t tq_sxtb16(uint32_t a)
{
	uint32_t r;

	__asm volatile("sxtb16 %0, %1" : "=r"(r) : "r"(a));
	return r;
}

static inline int32_t tq_smlad(uint32_t a, uint32_t b, int32_t acc)
{
	int32_t r;

	__asm volatile("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
	return r;
}
#else
#define TQ_HAVE_DSP 0
#endif

const char *tq_kernel_backend(void)
{
#if TQ_HAVE_DSP
	return "cortex-m7-dsp";
#else
	return "generic-c99";
#endif
}

/* ------------------------------------------------------------ dtype sizing */

uint64_t tq_dtype_nbytes(uint32_t dtype, uint64_t n)
{
	switch (dtype) {
	case TQ_DT_F32:
	case TQ_DT_U32:
		return n * 4u;
	case TQ_DT_U8:
		return n;
	case TQ_DT_Q8_0:
		return (n % TQ_GROUP_SIZE) ? 0u : (n / TQ_GROUP_SIZE) * sizeof(TqBlockQ80);
	case TQ_DT_Q4_0:
		return (n % TQ_GROUP_SIZE) ? 0u : (n / TQ_GROUP_SIZE) * sizeof(TqBlockQ40);
	default:
		return 0u;
	}
}

/* --------------------------------------------------------- quantize / deq */

void tq_quantize_q8(const float *x, TqBlockQ80 *out, int n)
{
	int nb = n / TQ_GROUP_SIZE;
	int b;

	for (b = 0; b < nb; b++) {
		const float *src = x + (size_t)b * TQ_GROUP_SIZE;
		float amax = 0.0f;
		float d, id;
		uint16_t dh;
		int j;

		for (j = 0; j < TQ_GROUP_SIZE; j++) {
			float v = src[j] < 0.0f ? -src[j] : src[j];

			if (v > amax) {
				amax = v;
			}
		}

		/* Round the scale to fp16 FIRST and quantize against the rounded
		 * value. Quantizing against the fp32 scale and then storing a
		 * rounded one puts a systematic error into every block. */
		dh = tq_float_to_half(amax / 127.0f);
		d  = tq_half_to_float(dh);
		id = (d != 0.0f) ? (1.0f / d) : 0.0f;

		out[b].d = dh;
		for (j = 0; j < TQ_GROUP_SIZE; j++) {
			float v = src[j] * id;
			int   q = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);

			/* fp16 rounding of d can push a value just past 127. */
			if (q > 127) {
				q = 127;
			} else if (q < -127) {
				q = -127;
			}
			out[b].qs[j] = (int8_t)q;
		}
	}
}

void tq_dequantize(const void *blocks, uint32_t dtype, float *out, int n)
{
	int nb = n / TQ_GROUP_SIZE;
	int b, j;

	if (dtype == TQ_DT_Q8_0) {
		const TqBlockQ80 *w = (const TqBlockQ80 *)blocks;

		for (b = 0; b < nb; b++) {
			float d = tq_half_to_float(w[b].d);

			for (j = 0; j < TQ_GROUP_SIZE; j++) {
				out[(size_t)b * TQ_GROUP_SIZE + j] = (float)w[b].qs[j] * d;
			}
		}
	} else if (dtype == TQ_DT_Q4_0) {
		const TqBlockQ40 *w = (const TqBlockQ40 *)blocks;

		for (b = 0; b < nb; b++) {
			float d = tq_half_to_float(w[b].d);
			float *o = out + (size_t)b * TQ_GROUP_SIZE;

			for (j = 0; j < TQ_GROUP_SIZE / 2; j++) {
				uint8_t byte = w[b].qs[j];

				o[j]      = (float)((int)(byte & 0x0Fu) - 8) * d;
				o[j + 16] = (float)((int)(byte >> 4) - 8) * d;
			}
		}
	} else if (dtype == TQ_DT_F32) {
		memcpy(out, blocks, (size_t)n * sizeof(float));
	}
}

/* ------------------------------------------------------------ dot products */

float tq_dot_q40_q80(const TqBlockQ40 *w, const TqBlockQ80 *x, int nb)
{
	float sum = 0.0f;
	int b;

	for (b = 0; b < nb; b++) {
		const uint8_t *q = w[b].qs;
		const int8_t  *p = x[b].qs;
		int32_t acc = 0;
#if TQ_HAVE_DSP
		int j;

		for (j = 0; j < TQ_GROUP_SIZE / 2; j += 4) {
			uint32_t v  = tq_ld32(q + j);
			/* One 32-bit load yields eight nibbles: the low nibbles are
			 * elements j..j+3, the high nibbles are j+16..j+19. That
			 * split is the entire reason the packing is what it is. */
			uint32_t lo = tq_ssub8(v & 0x0F0F0F0Fu, 0x08080808u);
			uint32_t hi = tq_ssub8((v >> 4) & 0x0F0F0F0Fu, 0x08080808u);
			uint32_t xa = tq_ld32(p + j);
			uint32_t xh = tq_ld32(p + j + 16);

			acc = tq_smlad(tq_sxtb16(lo),      tq_sxtb16(xa),      acc);
			acc = tq_smlad(tq_sxtb16(lo >> 8), tq_sxtb16(xa >> 8), acc);
			acc = tq_smlad(tq_sxtb16(hi),      tq_sxtb16(xh),      acc);
			acc = tq_smlad(tq_sxtb16(hi >> 8), tq_sxtb16(xh >> 8), acc);
		}
#else
		int j;

		for (j = 0; j < TQ_GROUP_SIZE / 2; j++) {
			int v0 = (int)(q[j] & 0x0Fu) - 8;
			int v1 = (int)(q[j] >> 4) - 8;

			acc += v0 * (int)p[j] + v1 * (int)p[j + 16];
		}
#endif
		sum += (float)acc * tq_half_to_float(w[b].d) * tq_half_to_float(x[b].d);
	}
	return sum;
}

float tq_dot_q80_q80(const TqBlockQ80 *w, const TqBlockQ80 *x, int nb)
{
	float sum = 0.0f;
	int b;

	for (b = 0; b < nb; b++) {
		const int8_t *q = w[b].qs;
		const int8_t *p = x[b].qs;
		int32_t acc = 0;
		int j;

#if TQ_HAVE_DSP
		for (j = 0; j < TQ_GROUP_SIZE; j += 4) {
			uint32_t wv = tq_ld32(q + j);
			uint32_t xv = tq_ld32(p + j);

			acc = tq_smlad(tq_sxtb16(wv),      tq_sxtb16(xv),      acc);
			acc = tq_smlad(tq_sxtb16(wv >> 8), tq_sxtb16(xv >> 8), acc);
		}
#else
		for (j = 0; j < TQ_GROUP_SIZE; j++) {
			acc += (int)q[j] * (int)p[j];
		}
#endif
		sum += (float)acc * tq_half_to_float(w[b].d) * tq_half_to_float(x[b].d);
	}
	return sum;
}

/* --------------------------------------------------------------- the store */

void tq_store_init_mapped(TqStore *st, const void *base, uint64_t bytes)
{
	memset(st, 0, sizeof(*st));
	st->base  = (const uint8_t *)base;
	st->bytes = bytes;
}

void tq_store_init_stream(TqStore *st, uint64_t bytes,
			  int (*read)(void *, uint64_t, void *, uint32_t),
			  void *ctx, uint8_t *tile, uint32_t tile_bytes)
{
	memset(st, 0, sizeof(*st));
	st->bytes      = bytes;
	st->read       = read;
	st->ctx        = ctx;
	st->tile       = tile;
	st->tile_bytes = tile_bytes;
}

/* Fetch `nbytes` at `off` into the tile (or hand back a direct pointer). */
static const void *tq_store_view(const TqStore *st, uint64_t off, uint32_t nbytes)
{
	if (off + nbytes > st->bytes) {
		return NULL;
	}
	if (st->base != NULL) {
		return st->base + off;
	}
	if (st->read == NULL || st->tile == NULL || nbytes > st->tile_bytes) {
		return NULL;
	}
	if (st->read(st->ctx, off, st->tile, nbytes) != 0) {
		return NULL;
	}
	return st->tile;
}

/* ---------------------------------------------------------------- matvec */

int tq_matvec(TqRuntime *rt, float *out, uint64_t w_off, uint32_t dtype,
	      int n, int d, const TqBlockQ80 *xq)
{
	const TqStore *st = rt->m->store;
	uint64_t row_bytes = tq_dtype_nbytes(dtype, (uint64_t)n);
	int nb = n / TQ_GROUP_SIZE;
	int rows_per_tile;
	int i;

	if (row_bytes == 0u || (n % TQ_GROUP_SIZE) != 0) {
		return TQ_ERR_SHAPE;
	}

	rt->bytes_read += row_bytes * (uint64_t)d;

	if (st->base != NULL) {
		/* Mapped: the hot path. One sequential sweep, no copies. */
		const uint8_t *p = st->base + w_off;

		if (w_off + row_bytes * (uint64_t)d > st->bytes) {
			return TQ_ERR_TRUNCATED;
		}
		if (dtype == TQ_DT_Q4_0) {
			for (i = 0; i < d; i++) {
				out[i] = tq_dot_q40_q80((const TqBlockQ40 *)p, xq, nb);
				p += row_bytes;
			}
		} else if (dtype == TQ_DT_Q8_0) {
			for (i = 0; i < d; i++) {
				out[i] = tq_dot_q80_q80((const TqBlockQ80 *)p, xq, nb);
				p += row_bytes;
			}
		} else {
			return TQ_ERR_UNSUPPORTED;
		}
		return TQ_OK;
	}

	/* Streaming: pull as many whole rows per read as the tile allows. Big
	 * reads matter here — SDIO gets ~22 MB/s on 32 KB reads and a small
	 * fraction of that on 512 B ones. */
	if (st->tile_bytes < row_bytes) {
		return TQ_ERR_NOMEM;
	}
	rows_per_tile = (int)(st->tile_bytes / row_bytes);

	for (i = 0; i < d; ) {
		int chunk = d - i;
		const uint8_t *p;
		int r;

		if (chunk > rows_per_tile) {
			chunk = rows_per_tile;
		}
		p = (const uint8_t *)tq_store_view(st, w_off + row_bytes * (uint64_t)i,
						   (uint32_t)(row_bytes * (uint64_t)chunk));
		if (p == NULL) {
			return TQ_ERR_IO;
		}
		for (r = 0; r < chunk; r++) {
			if (dtype == TQ_DT_Q4_0) {
				out[i + r] = tq_dot_q40_q80((const TqBlockQ40 *)p, xq, nb);
			} else if (dtype == TQ_DT_Q8_0) {
				out[i + r] = tq_dot_q80_q80((const TqBlockQ80 *)p, xq, nb);
			} else {
				return TQ_ERR_UNSUPPORTED;
			}
			p += row_bytes;
		}
		i += chunk;
	}
	return TQ_OK;
}

/* Read one row of a quantized matrix straight into floats (embedding lookup). */
int tq_row_to_float(const TqModel *m, uint64_t base_off, uint32_t dtype,
		    int n, int row, float *out)
{
	uint64_t row_bytes = tq_dtype_nbytes(dtype, (uint64_t)n);
	const void *p;

	if (row_bytes == 0u) {
		return TQ_ERR_SHAPE;
	}
	p = tq_store_view(m->store, base_off + row_bytes * (uint64_t)row,
			  (uint32_t)row_bytes);
	if (p == NULL) {
		return TQ_ERR_IO;
	}
	tq_dequantize(p, dtype, out, n);
	return TQ_OK;
}

/* Map a byte range for direct reading (tokenizer blobs, f32 norm weights). */
const void *tq_view(const TqModel *m, uint64_t off, uint32_t nbytes)
{
	return tq_store_view(m->store, off, nbytes);
}

const char *tq_strerror(int status)
{
	switch (status) {
	case TQ_OK:              return "ok";
	case TQ_ERR_MAGIC:       return "not an .etq file";
	case TQ_ERR_VERSION:     return "unsupported .etq version";
	case TQ_ERR_TRUNCATED:   return "file truncated";
	case TQ_ERR_UNSUPPORTED: return "unsupported architecture or dtype";
	case TQ_ERR_MISSING:     return "required tensor missing";
	case TQ_ERR_SHAPE:       return "tensor shape mismatch";
	case TQ_ERR_NOMEM:       return "arena too small";
	case TQ_ERR_IO:          return "backing store read failed";
	case TQ_ERR_RANGE:       return "argument out of range";
	default:                 return "unknown error";
	}
}

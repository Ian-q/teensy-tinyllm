/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Arithmetic coding against the model's own predictive distribution.
 *
 * Two devices holding the identical .etq transmit the *residual surprise* of a
 * message instead of its text. The model supplies P(next token | context) at
 * every step; a range coder spends -log2(P) bits on the token that actually
 * occurred. On text the model finds ordinary this approaches 1 bit per
 * character, where a classical codec on a 60-byte payload cannot get below
 * about 6 — there is no room to build a dictionary that short.
 *
 * DETERMINISM IS THE WHOLE CONTRACT. The decoder reconstructs the same
 * probability table the encoder used, so any disagreement — one count off by
 * one, anywhere — desynchronises the range and destroys the rest of the
 * message. Everything below the logits is therefore integer arithmetic with
 * no libm, no floating-point accumulation order to get wrong, and no
 * platform-dependent rounding. The logits themselves are still fp32 out of
 * tq_forward(), which calls expf/sinf/cosf; two boards running the identical
 * binary agree bit-for-bit, but host<->device does not yet. See docs/STATUS.md.
 */

#ifndef TQ_AC_H_
#define TQ_AC_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A range coder divides its 32-bit range by the frequency total, so the total
 * must stay well under the range floor or the division loses too much
 * precision. TOP = 2^24 with a 2^16 total keeps at least 8 bits of headroom
 * per symbol, which is the standard, well-tested operating point.
 */
#define TQ_AC_TOT_BITS    16
#define TQ_AC_TOT         (1u << TQ_AC_TOT_BITS)
#define TQ_AC_TOP         (1u << 24)

/*
 * ...but a 2^16 total cannot describe a 32000-symbol alphabet. Every symbol
 * needs a count of at least 1 or it is unencodable, and 32000 of those consume
 * half the mass: a token the model is 90% sure of would be coded at an
 * effective 46%, costing an extra bit every step.
 *
 * So the vocabulary is coded in two levels — first which bucket of 128, then
 * which symbol inside it. Neither level has more than a few hundred symbols,
 * so the reserved minimum counts cost ~0.01 bits per token instead of ~1.0,
 * and the product of the two probabilities is still exactly P(symbol).
 */
#define TQ_AC_BUCKET      128
#define TQ_AC_MAX_BUCKETS 512
#define TQ_AC_MAX_VOCAB   (TQ_AC_BUCKET * TQ_AC_MAX_BUCKETS)

/* --------------------------------------------------------------- the coder */

typedef struct {
	uint8_t *buf;
	size_t   cap;
	size_t   len;

	uint64_t low;
	uint32_t range;
	uint8_t  cache;
	uint64_t pending;   /* deferred 0xFF bytes awaiting a carry decision */
	int      primed;    /* cache holds a real byte, not the initial zero */
	int      finished;  /* flushed; further finish() calls are no-ops */
	int      overflow;  /* output did not fit in cap */
} TqAcEnc;

typedef struct {
	const uint8_t *buf;
	size_t         len;
	size_t         pos;

	uint32_t range;
	uint32_t code;
	uint32_t r;         /* range/total, carried between target and update */
	int      starved;   /* read past the end of the buffer */
} TqAcDec;

/*
 * The reconstructed probability table for one decode step. Holds counts, not
 * probabilities: `bcnt` sums to TQ_AC_TOT across the buckets, and `scnt` sums
 * to TQ_AC_TOT across the symbols of whichever bucket was last requested.
 */
typedef struct {
	int      vocab;
	int      nbuckets;
	int      bucket;                       /* bucket scnt currently describes */
	uint32_t bcnt[TQ_AC_MAX_BUCKETS];
	uint32_t scnt[TQ_AC_BUCKET];
	/*
	 * Unnormalised weights, held here rather than on the stack: summing a
	 * 32000-symbol vocabulary needs 64-bit accumulators, and 4 KB of locals
	 * does not belong on a Zephyr thread stack.
	 */
	uint64_t w[TQ_AC_MAX_BUCKETS];
} TqAcModel;

/* ------------------------------------------------------------------- setup */

/* Returns TQ_OK, or TQ_ERR_ARG if vocab is out of range for the bucketing. */
int tq_ac_model_init(TqAcModel *pm, int vocab);

void tq_ac_enc_init(TqAcEnc *e, uint8_t *buf, size_t cap);
void tq_ac_dec_init(TqAcDec *d, const uint8_t *buf, size_t len);

/* -------------------------------------------------------------- token codec */

/*
 * Code one token against `logits` (vocab_size entries, as written by
 * tq_forward). Neither call modifies the logits — unlike tq_sample(), which
 * softmaxes them in place.
 *
 * tq_ac_enc_token returns TQ_OK or TQ_ERR_ARG for a token outside the vocab.
 * tq_ac_dec_token returns the token id, or a negative TqStatus.
 */
int tq_ac_enc_token(TqAcEnc *e, TqAcModel *pm, const float *logits, int token);
int tq_ac_dec_token(TqAcDec *d, TqAcModel *pm, const float *logits);

/* Flush the encoder and return the byte count. Idempotent. */
size_t tq_ac_enc_finish(TqAcEnc *e);

/* ------------------------------------------------- primitives, for testing
 *
 * Exposed so the coder can be tested against synthetic distributions with no
 * model in the loop, which is where an off-by-one in the carry logic shows up
 * as something other than "the text came back wrong".
 */

/*
 * `tot` must be <= TQ_AC_TOT and the counts must tile [0, tot) exactly. A
 * larger total does not fail loudly — it quietly costs precision, because the
 * coder's range/total division stops leaving enough bits per symbol.
 */
void     tq_ac_enc_sym(TqAcEnc *e, uint32_t cum, uint32_t freq, uint32_t tot);
uint32_t tq_ac_dec_target(TqAcDec *d, uint32_t tot);
void     tq_ac_dec_update(TqAcDec *d, uint32_t cum, uint32_t freq);

/*
 * 2^(x/65536) in Q24, for x <= 0. Integer-only and exactly reproducible on any
 * conforming C99 implementation, which is why the probability table does not
 * call expf().
 */
uint32_t tq_ac_exp2_q24(int32_t x_q16);

/*
 * Fill pm->bcnt from `logits`, then pm->scnt for one bucket. tq_ac_enc_token
 * and tq_ac_dec_token call these; they are exposed so a test can assert the
 * two sides build byte-identical tables.
 */
void tq_ac_bucket_counts(TqAcModel *pm, const float *logits);
void tq_ac_symbol_counts(TqAcModel *pm, const float *logits, int bucket);

#ifdef __cplusplus
}
#endif

#endif /* TQ_AC_H_ */

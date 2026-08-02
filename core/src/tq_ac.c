/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Range coder plus the integer probability table it codes against.
 *
 * The coder is the standard LZMA-lineage range encoder: a 32-bit range, a
 * 64-bit low accumulator, and carry propagation deferred through a cache byte
 * and a run of pending 0xFFs. It is used here with explicit (cumulative,
 * frequency, total) triples rather than adaptive bit models, because the
 * probabilities come from a 15M-parameter transformer rather than from
 * counting the bytes seen so far.
 *
 * Nothing in this file calls libm or accumulates in floating point. The only
 * float operations are one subtraction and one multiply per logit, in a fixed
 * order, feeding a truncating conversion to int32 — all of which C99 pins
 * exactly. See the determinism note in tq_ac.h for what is *not* yet pinned.
 */

#include <string.h>

#include "tq/tq.h"
#include "tq/tq_ac.h"

/* ------------------------------------------------------- fixed-point exp2 */

/*
 * 2^f over f in [0,1) in Q24, as a degree-6 near-minimax polynomial fitted at
 * Chebyshev nodes.
 *
 * Deliberately NOT the Taylor expansion of exp(f*ln2): truncated at the same
 * degree, Taylor is off by 1.7e-4 at the top of the interval — 892 LSB — while
 * this is off by 5. The leading coefficient is exactly 2^24, so the most
 * probable symbol always weighs exactly 1.0 and the sequence is monotone
 * across all 65536 inputs.
 *
 * Evaluated by Horner in int64: nothing overflows (|acc| < 2^25, f < 2^16), the
 * operation order is fixed, and every intermediate is an exact integer. That is
 * what makes this reproducible on a machine that has never seen the encoder.
 */
static const int64_t TQ_EXP2_C[7] = {
	16777216,   /* 1.0                */
	11629076,   /* 0.6931469440460205 */
	4030398,    /* 0.2402304410934448 */
	930811,     /* 0.0554806590080261 */
	162474,     /* 0.0096842050552368 */
	20789,      /* 0.0012391209602356 */
	3668,       /* 0.0002186298370361 */
};

uint32_t tq_ac_exp2_q24(int32_t x_q16)
{
	int32_t ip;
	int64_t f, acc;
	int     sh, k;

	if (x_q16 > 0) {
		x_q16 = 0;                /* callers only ever pass logit deltas <= 0 */
	}

	/* Arithmetic shift floors toward -inf, which is what splitting a negative
	 * exponent into (integer, fraction in [0,1)) requires. */
	ip = x_q16 >> 16;
	f  = (int64_t)(uint32_t)(x_q16 & 0xFFFF);

	acc = TQ_EXP2_C[6];
	for (k = 5; k >= 0; k--) {
		acc = ((acc * f) >> 16) + TQ_EXP2_C[k];
	}

	sh = -ip;
	if (sh >= 32) {
		return 0u;
	}
	return (uint32_t)(acc >> sh);
}

/*
 * Convert a logit delta (always <= 0) to Q16 units of log2.
 *
 * The clamp matters for two reasons. It keeps the multiply inside int32 — a
 * float-to-int conversion that overflows is undefined behaviour, and logits
 * from a broken model can be arbitrarily negative. And 18 nats below the
 * maximum is already 1.5e-8 of the mass, far below what a 16-bit count can
 * represent, so clamping there changes no output.
 */
#define TQ_AC_MIN_DELTA (-18.0f)
#define TQ_AC_LOG2E_Q16 (1.4426950408889634f * 65536.0f)

static int32_t tq_ac_delta_q16(float logit, float max)
{
	float d = logit - max;

	if (!(d > TQ_AC_MIN_DELTA)) {   /* also catches NaN */
		d = TQ_AC_MIN_DELTA;
	}
	if (d > 0.0f) {
		d = 0.0f;
	}
	return (int32_t)(d * TQ_AC_LOG2E_Q16);
}

/* ------------------------------------------------------ probability tables */

int tq_ac_model_init(TqAcModel *pm, int vocab)
{
	if (vocab < 1 || vocab > TQ_AC_MAX_VOCAB) {
		return TQ_ERR_RANGE;
	}
	memset(pm, 0, sizeof(*pm));
	pm->vocab    = vocab;
	pm->nbuckets = (vocab + TQ_AC_BUCKET - 1) / TQ_AC_BUCKET;
	pm->bucket   = -1;
	return TQ_OK;
}

/*
 * Turn unnormalised weights into counts that sum to exactly TQ_AC_TOT, with
 * every count at least 1 so that every symbol stays codeable.
 *
 * Reserving one count per symbol first and distributing the rest
 * proportionally guarantees the floor without a second corrective pass. The
 * truncation leftover goes to the heaviest symbol, which is both the least
 * relative distortion and — more importantly — a rule the decoder applies
 * identically.
 */
static void tq_ac_normalize(uint32_t *cnt, const uint64_t *w, int n)
{
	uint64_t sum = 0;
	uint32_t spread;
	uint32_t used = 0;
	int      i, top = 0;

	for (i = 0; i < n; i++) {
		sum += w[i];
		if (w[i] > w[top]) {
			top = i;
		}
	}

	spread = (uint32_t)TQ_AC_TOT - (uint32_t)n;

	if (sum == 0u) {                       /* degenerate; stay uniform */
		for (i = 0; i < n; i++) {
			cnt[i] = 1u;
		}
		cnt[top] += spread;
		return;
	}

	for (i = 0; i < n; i++) {
		cnt[i] = 1u + (uint32_t)((w[i] * (uint64_t)spread) / sum);
		used += cnt[i];
	}
	cnt[top] += (uint32_t)TQ_AC_TOT - used;
}

void tq_ac_bucket_counts(TqAcModel *pm, const float *logits)
{
	float max = logits[0];
	int   i, b;

	for (i = 1; i < pm->vocab; i++) {
		if (logits[i] > max) {
			max = logits[i];
		}
	}

	for (b = 0; b < pm->nbuckets; b++) {
		pm->w[b] = 0;
	}
	/* Bucket mass is the exact sum of its symbols' weights, so the two coding
	 * levels multiply back out to the symbol's own probability. */
	for (i = 0; i < pm->vocab; i++) {
		pm->w[i / TQ_AC_BUCKET] +=
			tq_ac_exp2_q24(tq_ac_delta_q16(logits[i], max));
	}

	tq_ac_normalize(pm->bcnt, pm->w, pm->nbuckets);
}

void tq_ac_symbol_counts(TqAcModel *pm, const float *logits, int bucket)
{
	int   base = bucket * TQ_AC_BUCKET;
	int   n    = pm->vocab - base;
	float max;
	int   i;

	if (n > TQ_AC_BUCKET) {
		n = TQ_AC_BUCKET;
	}

	/*
	 * Renormalising against the bucket's own maximum rather than the global
	 * one is what keeps the within-bucket resolution usable: a bucket 30 nats
	 * below the mode would otherwise quantize to all-ones and cost 7 bits to
	 * say nothing.
	 */
	max = logits[base];
	for (i = 1; i < n; i++) {
		if (logits[base + i] > max) {
			max = logits[base + i];
		}
	}
	for (i = 0; i < n; i++) {
		pm->w[i] = tq_ac_exp2_q24(tq_ac_delta_q16(logits[base + i], max));
	}

	tq_ac_normalize(pm->scnt, pm->w, n);
	pm->bucket = bucket;
}

/* ------------------------------------------------------------- the encoder */

void tq_ac_enc_init(TqAcEnc *e, uint8_t *buf, size_t cap)
{
	memset(e, 0, sizeof(*e));
	e->buf   = buf;
	e->cap   = cap;
	e->range = 0xFFFFFFFFu;
}

static void tq_ac_put(TqAcEnc *e, uint8_t byte)
{
	if (e->len < e->cap) {
		e->buf[e->len++] = byte;
	} else {
		e->overflow = 1;
	}
}

/*
 * Emit the top byte of `low`, resolving any carry that reached it.
 *
 * A byte that is exactly 0xFF cannot be emitted yet: a later carry would turn
 * it into 0x00 and ripple further. Those are counted in `pending` and settled
 * once a byte arrives that can absorb the carry. `primed` suppresses the
 * leading zero byte the classic formulation emits — the decoder compensates by
 * loading only four bytes at init, which saves a byte on a nine-byte message.
 */
static void tq_ac_shift_low(TqAcEnc *e)
{
	if ((uint32_t)(e->low >> 32) != 0u || (uint32_t)e->low < 0xFF000000u) {
		uint8_t carry = (uint8_t)(e->low >> 32);

		if (e->primed) {
			tq_ac_put(e, (uint8_t)(e->cache + carry));
		}
		while (e->pending != 0u) {
			tq_ac_put(e, (uint8_t)(0xFFu + carry));
			e->pending--;
		}
		e->cache  = (uint8_t)(e->low >> 24);
		e->primed = 1;
	} else {
		e->pending++;
	}
	e->low = (uint64_t)((uint32_t)e->low << 8);
}

void tq_ac_enc_sym(TqAcEnc *e, uint32_t cum, uint32_t freq, uint32_t tot)
{
	uint32_t r = e->range / tot;

	e->low  += (uint64_t)r * cum;
	e->range = r * freq;

	while (e->range < TQ_AC_TOP) {
		tq_ac_shift_low(e);
		e->range <<= 8;
	}
}

/*
 * Flush, then shorten the message.
 *
 * The coder's final state says only that the message decodes to some point in
 * [low, low+range) — any value in that interval will do, so pick the one with
 * the most trailing zero bytes. Because the decoder feeds zeroes once it runs
 * off the end of the buffer (see tq_ac_get), those trailing zeroes need not be
 * transmitted at all, and truncating them is exactly lossless.
 *
 * This is worth real effort at these sizes. A textbook five-byte flush costs
 * four bytes on a message whose payload is six, so the naive version spends
 * 40% of a short transmission saying nothing. Since `range` is never below
 * TQ_AC_TOP = 2^24, rounding always succeeds at the first step and the tail
 * collapses to a single significant byte.
 */
size_t tq_ac_enc_finish(TqAcEnc *e)
{
	int i;

	if (!e->finished) {
		e->finished = 1;

		for (i = 1; i <= 4; i++) {
			uint32_t mask = (i == 4) ? 0u : (0xFFFFFFFFu >> (8 * i));
			uint64_t v    = (e->low + mask) & ~(uint64_t)mask;

			if (v < e->low + (uint64_t)e->range) {
				e->low = v;
				break;
			}
		}

		for (i = 0; i < 5; i++) {
			tq_ac_shift_low(e);
		}

		while (e->len > 0 && e->buf[e->len - 1] == 0u) {
			e->len--;
		}
	}
	return e->len;
}

/* ------------------------------------------------------------- the decoder */

void tq_ac_dec_init(TqAcDec *d, const uint8_t *buf, size_t len)
{
	int i;

	memset(d, 0, sizeof(*d));
	d->buf   = buf;
	d->len   = len;
	d->range = 0xFFFFFFFFu;

	for (i = 0; i < 4; i++) {
		d->code = (d->code << 8) | (uint32_t)(d->pos < d->len ? d->buf[d->pos++] : 0u);
	}
}

static uint32_t tq_ac_get(TqAcDec *d)
{
	if (d->pos < d->len) {
		return d->buf[d->pos++];
	}
	/*
	 * Reading past the end is normal for the last few symbols — the encoder's
	 * flush carries fewer bytes than the decoder consumes — so this feeds
	 * zeroes rather than failing. It is only a real error if the *stream* was
	 * truncated, which shows up as a decoded token the caller rejects.
	 */
	d->starved = 1;
	return 0u;
}

uint32_t tq_ac_dec_target(TqAcDec *d, uint32_t tot)
{
	uint32_t t;

	d->r = d->range / tot;
	t    = d->code / d->r;
	return (t >= tot) ? tot - 1u : t;
}

void tq_ac_dec_update(TqAcDec *d, uint32_t cum, uint32_t freq)
{
	d->code -= cum * d->r;
	d->range = d->r * freq;

	while (d->range < TQ_AC_TOP) {
		d->code = (d->code << 8) | tq_ac_get(d);
		d->range <<= 8;
	}
}

/* ------------------------------------------------------------ token codec */

/* Cumulative count below index `i`. Linear, but the whole table is at most 512
 * entries and the caller just spent ~300 ms reading weights to produce it. */
static uint32_t tq_ac_cum(const uint32_t *cnt, int i)
{
	uint32_t c = 0;
	int      k;

	for (k = 0; k < i; k++) {
		c += cnt[k];
	}
	return c;
}

static int tq_ac_find(const uint32_t *cnt, int n, uint32_t target, uint32_t *cum)
{
	uint32_t c = 0;
	int      i;

	for (i = 0; i < n; i++) {
		if (target < c + cnt[i]) {
			*cum = c;
			return i;
		}
		c += cnt[i];
	}
	*cum = c - cnt[n - 1];       /* unreachable while counts sum to TOT */
	return n - 1;
}

int tq_ac_enc_token(TqAcEnc *e, TqAcModel *pm, const float *logits, int token)
{
	int      b, s;
	uint32_t cum;

	if (token < 0 || token >= pm->vocab) {
		return TQ_ERR_RANGE;
	}
	b = token / TQ_AC_BUCKET;
	s = token % TQ_AC_BUCKET;

	tq_ac_bucket_counts(pm, logits);
	cum = tq_ac_cum(pm->bcnt, b);
	tq_ac_enc_sym(e, cum, pm->bcnt[b], TQ_AC_TOT);

	tq_ac_symbol_counts(pm, logits, b);
	cum = tq_ac_cum(pm->scnt, s);
	tq_ac_enc_sym(e, cum, pm->scnt[s], TQ_AC_TOT);

	return TQ_OK;
}

int tq_ac_dec_token(TqAcDec *d, TqAcModel *pm, const float *logits)
{
	uint32_t target, cum;
	int      b, s, n, token;

	tq_ac_bucket_counts(pm, logits);
	target = tq_ac_dec_target(d, TQ_AC_TOT);
	b      = tq_ac_find(pm->bcnt, pm->nbuckets, target, &cum);
	tq_ac_dec_update(d, cum, pm->bcnt[b]);

	tq_ac_symbol_counts(pm, logits, b);
	n = pm->vocab - b * TQ_AC_BUCKET;
	if (n > TQ_AC_BUCKET) {
		n = TQ_AC_BUCKET;
	}
	target = tq_ac_dec_target(d, TQ_AC_TOT);
	s      = tq_ac_find(pm->scnt, n, target, &cum);
	tq_ac_dec_update(d, cum, pm->scnt[s]);

	token = b * TQ_AC_BUCKET + s;
	return (token < pm->vocab) ? token : TQ_ERR_TRUNCATED;
}

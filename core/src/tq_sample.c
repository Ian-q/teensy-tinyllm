/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Logit sampling: greedy, temperature, and nucleus (top-p).
 *
 * Deliberately free of qsort. Sorting a 32000-entry vocabulary per token with
 * a libc comparator callback costs more than the attention block on a 42M
 * model. Instead top-p prefilters with an analytic cutoff (any token below
 * (1-p)/(V-1) can never enter the nucleus) which typically leaves a few dozen
 * to a few hundred candidates, then heapsorts those in place.
 */

#include <math.h>

#include "tq/tq_math.h"

#include "tq/tq.h"

/* xorshift64*, from Vigna. Cheap, well-distributed, and reproducible across
 * host and device, which matters because the host tests compare generated
 * token sequences against a fixed seed. */
static uint32_t tq_rand_u32(uint64_t *s)
{
	uint64_t x = *s;

	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	*s = x;
	return (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
}

static float tq_rand_f32(uint64_t *s)
{
	/* 24 mantissa bits, half-open [0, 1). */
	return (float)(tq_rand_u32(s) >> 8) * (1.0f / 16777216.0f);
}

void tq_sampler_init(TqSampler *s, int vocab_size, float temperature, float topp,
		     uint64_t seed, int32_t *idx_scratch, float *prob_scratch)
{
	s->rng         = seed ? seed : 0x853c49e6748fea9bULL;
	s->temperature = temperature;
	s->topp        = topp;
	s->vocab_size  = vocab_size;
	s->idx         = idx_scratch;
	s->prob        = prob_scratch;
}

static int tq_argmax(const float *v, int n)
{
	int best = 0;
	int i;

	for (i = 1; i < n; i++) {
		if (v[i] > v[best]) {
			best = i;
		}
	}
	return best;
}

/* Sift `root` down a max-heap ordered by prob[]. */
static void tq_sift(float *prob, int32_t *idx, int root, int n)
{
	for (;;) {
		int child = 2 * root + 1;
		int big = root;
		float t;
		int32_t ti;

		if (child < n && prob[child] > prob[big]) {
			big = child;
		}
		if (child + 1 < n && prob[child + 1] > prob[big]) {
			big = child + 1;
		}
		if (big == root) {
			return;
		}
		t = prob[root]; prob[root] = prob[big]; prob[big] = t;
		ti = idx[root]; idx[root] = idx[big]; idx[big] = ti;
		root = big;
	}
}

/* Descending sort of the parallel (prob, idx) arrays. */
static void tq_heapsort_desc(float *prob, int32_t *idx, int n)
{
	int i;

	for (i = n / 2 - 1; i >= 0; i--) {
		tq_sift(prob, idx, i, n);
	}
	/* Popping the max to the back yields ascending order, so walk the
	 * result backwards at the call site... or just reverse here. */
	for (i = n - 1; i > 0; i--) {
		float t = prob[0];
		int32_t ti = idx[0];

		prob[0] = prob[i]; prob[i] = t;
		idx[0] = idx[i];   idx[i] = ti;
		tq_sift(prob, idx, 0, i);
	}
	for (i = 0; i < n / 2; i++) {
		float t = prob[i];
		int32_t ti = idx[i];

		prob[i] = prob[n - 1 - i]; prob[n - 1 - i] = t;
		idx[i] = idx[n - 1 - i];   idx[n - 1 - i] = ti;
	}
}

int tq_sample(TqSampler *s, float *logits)
{
	int n = s->vocab_size;
	int i;
	float sum = 0.0f, max;
	float r, cdf;

	if (s->temperature <= 0.0f) {
		return tq_argmax(logits, n);
	}

	/* softmax(logits / T) in place */
	{
		float inv_t = 1.0f / s->temperature;

		for (i = 0; i < n; i++) {
			logits[i] *= inv_t;
		}
		max = logits[tq_argmax(logits, n)];
		for (i = 0; i < n; i++) {
			logits[i] = tq_expf(logits[i] - max);
			sum += logits[i];
		}
		sum = 1.0f / sum;
		for (i = 0; i < n; i++) {
			logits[i] *= sum;
		}
	}

	if (s->topp <= 0.0f || s->topp >= 1.0f || s->idx == NULL || s->prob == NULL) {
		/* Plain multinomial over the full distribution. */
		r = tq_rand_f32(&s->rng);
		cdf = 0.0f;
		for (i = 0; i < n; i++) {
			cdf += logits[i];
			if (r < cdf) {
				return i;
			}
		}
		return n - 1;
	}

	/* Nucleus. Everything below `cutoff` is provably outside the top-p
	 * mass, so it never needs sorting. */
	{
		float cutoff = (1.0f - s->topp) / (float)(n - 1);
		int k = 0;
		float cum = 0.0f;
		int last;

		for (i = 0; i < n; i++) {
			if (logits[i] >= cutoff) {
				s->idx[k] = (int32_t)i;
				s->prob[k] = logits[i];
				k++;
			}
		}
		if (k == 0) {
			return tq_argmax(logits, n);
		}
		tq_heapsort_desc(s->prob, s->idx, k);

		last = k - 1;
		for (i = 0; i < k; i++) {
			cum += s->prob[i];
			if (cum > s->topp) {
				last = i;
				break;
			}
		}

		r = tq_rand_f32(&s->rng) * cum;
		cdf = 0.0f;
		for (i = 0; i <= last; i++) {
			cdf += s->prob[i];
			if (r < cdf) {
				return (int)s->idx[i];
			}
		}
		return (int)s->idx[last];
	}
}

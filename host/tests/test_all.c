/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * libtq test suite. Runs on the host and, cross-compiled, under qemu-arm with
 * the ARM DSP extension enabled — which is how the SMLAD kernel gets executed
 * and checked without hardware in the loop. Same instructions the Cortex-M7
 * issues, same results.
 *
 *   make -C host test        # native
 *   make -C host test-arm    # armv7e-m style DSP kernels under QEMU
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tq/tq.h"
#include "tq/tq_ac.h"
#include "golden.h"

static int g_fail;
static int g_checks;

#define CHECK(cond, ...)                                            \
	do {                                                        \
		g_checks++;                                         \
		if (!(cond)) {                                      \
			g_fail++;                                   \
			printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
			printf(__VA_ARGS__);                        \
			printf("\n");                               \
		}                                                   \
	} while (0)

/* ------------------------------------------------------------ model bytes */

static uint8_t *g_buf;
static size_t g_len;

static int load_file(const char *path)
{
	int fd = open(path, O_RDONLY);
	struct stat st;
	ssize_t n;

	if (fd < 0 || fstat(fd, &st) != 0) {
		return -1;
	}
	g_len = (size_t)st.st_size;
	g_buf = malloc(g_len);
	n = read(fd, g_buf, g_len);
	close(fd);
	return (n == (ssize_t)g_len) ? 0 : -1;
}

/* ------------------------------------------------------- half-precision */

static void test_half(void)
{
	/* Values that historically break hand-rolled fp16 codecs: subnormals,
	 * the normal/subnormal boundary, exact powers of two, and the
	 * round-half-to-even ties. */
	static const float vals[] = {
		0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f,
		6.1035156e-05f,   /* smallest normal half            */
		5.9604645e-08f,   /* smallest subnormal half         */
		3.0517578e-05f,   /* mid-subnormal                   */
		1.0f / 3.0f, 1e-8f, 123.456f, -0.007f,
	};
	size_t i;

	printf("half-precision round trip\n");
	for (i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
		float back = tq_half_to_float(tq_float_to_half(vals[i]));
		float err = fabsf(back - vals[i]);
		float tol = fabsf(vals[i]) * 1e-3f + 6e-8f;

		CHECK(err <= tol, "%g -> %g (err %g > tol %g)",
		      (double)vals[i], (double)back, (double)err, (double)tol);
	}
	/* Denormal flush and exactness at representable values. */
	CHECK(tq_half_to_float(tq_float_to_half(0.25f)) == 0.25f, "0.25 not exact");
	CHECK(tq_float_to_half(0.0f) == 0u, "+0 not 0x0000");
}

/* ------------------------------------------------------------- kernels */

static uint32_t rnd_state = 12345u;

static uint32_t rnd(void)
{
	rnd_state = rnd_state * 1664525u + 1013904223u;
	return rnd_state;
}

static float rndf(void)
{
	return ((float)(rnd() >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

/*
 * Independent reference: fully dequantize both operands, then dot in double.
 * Written deliberately differently from the kernel (no block accumulator, no
 * nibble tricks) so that a shared misunderstanding of the packing cannot make
 * both agree on the wrong answer.
 */
static double ref_dot(const void *w, uint32_t dtype, const TqBlockQ80 *x, int n)
{
	float *wf = malloc(sizeof(float) * (size_t)n);
	float *xf = malloc(sizeof(float) * (size_t)n);
	double s = 0.0;
	int i;

	tq_dequantize(w, dtype, wf, n);
	tq_dequantize(x, TQ_DT_Q8_0, xf, n);
	for (i = 0; i < n; i++) {
		s += (double)wf[i] * (double)xf[i];
	}
	free(wf);
	free(xf);
	return s;
}

static void test_kernels(void)
{
	enum { N = 256, NB = N / TQ_GROUP_SIZE };
	float *fa = malloc(sizeof(float) * N);
	float *fb = malloc(sizeof(float) * N);
	TqBlockQ80 xq[NB], wq8[NB];
	TqBlockQ40 wq4[NB];
	int trial;

	printf("kernels: q8/q4 dot products vs independent dequantized reference\n");

	for (trial = 0; trial < 200; trial++) {
		double ref, got;
		float scale = (trial % 5 == 0) ? 1e-3f : (trial % 7 == 0) ? 50.0f : 1.0f;
		int i;

		for (i = 0; i < N; i++) {
			fa[i] = rndf() * scale;
			fb[i] = rndf();
		}
		/* A block of exact zeros must not produce NaN via 1/0. */
		if (trial == 3) {
			memset(fa, 0, sizeof(float) * TQ_GROUP_SIZE);
		}

		tq_quantize_q8(fb, xq, N);
		tq_quantize_q8(fa, wq8, N);

		got = (double)tq_dot_q80_q80(wq8, xq, NB);
		ref = ref_dot(wq8, TQ_DT_Q8_0, xq, N);
		CHECK(fabs(got - ref) <= 1e-4 * (fabs(ref) + 1e-6),
		      "q8 trial %d: got %g ref %g", trial, got, ref);

		/* Q4 weights come from the Python quantizer in the real flow;
		 * here derive them by re-encoding the dequantized q8 values
		 * through the same block layout the exporter emits. */
		{
			float tmp[N];
			int b;

			tq_dequantize(wq8, TQ_DT_Q8_0, tmp, N);
			for (b = 0; b < NB; b++) {
				const float *src = tmp + b * TQ_GROUP_SIZE;
				float amax = 0.0f, signed_max = 0.0f, d, id;
				int j;

				for (j = 0; j < TQ_GROUP_SIZE; j++) {
					float m = src[j] < 0 ? -src[j] : src[j];

					if (m > amax) {
						amax = m;
						signed_max = src[j];
					}
				}
				wq4[b].d = tq_float_to_half(signed_max / -8.0f);
				d = tq_half_to_float(wq4[b].d);
				id = (d != 0.0f) ? 1.0f / d : 0.0f;
				for (j = 0; j < TQ_GROUP_SIZE / 2; j++) {
					float v0 = src[j] * id;
					float v1 = src[j + 16] * id;
					int q0 = (int)(v0 < 0 ? v0 - 0.5f : v0 + 0.5f) + 8;
					int q1 = (int)(v1 < 0 ? v1 - 0.5f : v1 + 0.5f) + 8;

					q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
					q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
					wq4[b].qs[j] = (uint8_t)(q0 | (q1 << 4));
				}
			}
		}

		got = (double)tq_dot_q40_q80(wq4, xq, NB);
		ref = ref_dot(wq4, TQ_DT_Q4_0, xq, N);
		CHECK(fabs(got - ref) <= 1e-4 * (fabs(ref) + 1e-6),
		      "q4 trial %d: got %g ref %g", trial, got, ref);
	}

	/* Quantize/dequantize must never produce NaN, including all-zero input. */
	{
		float z[TQ_GROUP_SIZE] = { 0 };
		TqBlockQ80 zb;
		float back[TQ_GROUP_SIZE];
		int i;

		tq_quantize_q8(z, &zb, TQ_GROUP_SIZE);
		tq_dequantize(&zb, TQ_DT_Q8_0, back, TQ_GROUP_SIZE);
		for (i = 0; i < TQ_GROUP_SIZE; i++) {
			CHECK(back[i] == 0.0f, "zero block element %d = %g", i,
			      (double)back[i]);
		}
	}

	free(fa);
	free(fb);
}

/* --------------------------------------------------------------- format */

static void test_format(TqModel *m)
{
	printf("format: header and tensor directory\n");
	CHECK(m->hdr.magic == TQ_MAGIC, "magic 0x%08x", m->hdr.magic);
	CHECK(m->hdr.group_size == TQ_GROUP_SIZE, "group %u", m->hdr.group_size);
	CHECK(m->dim % TQ_GROUP_SIZE == 0, "dim %d", m->dim);
	CHECK(m->kv_dim == m->head_size * m->n_kv_heads, "kv_dim %d", m->kv_dim);
	CHECK(m->kv_mul == m->n_heads / m->n_kv_heads, "kv_mul %d", m->kv_mul);
	CHECK(m->tok_emb != 0, "tok_emb unresolved");
	CHECK(m->rms_final != 0, "rms_final unresolved");
	CHECK(m->classifier != 0, "classifier unresolved");
	CHECK(m->layers[0].wq != 0 && m->layers[m->n_layers - 1].w3 != 0,
	      "layer tensors unresolved");
	CHECK(m->vocab_size == GOLDEN_VOCAB, "vocab %d != golden %d",
	      m->vocab_size, GOLDEN_VOCAB);
}

/* --------------------------------------------------------------- golden */

/* Q4 and Q8 fixtures share the source weights but not the quantization
 * error, so each model is scored against its own oracle. Selected in
 * main() once the model's qtype is known. */
static const float (*g_golden)[GOLDEN_VOCAB] = golden_logits;

static void run_golden(TqRuntime *rt, const char *label)
{
	int step;
	double worst = 0.0;

	printf("golden (%s): %d steps vs NumPy reference\n", label, GOLDEN_STEPS);
	for (step = 0; step < GOLDEN_STEPS; step++) {
		int rc = tq_forward(rt, golden_tokens[step], step);
		int i;

		CHECK(rc == TQ_OK, "step %d: %s", step, tq_strerror(rc));
		if (rc != TQ_OK) {
			return;
		}
		for (i = 0; i < GOLDEN_VOCAB; i++) {
			double got = (double)rt->logits[i];
			double exp = (double)g_golden[step][i];
			double rel = fabs(got - exp) / (fabs(exp) + 1e-3);

			if (rel > worst) {
				worst = rel;
			}
		}
	}
	CHECK(worst < 2e-3, "worst relative logit error %.3g", worst);
	printf("  worst relative logit error: %.3g\n", worst);
}

/* -------------------------------------------------------------- streaming */

struct mem_ctx {
	uint64_t reads;
};

static int mem_read(void *vctx, uint64_t off, void *dst, uint32_t n)
{
	struct mem_ctx *c = (struct mem_ctx *)vctx;

	if (off + n > g_len) {
		return -1;
	}
	c->reads++;
	memcpy(dst, g_buf + off, n);
	return 0;
}

/*
 * The streaming path is what runs when the model lives on the SD card. There
 * is no SD card in CI, so simulate one: same bytes, but reached only through
 * the read callback and a small tile. Logits must come out identical to the
 * mapped path, because the tiling must not change any arithmetic.
 */
static void test_streaming(void)
{
	TqStore st;
	TqModel m;
	TqRuntime rt;
	struct mem_ctx ctx = { 0 };
	uint8_t *tile = malloc(32 * 1024);
	void *arena = malloc(64 * 1024);
	void *fast, *cache;
	size_t fb, cb;
	int rc, step, i;
	double worst = 0.0;

	printf("streaming: SD-style tiled reads must match the mapped path\n");
	tq_store_init_stream(&st, (uint64_t)g_len, mem_read, &ctx, tile, 32 * 1024);
	rc = tq_model_open(&m, &st, arena, 64 * 1024);
	CHECK(rc == TQ_OK, "open streamed: %s", tq_strerror(rc));
	if (rc != TQ_OK) {
		return;
	}

	fb = tq_runtime_bytes(&m, GOLDEN_STEPS);
	cb = tq_kv_bytes(&m, GOLDEN_STEPS, TQ_KV_F32);
	fast = malloc(fb);
	cache = malloc(cb);
	rc = tq_runtime_init(&rt, &m, GOLDEN_STEPS, TQ_KV_F32, fast, fb, cache, cb);
	CHECK(rc == TQ_OK, "runtime streamed: %s", tq_strerror(rc));
	if (rc != TQ_OK) {
		return;
	}

	for (step = 0; step < GOLDEN_STEPS; step++) {
		rc = tq_forward(&rt, golden_tokens[step], step);
		CHECK(rc == TQ_OK, "streamed step %d: %s", step, tq_strerror(rc));
		if (rc != TQ_OK) {
			return;
		}
		for (i = 0; i < GOLDEN_VOCAB; i++) {
			double d = fabs((double)rt.logits[i] -
					(double)g_golden[step][i]);
			double rel = d / (fabs((double)g_golden[step][i]) + 1e-3);

			if (rel > worst) {
				worst = rel;
			}
		}
	}
	CHECK(worst < 2e-3, "streamed worst relative error %.3g", worst);
	CHECK(ctx.reads > 0, "streaming backend was never called");
	printf("  %llu tiled reads, worst relative error %.3g\n",
	       (unsigned long long)ctx.reads, worst);

	free(tile);
	free(arena);
	free(fast);
	free(cache);
}

/* ------------------------------------------------------------- tokenizer */

static void test_tokenizer(TqModel *m)
{
	TqTokenizer t;
	char buf[64], out[64];
	int32_t toks[128];
	int rc, n, i;
	static const char *cases[] = {
		"a", "the", " the", "hello", "a b c", "in on to of",
		"tokenizer round trip", "z", "",
	};
	size_t ci;

	printf("tokenizer: encode/decode round trip\n");
	rc = tq_tokenizer_init(&t, m, buf, sizeof(buf));
	CHECK(rc == TQ_OK, "tokenizer init: %s", tq_strerror(rc));
	if (rc != TQ_OK) {
		return;
	}

	for (ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
		char joined[256];
		size_t used = 0;

		n = tq_encode(&t, cases[ci], 0, 0, toks, 128);
		CHECK(n >= 0, "encode %s: %s", cases[ci], tq_strerror(n));
		if (n < 0) {
			continue;
		}
		joined[0] = '\0';
		for (i = 0; i < n; i++) {
			int len = tq_decode(&t, -1, (int)toks[i], out, sizeof(out));

			if (len > 0 && used + (size_t)len < sizeof(joined)) {
				memcpy(joined + used, out, (size_t)len);
				used += (size_t)len;
			}
		}
		joined[used] = '\0';

		/* Encoding prepends the SentencePiece space, so the round trip
		 * reproduces the input with one leading space. */
		if (cases[ci][0] == '\0') {
			CHECK(used == 0, "empty input produced %s", joined);
		} else {
			CHECK(joined[0] == ' ' && !strcmp(joined + 1, cases[ci]),
			      "round trip %s -> %s (%d tokens)", cases[ci], joined, n);
		}
	}

	/* Byte fallback: a character outside the toy merge set must still
	 * survive as <0xNN> tokens. */
	n = tq_encode(&t, "\xC3\xA9", 0, 0, toks, 128);
	CHECK(n >= 2, "utf-8 byte fallback produced %d tokens", n);

	/* BOS/EOS bracketing. */
	n = tq_encode(&t, "the", 1, 1, toks, 128);
	CHECK(n >= 3 && toks[0] == (int32_t)m->hdr.bos_token &&
	      toks[n - 1] == (int32_t)m->hdr.eos_token,
	      "bos/eos bracketing: n=%d first=%d last=%d", n,
	      n ? (int)toks[0] : -1, n ? (int)toks[n - 1] : -1);
}

/* ---------------------------------------------------------------- sampler */

static void test_sampler(int vocab)
{
	TqSampler s;
	float *logits = malloc(sizeof(float) * (size_t)vocab);
	int32_t *idx = malloc(sizeof(int32_t) * (size_t)vocab);
	float *prob = malloc(sizeof(float) * (size_t)vocab);
	int i, k;

	printf("sampler: greedy, temperature, nucleus\n");

	for (i = 0; i < vocab; i++) {
		logits[i] = -10.0f;
	}
	logits[vocab / 3] = 5.0f;

	tq_sampler_init(&s, vocab, 0.0f, 0.0f, 1, idx, prob);
	CHECK(tq_sample(&s, logits) == vocab / 3, "greedy missed the argmax");

	/* With a dominant logit and tight nucleus, every draw is that token. */
	for (k = 0; k < 32; k++) {
		for (i = 0; i < vocab; i++) {
			logits[i] = -10.0f;
		}
		logits[7] = 20.0f;
		tq_sampler_init(&s, vocab, 1.0f, 0.9f, (uint64_t)k + 1, idx, prob);
		CHECK(tq_sample(&s, logits) == 7, "nucleus draw %d strayed", k);
	}

	/* Same seed must give the same sequence — the host/device comparison
	 * depends on it. */
	{
		int a[8], b[8];
		float *save = malloc(sizeof(float) * (size_t)vocab);

		for (k = 0; k < 8; k++) {
			for (i = 0; i < vocab; i++) {
				save[i] = rndf() * 3.0f;
			}
			/* tq_sample softmaxes in place and destroys its input,
			 * so each run needs a fresh copy. */
			memcpy(logits, save, sizeof(float) * (size_t)vocab);
			tq_sampler_init(&s, vocab, 1.0f, 0.95f, 99, idx, prob);
			a[k] = tq_sample(&s, logits);
			memcpy(logits, save, sizeof(float) * (size_t)vocab);
			tq_sampler_init(&s, vocab, 1.0f, 0.95f, 99, idx, prob);
			b[k] = tq_sample(&s, logits);
		}
		free(save);
		for (k = 0; k < 8; k++) {
			CHECK(a[k] == b[k], "seed 99 not reproducible at %d", k);
		}
	}

	free(logits);
	free(idx);
	free(prob);
}

/* ------------------------------------------------------- arithmetic coding */

static uint64_t ac_rng = 0x9E3779B97F4A7C15ull;

static uint32_t ac_rand(void)
{
	ac_rng ^= ac_rng >> 12;
	ac_rng ^= ac_rng << 25;
	ac_rng ^= ac_rng >> 27;
	return (uint32_t)((ac_rng * 0x2545F4914F6CDD1Dull) >> 32);
}

/*
 * Drive the raw coder with an explicit count table, no model involved. This is
 * where carry-propagation bugs surface as something diagnosable: a round trip
 * that fails here is arithmetic, while one that only fails on real logits is a
 * probability-table disagreement.
 */
static void ac_roundtrip(const char *label, const uint32_t *cnt, int n, int nsym)
{
	uint8_t  *out = malloc(4u * (size_t)nsym + 64u);
	int      *sym = malloc(sizeof(int) * (size_t)nsym);
	TqAcEnc   e;
	TqAcDec   d;
	uint32_t  tot = 0, cum;
	size_t    len;
	int       i, k, bad = -1;

	for (i = 0; i < n; i++) {
		tot += cnt[i];
	}

	/* Sample from the table itself, so the stream really is typical of the
	 * distribution being coded against and the length check below means
	 * something. */
	for (i = 0; i < nsym; i++) {
		uint32_t t = ac_rand() % tot;
		uint32_t c = 0;

		for (k = 0; k < n; k++) {
			if (t < c + cnt[k]) {
				break;
			}
			c += cnt[k];
		}
		sym[i] = k;
	}

	tq_ac_enc_init(&e, out, 4u * (size_t)nsym + 64u);
	for (i = 0; i < nsym; i++) {
		cum = 0;
		for (k = 0; k < sym[i]; k++) {
			cum += cnt[k];
		}
		tq_ac_enc_sym(&e, cum, cnt[sym[i]], tot);
	}
	len = tq_ac_enc_finish(&e);
	CHECK(!e.overflow, "%s: encoder overflowed", label);
	CHECK(len == tq_ac_enc_finish(&e), "%s: finish() not idempotent", label);

	tq_ac_dec_init(&d, out, len);
	for (i = 0; i < nsym; i++) {
		uint32_t t = tq_ac_dec_target(&d, tot);
		uint32_t c = 0;

		for (k = 0; k < n; k++) {
			if (t < c + cnt[k]) {
				break;
			}
			c += cnt[k];
		}
		tq_ac_dec_update(&d, c, cnt[k]);
		if (k != sym[i] && bad < 0) {
			bad = i;
		}
	}
	CHECK(bad < 0, "%s: first mismatch at symbol %d", label, bad);

	/* Within 1% of entropy plus the 5-byte flush. A coder that round-trips but
	 * codes to the wrong length is using a different distribution than it
	 * thinks. */
	{
		double bits = 0.0;

		for (i = 0; i < nsym; i++) {
			bits -= log2((double)cnt[sym[i]] / (double)tot);
		}
		CHECK((double)len * 8.0 < bits * 1.01 + 48.0,
		      "%s: %zu bytes vs %.1f bits of entropy", label, len, bits);
		printf("  %-18s %5zu bytes for %.0f bits of entropy\n",
		       label, len, bits);
	}

	free(out);
	free(sym);
}

static void test_arith(TqModel *m)
{
	uint32_t cnt[64];
	int      i;

	printf("arithmetic coder: fixed-point exp2\n");

	CHECK(tq_ac_exp2_q24(0) == (1u << 24), "exp2(0) = %u", tq_ac_exp2_q24(0));
	CHECK(tq_ac_exp2_q24(-(1 << 16)) == (1u << 23), "exp2(-1) wrong");
	CHECK(tq_ac_exp2_q24(-(4 << 16)) == (1u << 20), "exp2(-4) wrong");
	CHECK(tq_ac_exp2_q24(-(64 << 16)) == 0u, "exp2(-64) should underflow to 0");
	/*
	 * Two different properties, measured where each one means something.
	 *
	 * Relative accuracy belongs to the polynomial, so it is checked over the
	 * polynomial's own domain [-1, 0] where the result has not been shifted.
	 * Further down, the Q24 LSB *is* the value — at 2^-24 the answer is a
	 * single count — so no implementation could hold a relative bound there,
	 * and absolute error is the honest measure. Both bounds are far finer
	 * than the 16-bit counts downstream can express, which is what actually
	 * matters.
	 */
	{
		double worst_rel = 0.0, worst_abs = 0.0;

		for (i = 0; i >= -(1 << 16); i -= 7) {
			double want = pow(2.0, (double)i / 65536.0) * (double)(1u << 24);
			double rel  = fabs((double)tq_ac_exp2_q24(i) - want) / want;

			if (rel > worst_rel) {
				worst_rel = rel;
			}
		}
		for (i = 0; i > -24 * 65536; i -= 997) {
			double want = pow(2.0, (double)i / 65536.0) * (double)(1u << 24);
			double err  = fabs((double)tq_ac_exp2_q24(i) - want);

			if (err > worst_abs) {
				worst_abs = err;
			}
		}
		CHECK(worst_rel < 1e-6, "exp2 worst relative error %.3g", worst_rel);
		CHECK(worst_abs < 8.0, "exp2 worst absolute error %.3g LSB", worst_abs);
		printf("  worst error vs pow(): %.2e relative, %.2f LSB absolute\n",
		       worst_rel, worst_abs);
	}
	{
		/* Monotonicity across a whole octave. A polynomial that dips would
		 * make a more probable symbol cheaper to code than a less probable
		 * one — self-consistent, so the round trip would still pass, and the
		 * compression would silently be worse than the model warrants. */
		uint32_t prev = tq_ac_exp2_q24(-(1 << 16));
		int      mono = 1;

		for (i = -(1 << 16) + 1; i <= 0; i++) {
			uint32_t v = tq_ac_exp2_q24(i);

			if (v < prev) {
				mono = 0;
			}
			prev = v;
		}
		CHECK(mono, "exp2 is not monotone over [-1, 0]");
	}

	printf("arithmetic coder: round trip against explicit tables\n");

	/* Near-uniform: the coder's worst case for output size. */
	for (i = 0; i < 16; i++) {
		cnt[i] = TQ_AC_TOT / 16u;
	}
	ac_roundtrip("uniform/16", cnt, 16, 4000);

	/*
	 * One symbol at 65520/65536. Long runs of a near-certain symbol are what
	 * drive `low` into extended 0xFF territory, which is the only path that
	 * exercises deferred carry propagation.
	 */
	cnt[0] = TQ_AC_TOT - 15u;
	for (i = 1; i < 16; i++) {
		cnt[i] = 1u;
	}
	ac_roundtrip("skewed/16", cnt, 16, 20000);

	/*
	 * Ragged counts with no power-of-two structure anywhere. The 900 bound is
	 * load-bearing: 63 symbols cannot then exceed TQ_AC_TOT, and a total above
	 * 2^16 silently breaks the coder's precision rather than failing loudly.
	 */
	{
		uint32_t used = 0;

		for (i = 0; i < 63; i++) {
			cnt[i] = 1u + (ac_rand() % 900u);
			used += cnt[i];
		}
		CHECK(used < TQ_AC_TOT, "ragged fixture overran the total: %u", used);
		cnt[63] = TQ_AC_TOT - used;
		ac_roundtrip("ragged/64", cnt, 64, 4000);
	}

	/* --------------------------------------------- against real logits --- */

	printf("arithmetic coder: round trip against model logits\n");
	{
		TqRuntime rt;
		TqAcModel pm;
		TqAcEnc   e;
		TqAcDec   d;
		size_t    fb = tq_runtime_bytes(m, GOLDEN_STEPS);
		size_t    cb = tq_kv_bytes(m, GOLDEN_STEPS, TQ_KV_F32);
		void     *fast = malloc(fb);
		void     *cache = malloc(cb);
		uint8_t   out[256];
		float    *saved = malloc(sizeof(float) * (size_t)m->vocab_size *
					 GOLDEN_STEPS);
		size_t    len;
		int       rc, step, bad = -1;
		double    bits = 0.0;

		rc = tq_runtime_init(&rt, m, GOLDEN_STEPS, TQ_KV_F32, fast, fb,
				     cache, cb);
		CHECK(rc == TQ_OK, "ac runtime: %s", tq_strerror(rc));
		CHECK(tq_ac_model_init(&pm, m->vocab_size) == TQ_OK, "ac model init");

		/*
		 * Encode golden_tokens[step+1] under the distribution the model
		 * predicts after golden_tokens[step] — exactly the causal ordering a
		 * real message uses. The logits are stashed because the decoder must
		 * see bit-identical values, which on one machine it does by
		 * construction; the cross-architecture case is what P1 is for.
		 */
		for (step = 0; rc == TQ_OK && step < GOLDEN_STEPS; step++) {
			rc = tq_forward(&rt, golden_tokens[step], step);
			CHECK(rc == TQ_OK, "ac forward %d: %s", step, tq_strerror(rc));
			memcpy(saved + (size_t)step * m->vocab_size, rt.logits,
			       sizeof(float) * (size_t)m->vocab_size);
		}

		tq_ac_enc_init(&e, out, sizeof(out));
		for (step = 0; step + 1 < GOLDEN_STEPS; step++) {
			const float *lg = saved + (size_t)step * m->vocab_size;
			int          tk = golden_tokens[step + 1];

			CHECK(tq_ac_enc_token(&e, &pm, lg, tk) == TQ_OK,
			      "encode token %d", tk);
			/* Both levels, since the pair is what actually goes on
			 * the wire and their product is P(token). */
			bits += -log2((double)pm.bcnt[tk / TQ_AC_BUCKET] /
				      (double)TQ_AC_TOT)
				- log2((double)pm.scnt[tk % TQ_AC_BUCKET] /
				       (double)TQ_AC_TOT);
		}
		len = tq_ac_enc_finish(&e);
		CHECK(!e.overflow, "model encode overflowed");
		CHECK((double)len * 8.0 < bits * 1.01 + 48.0,
		      "%zu bytes vs %.1f bits the table charges", len, bits);

		tq_ac_dec_init(&d, out, len);
		for (step = 0; step + 1 < GOLDEN_STEPS; step++) {
			const float *lg = saved + (size_t)step * m->vocab_size;
			int          got = tq_ac_dec_token(&d, &pm, lg);

			if (got != golden_tokens[step + 1] && bad < 0) {
				bad = step;
			}
		}
		CHECK(bad < 0, "model round trip diverged at step %d", bad);
		printf("  %d tokens coded in %zu bytes\n", GOLDEN_STEPS - 1, len);

		/* Counts must always tile the range exactly and leave nothing
		 * unencodable, or some token becomes impossible to transmit. */
		{
			uint32_t sum = 0;
			int      lo = 1;

			for (i = 0; i < pm.nbuckets; i++) {
				sum += pm.bcnt[i];
				if (pm.bcnt[i] < 1u) {
					lo = 0;
				}
			}
			CHECK(sum == TQ_AC_TOT, "bucket counts sum to %u", sum);
			CHECK(lo, "a bucket had a zero count");
		}

		free(fast);
		free(cache);
		free(saved);
	}
}

/* -------------------------------------------------------------- rejection */

static void test_rejects_garbage(void)
{
	TqStore st;
	TqModel m;
	void *arena = malloc(64 * 1024);
	uint8_t *bad = malloc(g_len);

	printf("robustness: malformed files are rejected, not executed\n");

	memcpy(bad, g_buf, g_len);
	((uint32_t *)bad)[0] = 0xDEADBEEFu;
	tq_store_init_mapped(&st, bad, (uint64_t)g_len);
	CHECK(tq_model_open(&m, &st, arena, 64 * 1024) == TQ_ERR_MAGIC,
	      "bad magic accepted");

	memcpy(bad, g_buf, g_len);
	((uint32_t *)bad)[1] = 99u;
	tq_store_init_mapped(&st, bad, (uint64_t)g_len);
	CHECK(tq_model_open(&m, &st, arena, 64 * 1024) == TQ_ERR_VERSION,
	      "bad version accepted");

	/* Truncated file: the declared size exceeds what the store holds. */
	tq_store_init_mapped(&st, g_buf, (uint64_t)(g_len / 2));
	CHECK(tq_model_open(&m, &st, arena, 64 * 1024) == TQ_ERR_TRUNCATED,
	      "truncated file accepted");

	/* Arena too small for the layer table. */
	tq_store_init_mapped(&st, g_buf, (uint64_t)g_len);
	CHECK(tq_model_open(&m, &st, arena, 8) == TQ_ERR_NOMEM,
	      "undersized arena accepted");

	free(bad);
	free(arena);
}

/* -------------------------------------------------------------------- main */

int main(int argc, char **argv)
{
	const char *path = (argc > 1) ? argv[1] : "build/test_q4.etq";
	TqStore st;
	TqModel m;
	TqRuntime rt;
	void *arena, *fast, *cache;
	size_t fb, cb;
	int rc;

	printf("libtq test suite  [kernel backend: %s]\n\n", tq_kernel_backend());

	test_half();
	test_kernels();

	if (load_file(path) != 0) {
		printf("\nFATAL: cannot read %s\n"
		       "  build it with: make -C host fixtures\n", path);
		return 1;
	}

	tq_store_init_mapped(&st, g_buf, (uint64_t)g_len);
	arena = malloc(64 * 1024);
	rc = tq_model_open(&m, &st, arena, 64 * 1024);
	if (rc != TQ_OK) {
		printf("\nFATAL: open %s: %s\n", path, tq_strerror(rc));
		return 1;
	}

	test_format(&m);

	g_golden = (m.qtype == TQ_DT_Q8_0) ? golden_logits_q8 : golden_logits;

	fb = tq_runtime_bytes(&m, GOLDEN_STEPS);
	cb = tq_kv_bytes(&m, GOLDEN_STEPS, TQ_KV_F32);
	fast = malloc(fb);
	cache = malloc(cb);
	rc = tq_runtime_init(&rt, &m, GOLDEN_STEPS, TQ_KV_F32, fast, fb, cache, cb);
	if (rc != TQ_OK) {
		printf("\nFATAL: runtime: %s\n", tq_strerror(rc));
		return 1;
	}
	run_golden(&rt, "fp32 kv");

	/* int8 KV is an approximation, so it gets its own looser gate below
	 * rather than being held to the golden tolerance. */
	{
		TqRuntime rt8;
		size_t cb8 = tq_kv_bytes(&m, GOLDEN_STEPS, TQ_KV_Q8);
		void *cache8 = malloc(cb8);
		double worst = 0.0;
		int step, i;

		/* Scored against the RMS of the step's logits, not per element.
		 * A per-element relative error is meaningless where the golden
		 * logit is near zero — it reports 300% for a 0.01 absolute
		 * difference on a distribution whose spread is 0.4. */
		printf("int8 KV cache: quality vs fp32 golden\n");
		rc = tq_runtime_init(&rt8, &m, GOLDEN_STEPS, TQ_KV_Q8,
				     fast, fb, cache8, cb8);
		CHECK(rc == TQ_OK, "kv8 runtime: %s", tq_strerror(rc));
		for (step = 0; rc == TQ_OK && step < GOLDEN_STEPS; step++) {
			rc = tq_forward(&rt8, golden_tokens[step], step);
			CHECK(rc == TQ_OK, "kv8 step %d: %s", step, tq_strerror(rc));
			{
				double ss = 0.0, rms;

				for (i = 0; i < GOLDEN_VOCAB; i++) {
					double g = (double)g_golden[step][i];

					ss += g * g;
				}
				rms = sqrt(ss / GOLDEN_VOCAB) + 1e-9;
				for (i = 0; rc == TQ_OK && i < GOLDEN_VOCAB; i++) {
					double d = fabs((double)rt8.logits[i] -
							(double)g_golden[step][i]);

					if (d / rms > worst) {
						worst = d / rms;
					}
				}
			}
		}
		CHECK(worst < 0.10, "int8 kv error %.3g of logit RMS", worst);
		printf("  worst error: %.2f%% of logit RMS\n", worst * 100.0);
		free(cache8);
	}

	test_tokenizer(&m);
	test_sampler(m.vocab_size);
	test_arith(&m);
	test_streaming();
	test_rejects_garbage();

	printf("\n%d checks, %d failures\n", g_checks, g_fail);
	return g_fail ? 1 : 0;
}

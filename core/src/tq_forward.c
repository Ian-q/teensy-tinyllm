/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * The llama2 decode step: RMSNorm, RoPE, grouped-query attention, SwiGLU.
 *
 * Activations stay in fp32 — the Cortex-M7 has a real single-precision FPU
 * and activations are tiny next to the weights, so there is nothing to gain
 * from quantizing them for storage. They ARE quantized to Q8_0 just before
 * each matmul, because that is what lets the weight side stay int8/int4 and
 * feed SMLAD.
 *
 * One activation vector is quantized once and reused across every matrix that
 * consumes it: xq feeds Wq/Wk/Wv (three matrices, one input) and again
 * W1/W3. That is free accuracy and free time.
 */

#include <math.h>   /* sqrtf only; see tq_math.h */

#include "tq/tq_math.h"
#include <string.h>

#include "tq_internal.h"

/* ------------------------------------------------------------- primitives */

static void tq_rmsnorm(float *out, const float *x, const float *w, int n, float eps)
{
	float ss = 0.0f;
	int i;

	for (i = 0; i < n; i++) {
		ss += x[i] * x[i];
	}
	ss = 1.0f / sqrtf(ss / (float)n + eps);
	for (i = 0; i < n; i++) {
		out[i] = w[i] * (ss * x[i]);
	}
}

static void tq_softmax(float *x, int n)
{
	float max = x[0];
	float sum = 0.0f;
	int i;

	for (i = 1; i < n; i++) {
		if (x[i] > max) {
			max = x[i];
		}
	}
	for (i = 0; i < n; i++) {
		x[i] = tq_expf(x[i] - max);
		sum += x[i];
	}
	sum = 1.0f / sum;
	for (i = 0; i < n; i++) {
		x[i] *= sum;
	}
}

/* --------------------------------------------------------------- KV cache */

/* Write one kv_dim vector into the cache at (layer, pos). */
static void tq_kv_store(TqRuntime *rt, int layer, int pos, const float *k, const float *v)
{
	const TqModel *m = rt->m;
	size_t idx = ((size_t)layer * (size_t)rt->max_seq + (size_t)pos) * (size_t)m->kv_dim;

	if (rt->kv_dtype == TQ_KV_F32) {
		memcpy((float *)rt->k_cache + idx, k, (size_t)m->kv_dim * sizeof(float));
		memcpy((float *)rt->v_cache + idx, v, (size_t)m->kv_dim * sizeof(float));
		return;
	}

	{
		/* int8 with one scale per head. Per-head rather than per-vector
		 * because attention heads routinely differ by an order of
		 * magnitude in scale, and a shared scale would crush the quiet
		 * ones into the noise floor. */
		int8_t *kq = (int8_t *)rt->k_cache + idx;
		int8_t *vq = (int8_t *)rt->v_cache + idx;
		size_t sidx = ((size_t)layer * (size_t)rt->max_seq + (size_t)pos) *
			      (size_t)m->n_kv_heads;
		int h;

		for (h = 0; h < m->n_kv_heads; h++) {
			int off = h * m->head_size;
			float ka = 0.0f, va = 0.0f, ks, vs;
			int i;

			for (i = 0; i < m->head_size; i++) {
				float a = k[off + i] < 0.0f ? -k[off + i] : k[off + i];
				float b = v[off + i] < 0.0f ? -v[off + i] : v[off + i];

				if (a > ka) {
					ka = a;
				}
				if (b > va) {
					va = b;
				}
			}
			ks = ka / 127.0f;
			vs = va / 127.0f;
			rt->k_scale[sidx + (size_t)h] = ks;
			rt->v_scale[sidx + (size_t)h] = vs;
			{
				float ki = (ks != 0.0f) ? 1.0f / ks : 0.0f;
				float vi = (vs != 0.0f) ? 1.0f / vs : 0.0f;

				for (i = 0; i < m->head_size; i++) {
					float a = k[off + i] * ki;
					float b = v[off + i] * vi;
					int qa = (int)(a < 0.0f ? a - 0.5f : a + 0.5f);
					int qb = (int)(b < 0.0f ? b - 0.5f : b + 0.5f);

					kq[off + i] = (int8_t)(qa > 127 ? 127 :
							       (qa < -127 ? -127 : qa));
					vq[off + i] = (int8_t)(qb > 127 ? 127 :
							       (qb < -127 ? -127 : qb));
				}
			}
		}
	}
}

/* q . K[layer][t][kvh] */
static float tq_kv_dot_k(const TqRuntime *rt, int layer, int t, int kvh, const float *q)
{
	const TqModel *m = rt->m;
	size_t idx = ((size_t)layer * (size_t)rt->max_seq + (size_t)t) *
		     (size_t)m->kv_dim + (size_t)kvh * (size_t)m->head_size;
	float s = 0.0f;
	int i;

	if (rt->kv_dtype == TQ_KV_F32) {
		const float *k = (const float *)rt->k_cache + idx;

		for (i = 0; i < m->head_size; i++) {
			s += q[i] * k[i];
		}
		return s;
	}
	{
		const int8_t *k = (const int8_t *)rt->k_cache + idx;
		size_t sidx = ((size_t)layer * (size_t)rt->max_seq + (size_t)t) *
			      (size_t)m->n_kv_heads + (size_t)kvh;

		for (i = 0; i < m->head_size; i++) {
			s += q[i] * (float)k[i];
		}
		return s * rt->k_scale[sidx];
	}
}

/* out[0..head_size) += a * V[layer][t][kvh] */
static void tq_kv_acc_v(const TqRuntime *rt, int layer, int t, int kvh, float a, float *out)
{
	const TqModel *m = rt->m;
	size_t idx = ((size_t)layer * (size_t)rt->max_seq + (size_t)t) *
		     (size_t)m->kv_dim + (size_t)kvh * (size_t)m->head_size;
	int i;

	if (rt->kv_dtype == TQ_KV_F32) {
		const float *v = (const float *)rt->v_cache + idx;

		for (i = 0; i < m->head_size; i++) {
			out[i] += a * v[i];
		}
		return;
	}
	{
		const int8_t *v = (const int8_t *)rt->v_cache + idx;
		size_t sidx = ((size_t)layer * (size_t)rt->max_seq + (size_t)t) *
			      (size_t)m->n_kv_heads + (size_t)kvh;
		float c = a * rt->v_scale[sidx];

		for (i = 0; i < m->head_size; i++) {
			out[i] += c * (float)v[i];
		}
	}
}

/* ------------------------------------------------------------------ RoPE */

static void tq_rope_prepare(TqRuntime *rt, int pos)
{
	const TqModel *m = rt->m;
	int half = m->head_size / 2;
	int i;

	for (i = 0; i < half; i++) {
		float freq = 1.0f / tq_powf(m->hdr.rope_theta,
					 (float)(2 * i) / (float)m->head_size);
		float ang = (float)pos * freq;

		rt->rope_cos[i] = tq_cosf(ang);
		rt->rope_sin[i] = tq_sinf(ang);
	}
}

/* Rotate `n` values (a whole number of heads) in place. */
static void tq_rope_apply(const TqRuntime *rt, float *v, int n)
{
	int head_size = rt->m->head_size;
	int half = head_size / 2;
	int base;

	for (base = 0; base < n; base += head_size) {
		int i;

		/* Interleaved pair convention, matching llama2.c: element 2i is
		 * rotated against element 2i+1 within each head. HF checkpoints
		 * use the split-half convention instead; tools/etq/convert.py
		 * permutes wq/wk at export time so this stays the only layout
		 * the device ever sees. */
		for (i = 0; i < half; i++) {
			float c = rt->rope_cos[i];
			float s = rt->rope_sin[i];
			float v0 = v[base + 2 * i];
			float v1 = v[base + 2 * i + 1];

			v[base + 2 * i]     = v0 * c - v1 * s;
			v[base + 2 * i + 1] = v0 * s + v1 * c;
		}
	}
}

/* --------------------------------------------------------------- forward */

int tq_forward(TqRuntime *rt, int token, int pos)
{
	const TqModel *m = rt->m;
	const float *norm_w;
	int dim = m->dim;
	int kv_dim = m->kv_dim;
	int head_size = m->head_size;
	float inv_sqrt_hs;
	int l, rc;

	if (token < 0 || token >= m->vocab_size) {
		return TQ_ERR_RANGE;
	}
	if (pos < 0 || pos >= rt->max_seq) {
		return TQ_ERR_RANGE;
	}

	inv_sqrt_hs = 1.0f / sqrtf((float)head_size);

	/* Token embedding: one row out of a matrix that may be 16 MB. This is
	 * the only truly random read in the whole step. */
	rc = tq_row_to_float(m, m->tok_emb, m->qtype, dim, token, rt->x);
	if (rc != TQ_OK) {
		return rc;
	}

	tq_rope_prepare(rt, pos);

	for (l = 0; l < m->n_layers; l++) {
		const TqLayerOffsets *lo = &m->layers[l];
		int h;

		norm_w = (const float *)tq_view(m, lo->rms_att,
						(uint32_t)dim * sizeof(float));
		if (norm_w == NULL) {
			return TQ_ERR_IO;
		}
		tq_rmsnorm(rt->xb, rt->x, norm_w, dim, m->hdr.norm_eps);

		/* One quantization feeds three matmuls. */
		tq_quantize_q8(rt->xb, rt->xq, dim);

		rc = tq_matvec(rt, rt->q,  lo->wq, m->qtype, dim, dim,    rt->xq);
		if (rc != TQ_OK) {
			return rc;
		}
		rc = tq_matvec(rt, rt->kb, lo->wk, m->qtype, dim, kv_dim, rt->xq);
		if (rc != TQ_OK) {
			return rc;
		}
		rc = tq_matvec(rt, rt->vb, lo->wv, m->qtype, dim, kv_dim, rt->xq);
		if (rc != TQ_OK) {
			return rc;
		}

		tq_rope_apply(rt, rt->q,  dim);
		tq_rope_apply(rt, rt->kb, kv_dim);
		tq_kv_store(rt, l, pos, rt->kb, rt->vb);

		/* Grouped-query attention. */
		for (h = 0; h < m->n_heads; h++) {
			const float *qh = rt->q + h * head_size;
			float *att = rt->att + (size_t)h * (size_t)rt->max_seq;
			float *out = rt->xb + h * head_size;
			int kvh = h / m->kv_mul;
			int t;

			for (t = 0; t <= pos; t++) {
				att[t] = tq_kv_dot_k(rt, l, t, kvh, qh) * inv_sqrt_hs;
			}
			tq_softmax(att, pos + 1);

			memset(out, 0, (size_t)head_size * sizeof(float));
			for (t = 0; t <= pos; t++) {
				tq_kv_acc_v(rt, l, t, kvh, att[t], out);
			}
		}

		tq_quantize_q8(rt->xb, rt->xq, dim);
		rc = tq_matvec(rt, rt->xb2, lo->wo, m->qtype, dim, dim, rt->xq);
		if (rc != TQ_OK) {
			return rc;
		}
		{
			int i;

			for (i = 0; i < dim; i++) {
				rt->x[i] += rt->xb2[i];
			}
		}

		/* SwiGLU feed-forward: w2( silu(w1 x) * w3 x ). */
		norm_w = (const float *)tq_view(m, lo->rms_ffn,
						(uint32_t)dim * sizeof(float));
		if (norm_w == NULL) {
			return TQ_ERR_IO;
		}
		tq_rmsnorm(rt->xb, rt->x, norm_w, dim, m->hdr.norm_eps);
		tq_quantize_q8(rt->xb, rt->xq, dim);

		rc = tq_matvec(rt, rt->hb,  lo->w1, m->qtype, dim, m->hidden_dim, rt->xq);
		if (rc != TQ_OK) {
			return rc;
		}
		rc = tq_matvec(rt, rt->hb3, lo->w3, m->qtype, dim, m->hidden_dim, rt->xq);
		if (rc != TQ_OK) {
			return rc;
		}
		{
			int i;

			for (i = 0; i < m->hidden_dim; i++) {
				float v = rt->hb[i];

				rt->hb[i] = (v / (1.0f + tq_expf(-v))) * rt->hb3[i];
			}
		}
		tq_quantize_q8(rt->hb, rt->hq, m->hidden_dim);
		rc = tq_matvec(rt, rt->xb2, lo->w2, m->qtype, m->hidden_dim, dim, rt->hq);
		if (rc != TQ_OK) {
			return rc;
		}
		{
			int i;

			for (i = 0; i < dim; i++) {
				rt->x[i] += rt->xb2[i];
			}
		}
	}

	norm_w = (const float *)tq_view(m, m->rms_final, (uint32_t)dim * sizeof(float));
	if (norm_w == NULL) {
		return TQ_ERR_IO;
	}
	tq_rmsnorm(rt->x, rt->x, norm_w, dim, m->hdr.norm_eps);
	tq_quantize_q8(rt->x, rt->xq, dim);

	/*
	 * The classifier is the single most expensive matmul in the model: for
	 * stories15M it is 9.2 MB of the 15 MB read per token, because vocab
	 * (32000) dwarfs everything else. It is also where a smaller tokenizer
	 * would pay off most — see docs/PERFORMANCE.md.
	 */
	return tq_matvec(rt, rt->logits, m->classifier, m->qtype, dim,
			 m->vocab_size, rt->xq);
}

/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * .etq parsing: header validation, tensor directory walk, arena layout.
 *
 * Nothing here allocates. tq_model_open carves the per-layer offset table out
 * of a caller arena; tq_runtime_init carves activations out of a second one.
 * The firmware uses that to put activations in DTCM and the KV cache in PSRAM,
 * which is worth roughly an order of magnitude on the attention inner loop.
 */

#include <string.h>

#include "tq_internal.h"

/* ------------------------------------------------------------------ utils */

static size_t tq_align_up(size_t v, size_t a)
{
	return (v + a - 1u) & ~(a - 1u);
}

/*
 * Carve `bytes` out of an arena at a 16-byte-aligned ADDRESS. Returns NULL
 * when full.
 *
 * Aligning the offset is not enough: callers hand us plain `uint8_t[]`
 * arrays, and a 16-byte offset from a 2-byte-aligned base is still 2-byte
 * aligned. The structures carved here hold uint64_t, which the M7 loads
 * with LDRD — that faults on a misaligned address. (A host caller never
 * saw this because malloc is already 16-byte aligned.)
 */
static void *tq_carve(uint8_t *base, size_t cap, size_t *used, size_t bytes)
{
	size_t skew = (size_t)((uintptr_t)base & 15u);
	size_t off = tq_align_up(*used + skew, 16u) - skew;

	if (off + bytes > cap) {
		return NULL;
	}
	*used = off + bytes;
	return base + off;
}

/* Compare a fixed-width, possibly unterminated directory name. */
static int tq_name_eq(const char *field, const char *want)
{
	size_t i;

	for (i = 0; i < TQ_NAME_BYTES; i++) {
		char c = field[i];

		if (c != want[i]) {
			return 0;
		}
		if (c == '\0') {
			return 1;
		}
	}
	/* All TQ_NAME_BYTES matched with no terminator on either side. */
	return 1;
}

/* ------------------------------------------------------------------- open */

/* One resolved directory entry we care about. */
typedef struct {
	const char *name;
	uint64_t   *offset;   /* where to store the payload offset */
	uint32_t    dtype;    /* expected dtype                    */
	uint32_t    rows;     /* expected shape[0], 0 = don't care */
	uint32_t    cols;     /* expected shape[1], 0 = 1-D        */
	uint32_t   *nbytes;   /* optional: capture payload size    */
	int         required;
	int         found;
} TqWant;

static int tq_match(TqWant *w, size_t nwant, const TqTensorEntry *e)
{
	size_t i;

	for (i = 0; i < nwant; i++) {
		if (w[i].found || !tq_name_eq(e->name, w[i].name)) {
			continue;
		}
		if (e->dtype != w[i].dtype) {
			return TQ_ERR_SHAPE;
		}
		if (w[i].cols != 0u) {
			if (e->ndim != 2u || e->shape[0] != w[i].rows ||
			    e->shape[1] != w[i].cols) {
				return TQ_ERR_SHAPE;
			}
		} else if (w[i].rows != 0u) {
			if (e->ndim != 1u || e->shape[0] != w[i].rows) {
				return TQ_ERR_SHAPE;
			}
		}
		*w[i].offset = e->offset;
		if (w[i].nbytes != NULL) {
			*w[i].nbytes = (uint32_t)e->nbytes;
		}
		w[i].found = 1;
		return TQ_OK;
	}
	return TQ_OK;
}

int tq_model_open(TqModel *m, const TqStore *store, void *arena, size_t arena_bytes)
{
	const TqHeader *hdr;
	size_t used = 0;
	size_t nwant = 0;
	uint32_t idx;
	int rc;
	int L;

	/* A directory chunk small enough for any sane tile, big enough to make
	 * streaming reads worthwhile. */
	enum { TQ_DIR_CHUNK = 32 };
	TqTensorEntry dir[TQ_DIR_CHUNK];

	memset(m, 0, sizeof(*m));
	m->store = store;

	hdr = (const TqHeader *)tq_view(m, 0, TQ_HEADER_BYTES);
	if (hdr == NULL) {
		return TQ_ERR_TRUNCATED;
	}
	if (hdr->magic != TQ_MAGIC) {
		return TQ_ERR_MAGIC;
	}
	if (hdr->version != TQ_VERSION || hdr->header_bytes != TQ_HEADER_BYTES) {
		return TQ_ERR_VERSION;
	}
	/* The view may live in the store tile, which the directory walk below
	 * will overwrite, so take a copy now. */
	memcpy(&m->hdr, hdr, sizeof(m->hdr));
	hdr = &m->hdr;

	if (hdr->arch != TQ_ARCH_LLAMA2) {
		return TQ_ERR_UNSUPPORTED;
	}
	if (hdr->qtype != TQ_DT_Q4_0 && hdr->qtype != TQ_DT_Q8_0) {
		return TQ_ERR_UNSUPPORTED;
	}
	if (hdr->group_size != TQ_GROUP_SIZE) {
		return TQ_ERR_UNSUPPORTED;
	}
	if (hdr->file_bytes > store->bytes) {
		return TQ_ERR_TRUNCATED;
	}
	if (hdr->dim <= 0 || hdr->n_layers <= 0 || hdr->n_heads <= 0 ||
	    hdr->n_kv_heads <= 0 || hdr->vocab_size <= 0 || hdr->seq_len <= 0 ||
	    hdr->hidden_dim <= 0) {
		return TQ_ERR_SHAPE;
	}
	if ((hdr->dim % hdr->n_heads) != 0 || (hdr->n_heads % hdr->n_kv_heads) != 0) {
		return TQ_ERR_SHAPE;
	}
	/* Every quantized matmul dimension has to be a whole number of blocks. */
	if ((hdr->dim % TQ_GROUP_SIZE) != 0 || (hdr->hidden_dim % TQ_GROUP_SIZE) != 0) {
		return TQ_ERR_SHAPE;
	}

	m->dim        = hdr->dim;
	m->hidden_dim = hdr->hidden_dim;
	m->n_layers   = hdr->n_layers;
	m->n_heads    = hdr->n_heads;
	m->n_kv_heads = hdr->n_kv_heads;
	m->vocab_size = hdr->vocab_size;
	m->seq_len    = hdr->seq_len;
	m->qtype      = hdr->qtype;
	m->head_size  = hdr->dim / hdr->n_heads;
	m->kv_dim     = m->head_size * hdr->n_kv_heads;
	m->kv_mul     = hdr->n_heads / hdr->n_kv_heads;

	m->layers = (TqLayerOffsets *)tq_carve((uint8_t *)arena, arena_bytes, &used,
					       sizeof(TqLayerOffsets) * (size_t)m->n_layers);
	if (m->layers == NULL) {
		return TQ_ERR_NOMEM;
	}
	memset(m->layers, 0, sizeof(TqLayerOffsets) * (size_t)m->n_layers);

	/*
	 * Build the want-list. Layer tensors dominate, so the list is
	 * 9*n_layers + up to 7 globals. Rather than allocate it, walk the
	 * directory once per chunk and match against a generated name — but
	 * that is O(entries * wants). With <1000 entries and a handful of
	 * string compares each, it costs microseconds and saves an arena.
	 */
	{
		/* Globals first. */
		static const char n_tok_emb[]   = "tok_emb";
		static const char n_rms_final[] = "rms_final";
		static const char n_cls[]       = "cls";
		static const char n_scores[]    = "tok.scores";
		static const char n_offs[]      = "tok.offs";
		static const char n_blob[]      = "tok.blob";
		static const char n_sorted[]    = "tok.sorted";

		/* Bounded: 6 globals + 9 per layer. Allocated on the stack in
		 * chunks so a 32-layer model does not need 300 entries live. */
		TqWant want[7];

		want[0] = (TqWant){ n_tok_emb,   &m->tok_emb,    m->qtype,
				    (uint32_t)m->vocab_size, (uint32_t)m->dim, NULL, 1, 0 };
		want[1] = (TqWant){ n_rms_final, &m->rms_final,  TQ_DT_F32,
				    (uint32_t)m->dim, 0u, NULL, 1, 0 };
		want[2] = (TqWant){ n_cls,       &m->classifier, m->qtype,
				    (uint32_t)m->vocab_size, (uint32_t)m->dim, NULL,
				    hdr->shared_classifier ? 0 : 1, 0 };
		want[3] = (TqWant){ n_scores,    &m->tok_scores, TQ_DT_F32,
				    (uint32_t)m->vocab_size, 0u, NULL, 0, 0 };
		want[4] = (TqWant){ n_offs,      &m->tok_offs,   TQ_DT_U32,
				    (uint32_t)m->vocab_size + 1u, 0u, NULL, 0, 0 };
		want[5] = (TqWant){ n_blob,      &m->tok_blob,   TQ_DT_U8,
				    0u, 0u, &m->tok_blob_bytes, 0, 0 };
		want[6] = (TqWant){ n_sorted,    &m->tok_sorted, TQ_DT_U32,
				    (uint32_t)m->vocab_size, 0u, NULL, 0, 0 };
		nwant = 7;

		for (idx = 0; idx < hdr->tensor_count; ) {
			uint32_t take = hdr->tensor_count - idx;
			const TqTensorEntry *src;
			uint32_t k;

			if (take > TQ_DIR_CHUNK) {
				take = TQ_DIR_CHUNK;
			}
			src = (const TqTensorEntry *)tq_view(
				m, hdr->tensor_table_offset +
				   (uint64_t)idx * sizeof(TqTensorEntry),
				take * (uint32_t)sizeof(TqTensorEntry));
			if (src == NULL) {
				return TQ_ERR_TRUNCATED;
			}
			memcpy(dir, src, take * sizeof(TqTensorEntry));

			for (k = 0; k < take; k++) {
				const TqTensorEntry *e = &dir[k];
				char nm[TQ_NAME_BYTES + 1];
				unsigned layer;

				if (e->offset + e->nbytes > hdr->file_bytes) {
					return TQ_ERR_TRUNCATED;
				}
				if ((e->offset % TQ_ALIGN) != 0u) {
					return TQ_ERR_SHAPE;
				}

				rc = tq_match(want, nwant, e);
				if (rc != TQ_OK) {
					return rc;
				}

				/* Layer tensors: decode "lNNN." then dispatch. */
				if (e->name[0] != 'l' || e->name[4] != '.') {
					continue;
				}
				layer = (unsigned)(e->name[1] - '0') * 100u +
					(unsigned)(e->name[2] - '0') * 10u +
					(unsigned)(e->name[3] - '0');
				if (layer >= (unsigned)m->n_layers) {
					continue;
				}
				memcpy(nm, e->name, TQ_NAME_BYTES);
				nm[TQ_NAME_BYTES] = '\0';

				{
					TqLayerOffsets *lo = &m->layers[layer];
					const char *suf = nm + 5;
					uint64_t *slot = NULL;
					uint32_t want_dt = m->qtype;
					uint32_t r = 0, c = 0;

					if (!strcmp(suf, "rms_att")) {
						slot = &lo->rms_att; want_dt = TQ_DT_F32;
						r = (uint32_t)m->dim;
					} else if (!strcmp(suf, "rms_ffn")) {
						slot = &lo->rms_ffn; want_dt = TQ_DT_F32;
						r = (uint32_t)m->dim;
					} else if (!strcmp(suf, "wq")) {
						slot = &lo->wq; r = (uint32_t)m->dim;
						c = (uint32_t)m->dim;
					} else if (!strcmp(suf, "wk")) {
						slot = &lo->wk; r = (uint32_t)m->kv_dim;
						c = (uint32_t)m->dim;
					} else if (!strcmp(suf, "wv")) {
						slot = &lo->wv; r = (uint32_t)m->kv_dim;
						c = (uint32_t)m->dim;
					} else if (!strcmp(suf, "wo")) {
						slot = &lo->wo; r = (uint32_t)m->dim;
						c = (uint32_t)m->dim;
					} else if (!strcmp(suf, "w1")) {
						slot = &lo->w1; r = (uint32_t)m->hidden_dim;
						c = (uint32_t)m->dim;
					} else if (!strcmp(suf, "w2")) {
						slot = &lo->w2; r = (uint32_t)m->dim;
						c = (uint32_t)m->hidden_dim;
					} else if (!strcmp(suf, "w3")) {
						slot = &lo->w3; r = (uint32_t)m->hidden_dim;
						c = (uint32_t)m->dim;
					}
					if (slot == NULL) {
						continue;
					}
					if (e->dtype != want_dt) {
						return TQ_ERR_SHAPE;
					}
					if (c != 0u) {
						if (e->ndim != 2u || e->shape[0] != r ||
						    e->shape[1] != c) {
							return TQ_ERR_SHAPE;
						}
					} else if (e->ndim != 1u || e->shape[0] != r) {
						return TQ_ERR_SHAPE;
					}
					*slot = e->offset;
				}
			}
			idx += take;
		}

		for (idx = 0; idx < nwant; idx++) {
			if (want[idx].required && !want[idx].found) {
				return TQ_ERR_MISSING;
			}
		}
	}

	if (hdr->shared_classifier) {
		m->classifier = m->tok_emb;
	}

	for (L = 0; L < m->n_layers; L++) {
		const TqLayerOffsets *lo = &m->layers[L];

		/* offset 0 is the header, so it can never be a payload. */
		if (lo->rms_att == 0u || lo->rms_ffn == 0u || lo->wq == 0u ||
		    lo->wk == 0u || lo->wv == 0u || lo->wo == 0u ||
		    lo->w1 == 0u || lo->w2 == 0u || lo->w3 == 0u) {
			return TQ_ERR_MISSING;
		}
	}

	(void)used;
	return TQ_OK;
}

/* ------------------------------------------------------------ arena sizing */

size_t tq_kv_bytes(const TqModel *m, int max_seq, int kv_dtype)
{
	size_t n = (size_t)m->n_layers * (size_t)max_seq * (size_t)m->kv_dim;

	if (kv_dtype == TQ_KV_Q8) {
		/* int8 values plus one fp32 scale per (layer, pos, head). */
		size_t scales = (size_t)m->n_layers * (size_t)max_seq *
				(size_t)m->n_kv_heads;

		return 2u * (n + tq_align_up(scales * sizeof(float), 16u) + 16u);
	}
	return 2u * n * sizeof(float) + 32u;
}

size_t tq_runtime_bytes(const TqModel *m, int max_seq)
{
	size_t dim    = (size_t)m->dim;
	size_t hidden = (size_t)m->hidden_dim;
	size_t total  = 0;

	total += 4u * dim * sizeof(float);                    /* x, xb, xb2, q   */
	total += 2u * hidden * sizeof(float);                 /* hb, hb3         */
	total += 2u * (size_t)m->kv_dim * sizeof(float);      /* kb, vb          */
	total += (size_t)m->head_size * sizeof(float);        /* rope cos + sin  */
	total += (size_t)m->n_heads * (size_t)max_seq * sizeof(float); /* att    */
	total += (size_t)m->vocab_size * sizeof(float);       /* logits          */
	total += (dim / TQ_GROUP_SIZE) * sizeof(TqBlockQ80);  /* xq              */
	total += (hidden / TQ_GROUP_SIZE) * sizeof(TqBlockQ80); /* hq            */
	total += 16u * 14u;                                   /* alignment slack */
	return total;
}

int tq_runtime_init(TqRuntime *rt, const TqModel *m, int max_seq, int kv_dtype,
		    void *fast, size_t fast_bytes,
		    void *cache, size_t cache_bytes)
{
	uint8_t *f = (uint8_t *)fast;
	uint8_t *c = (uint8_t *)cache;
	size_t fu = 0, cu = 0;
	size_t dim = (size_t)m->dim;
	size_t hidden = (size_t)m->hidden_dim;
	size_t kvn;

	if (max_seq <= 0 || max_seq > m->seq_len) {
		return TQ_ERR_RANGE;
	}
	if (kv_dtype != TQ_KV_F32 && kv_dtype != TQ_KV_Q8) {
		return TQ_ERR_RANGE;
	}

	memset(rt, 0, sizeof(*rt));
	rt->m = m;
	rt->max_seq = max_seq;
	rt->kv_dtype = kv_dtype;

	rt->x      = tq_carve(f, fast_bytes, &fu, dim * sizeof(float));
	rt->xb     = tq_carve(f, fast_bytes, &fu, dim * sizeof(float));
	rt->xb2    = tq_carve(f, fast_bytes, &fu, dim * sizeof(float));
	rt->q      = tq_carve(f, fast_bytes, &fu, dim * sizeof(float));
	rt->hb     = tq_carve(f, fast_bytes, &fu, hidden * sizeof(float));
	rt->hb3    = tq_carve(f, fast_bytes, &fu, hidden * sizeof(float));
	rt->att    = tq_carve(f, fast_bytes, &fu,
			      (size_t)m->n_heads * (size_t)max_seq * sizeof(float));
	rt->logits = tq_carve(f, fast_bytes, &fu, (size_t)m->vocab_size * sizeof(float));
	rt->xq     = tq_carve(f, fast_bytes, &fu,
			      (dim / TQ_GROUP_SIZE) * sizeof(TqBlockQ80));
	rt->hq     = tq_carve(f, fast_bytes, &fu,
			      (hidden / TQ_GROUP_SIZE) * sizeof(TqBlockQ80));
	rt->kb       = tq_carve(f, fast_bytes, &fu, (size_t)m->kv_dim * sizeof(float));
	rt->vb       = tq_carve(f, fast_bytes, &fu, (size_t)m->kv_dim * sizeof(float));
	rt->rope_cos = tq_carve(f, fast_bytes, &fu,
				(size_t)(m->head_size / 2) * sizeof(float));
	rt->rope_sin = tq_carve(f, fast_bytes, &fu,
				(size_t)(m->head_size / 2) * sizeof(float));

	if (rt->x == NULL || rt->xb == NULL || rt->xb2 == NULL || rt->q == NULL ||
	    rt->hb == NULL || rt->hb3 == NULL || rt->att == NULL ||
	    rt->logits == NULL || rt->xq == NULL || rt->hq == NULL ||
	    rt->kb == NULL || rt->vb == NULL ||
	    rt->rope_cos == NULL || rt->rope_sin == NULL) {
		return TQ_ERR_NOMEM;
	}

	kvn = (size_t)m->n_layers * (size_t)max_seq * (size_t)m->kv_dim;
	if (kv_dtype == TQ_KV_F32) {
		rt->k_cache = tq_carve(c, cache_bytes, &cu, kvn * sizeof(float));
		rt->v_cache = tq_carve(c, cache_bytes, &cu, kvn * sizeof(float));
	} else {
		size_t ns = (size_t)m->n_layers * (size_t)max_seq * (size_t)m->n_kv_heads;

		rt->k_cache = tq_carve(c, cache_bytes, &cu, kvn);
		rt->v_cache = tq_carve(c, cache_bytes, &cu, kvn);
		rt->k_scale = tq_carve(c, cache_bytes, &cu, ns * sizeof(float));
		rt->v_scale = tq_carve(c, cache_bytes, &cu, ns * sizeof(float));
		if (rt->k_scale == NULL || rt->v_scale == NULL) {
			return TQ_ERR_NOMEM;
		}
	}
	if (rt->k_cache == NULL || rt->v_cache == NULL) {
		return TQ_ERR_NOMEM;
	}

	return TQ_OK;
}

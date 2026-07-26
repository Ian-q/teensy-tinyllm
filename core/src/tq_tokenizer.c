/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * SentencePiece-style BPE encode/decode.
 *
 * The vocabulary ships inside the .etq file as three blobs: a packed string
 * table, its offsets, and a lexicographic permutation of the token ids. That
 * last one is the trick — string->id lookup is a binary search over a
 * precomputed sort order, so the device does zero work at startup. Building
 * the sort order on-board would mean sorting 32000 strings in PSRAM before
 * the first token, which is exactly the kind of avoidable cost that makes an
 * embedded demo feel broken.
 *
 * Encode requires a memory-mapped store. It does thousands of tiny random
 * reads and streaming those off SD would take longer than generating the
 * reply.
 */

#include <string.h>

#include "tq_internal.h"

static void tq_piece(const TqTokenizer *t, int id, const char **s, uint32_t *len)
{
	uint32_t a = t->offs[id];
	uint32_t b = t->offs[id + 1];

	*s = t->blob + a;
	*len = b - a;
}

/* Lexicographic compare of two length-delimited strings. */
static int tq_cmp(const char *a, uint32_t alen, const char *b, uint32_t blen)
{
	uint32_t n = alen < blen ? alen : blen;
	int c = n ? memcmp(a, b, n) : 0;

	if (c != 0) {
		return c;
	}
	if (alen == blen) {
		return 0;
	}
	return alen < blen ? -1 : 1;
}

/* Token id for an exact string, or -1. */
static int tq_lookup(const TqTokenizer *t, const char *s, uint32_t len)
{
	int lo = 0;
	int hi = t->vocab_size - 1;

	while (lo <= hi) {
		int mid = lo + (hi - lo) / 2;
		int id = (int)t->sorted[mid];
		const char *p;
		uint32_t pl;
		int c;

		tq_piece(t, id, &p, &pl);
		c = tq_cmp(s, len, p, pl);
		if (c == 0) {
			return id;
		}
		if (c < 0) {
			hi = mid - 1;
		} else {
			lo = mid + 1;
		}
	}
	return -1;
}

int tq_tokenizer_init(TqTokenizer *t, const TqModel *m, char *buf, size_t buf_bytes)
{
	const uint8_t *base = m->store->base;

	memset(t, 0, sizeof(*t));
	if (base == NULL) {
		return TQ_ERR_UNSUPPORTED;
	}
	if (m->tok_scores == 0u || m->tok_offs == 0u || m->tok_blob == 0u ||
	    m->tok_sorted == 0u) {
		return TQ_ERR_MISSING;
	}
	if (buf_bytes < 64u) {
		return TQ_ERR_NOMEM;
	}

	t->m          = m;
	t->scores     = (const float *)(const void *)(base + m->tok_scores);
	t->offs       = (const uint32_t *)(const void *)(base + m->tok_offs);
	t->sorted     = (const uint32_t *)(const void *)(base + m->tok_sorted);
	t->blob       = (const char *)(base + m->tok_blob);
	t->vocab_size = m->vocab_size;
	t->buf        = buf;
	t->buf_bytes  = buf_bytes;
	return TQ_OK;
}

/* Byte-fallback tokens are spelled "<0xNN>" in the vocabulary. */
static int tq_byte_token(const TqTokenizer *t, unsigned char b)
{
	static const char hex[] = "0123456789ABCDEF";
	char s[6];

	s[0] = '<';
	s[1] = '0';
	s[2] = 'x';
	s[3] = hex[b >> 4];
	s[4] = hex[b & 0x0Fu];
	s[5] = '>';
	return tq_lookup(t, s, 6u);
}

int tq_encode(TqTokenizer *t, const char *text, int add_bos, int add_eos,
	      int32_t *tokens, int max_tokens)
{
	size_t i = 0;
	int n = 0;
	size_t textlen = strlen(text);

#define TQ_PUSH(v)                          \
	do {                                \
		if (n >= max_tokens) {      \
			return TQ_ERR_NOMEM; \
		}                           \
		tokens[n++] = (int32_t)(v); \
	} while (0)

	if (add_bos) {
		TQ_PUSH(t->m->hdr.bos_token);
	}

	/* SentencePiece prepends a space to non-empty input. */
	if (textlen > 0u) {
		int d = tq_lookup(t, " ", 1u);

		if (d >= 0) {
			TQ_PUSH(d);
		}
	}

	/* Split into UTF-8 codepoints, longest-match each, byte-fallback the
	 * rest. */
	while (i < textlen) {
		unsigned char c = (unsigned char)text[i];
		size_t len = 1;
		int id;

		if ((c & 0xE0u) == 0xC0u) {
			len = 2;
		} else if ((c & 0xF0u) == 0xE0u) {
			len = 3;
		} else if ((c & 0xF8u) == 0xF0u) {
			len = 4;
		}
		if (i + len > textlen) {
			len = 1;   /* truncated sequence: fall back to bytes */
		}

		id = tq_lookup(t, text + i, (uint32_t)len);
		if (id >= 0) {
			TQ_PUSH(id);
		} else {
			size_t j;

			for (j = 0; j < len; j++) {
				int b = tq_byte_token(t, (unsigned char)text[i + j]);

				if (b >= 0) {
					TQ_PUSH(b);
				}
			}
		}
		i += len;
	}

	/* Greedy BPE: repeatedly apply the highest-scoring adjacent merge. */
	for (;;) {
		float best_score = 0.0f;
		int best_id = -1;
		int best_at = -1;
		int k;

		for (k = 0; k < n - 1; k++) {
			const char *a, *b;
			uint32_t al, bl;
			int id;

			tq_piece(t, (int)tokens[k], &a, &al);
			tq_piece(t, (int)tokens[k + 1], &b, &bl);
			if ((size_t)al + bl > t->buf_bytes) {
				continue;
			}
			memcpy(t->buf, a, al);
			memcpy(t->buf + al, b, bl);

			id = tq_lookup(t, t->buf, al + bl);
			if (id >= 0 && (best_id < 0 || t->scores[id] > best_score)) {
				best_score = t->scores[id];
				best_id = id;
				best_at = k;
			}
		}
		if (best_at < 0) {
			break;
		}
		tokens[best_at] = (int32_t)best_id;
		memmove(&tokens[best_at + 1], &tokens[best_at + 2],
			(size_t)(n - best_at - 2) * sizeof(tokens[0]));
		n--;
	}

	if (add_eos) {
		TQ_PUSH(t->m->hdr.eos_token);
	}
#undef TQ_PUSH
	return n;
}

static int tq_hexval(char c)
{
	if (c >= '0' && c <= '9') {
		return c - '0';
	}
	if (c >= 'a' && c <= 'f') {
		return c - 'a' + 10;
	}
	if (c >= 'A' && c <= 'F') {
		return c - 'A' + 10;
	}
	return -1;
}

int tq_decode(TqTokenizer *t, int prev, int token, char *out, size_t out_bytes)
{
	const char *p;
	uint32_t len;

	if (token < 0 || token >= t->vocab_size || out_bytes < 2u) {
		return TQ_ERR_RANGE;
	}
	tq_piece(t, token, &p, &len);

	/* Llama emits a leading space on the first real token after BOS; the
	 * reference decoder drops it so output does not start with a blank. */
	if (prev == (int)t->m->hdr.bos_token && len > 0u && p[0] == ' ') {
		p++;
		len--;
	}

	if (len == 6u && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
		int hi = tq_hexval(p[3]);
		int lo = tq_hexval(p[4]);

		if (hi >= 0 && lo >= 0) {
			out[0] = (char)((hi << 4) | lo);
			out[1] = '\0';
			return 1;
		}
	}

	if ((size_t)len + 1u > out_bytes) {
		len = (uint32_t)(out_bytes - 1u);
	}
	memcpy(out, p, len);
	out[len] = '\0';
	return (int)len;
}

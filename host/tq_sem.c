/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Semaphore: compress a short message against the model's own predictions.
 *
 * The premise is that two endpoints holding the identical .etq do not need to
 * exchange text — only the residual surprise of the text under a distribution
 * they can both compute. tools/semaphore_probe.py measured the ceiling (the
 * model's cross-entropy); this runs the actual coder, so the number it prints
 * is bytes that would really go on the air, including the flush and the cost
 * of keeping every symbol codeable.
 *
 *   tq_sem MODEL.etq encode "text"
 *   tq_sem MODEL.etq decode 4f2a...
 *   tq_sem MODEL.etq bench corpus.txt
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>

#include "tq/tq.h"
#include "tq/tq_ac.h"

#define MAX_MSG_TOKENS 512
#define MAX_WIRE       4096

static TqStore     g_store;
static TqModel     g_model;
static TqRuntime   g_rt;
static TqTokenizer g_tok;
static TqAcModel   g_pm;
static int         g_have_tok;

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int open_model(const char *path)
{
	static char tokbuf[64];
	int    fd = open(path, O_RDONLY);
	struct stat st;
	void  *map;
	void  *marena, *fast, *cache;
	size_t fb, cb;
	int    rc, ctx;

	if (fd < 0 || fstat(fd, &st) != 0) {
		perror(path);
		return -1;
	}
	map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (map == MAP_FAILED) {
		perror("mmap");
		return -1;
	}
	tq_store_init_mapped(&g_store, map, (uint64_t)st.st_size);

	marena = malloc(64 * 1024);
	rc = tq_model_open(&g_model, &g_store, marena, 64 * 1024);
	if (rc != TQ_OK) {
		fprintf(stderr, "open: %s\n", tq_strerror(rc));
		return -1;
	}

	ctx = MAX_MSG_TOKENS < g_model.seq_len ? MAX_MSG_TOKENS : g_model.seq_len;
	fb  = tq_runtime_bytes(&g_model, ctx);
	cb  = tq_kv_bytes(&g_model, ctx, TQ_KV_F32);
	fast  = malloc(fb);
	cache = malloc(cb);
	rc = tq_runtime_init(&g_rt, &g_model, ctx, TQ_KV_F32, fast, fb, cache, cb);
	if (rc != TQ_OK) {
		fprintf(stderr, "runtime: %s\n", tq_strerror(rc));
		return -1;
	}

	rc = tq_ac_model_init(&g_pm, g_model.vocab_size);
	if (rc != TQ_OK) {
		fprintf(stderr, "ac model: %s\n", tq_strerror(rc));
		return -1;
	}

	g_have_tok = (tq_tokenizer_init(&g_tok, &g_model, tokbuf,
					sizeof(tokbuf)) == TQ_OK);
	if (!g_have_tok) {
		fprintf(stderr, "this model has no tokenizer; encode needs one\n");
	}
	return 0;
}

/* ---------------------------------------------------------------- encoding */

/*
 * Wire format: an 8-bit token count, then every token after the BOS.
 *
 * The obvious alternative is to end the stream with EOS and let the model
 * predict it, which needs no length field at all. Measured, that is much
 * worse: EOS cost 25.4 bits per message on the TinyStories corpus — 29% of
 * the entire transmission — because a story model mid-narrative assigns almost
 * no probability to stopping. It is the one symbol the model has no reason to
 * expect, so the thing that makes Semaphore work is exactly what makes an
 * in-band terminator expensive.
 *
 * A flat count over [0,256) costs 8 bits, always, and wins outright. Coding it
 * against a realistic length prior would save another 2-3, and a per-token
 * binary continue/stop flag would let the encoder stream without knowing the
 * length up front — both worth doing, neither done.
 *
 * Also reports two reference figures, because the gap between them is where
 * any future improvement has to come from:
 *
 *   `ideal` is the model's own cross-entropy in fp32 — the same quantity
 *   tools/semaphore_probe.py measures, and a floor no coder can beat.
 *   `charged` is what the integer count tables actually bill, so
 *   charged - ideal is the price of quantizing the distribution, and
 *   8*len - charged is the price of the coder itself plus the flush.
 */
#define SEM_LEN_SYMBOLS 256
#define SEM_LEN_FREQ    (TQ_AC_TOT / SEM_LEN_SYMBOLS)

static int sem_encode(const int32_t *toks, int n, uint8_t *out, size_t cap,
		      size_t *len, double *ideal, double *charged, double *term)
{
	TqAcEnc e;
	int     i, rc;

	if (n < 1 || n - 1 >= SEM_LEN_SYMBOLS) {
		return TQ_ERR_RANGE;
	}

	tq_ac_enc_init(&e, out, cap);
	*ideal = *charged = 0.0;

	tq_ac_enc_sym(&e, (uint32_t)(n - 1) * SEM_LEN_FREQ, SEM_LEN_FREQ,
		      TQ_AC_TOT);
	*term = 8.0;

	for (i = 0; i + 1 < n; i++) {
		int next = toks[i + 1];

		rc = tq_forward(&g_rt, toks[i], i);
		if (rc != TQ_OK) {
			return rc;
		}

		{
			const float *lg = g_rt.logits;
			double max = lg[0], sum = 0.0;
			int    k;

			for (k = 1; k < g_model.vocab_size; k++) {
				if ((double)lg[k] > max) {
					max = (double)lg[k];
				}
			}
			for (k = 0; k < g_model.vocab_size; k++) {
				sum += exp((double)lg[k] - max);
			}
			*ideal -= log2(exp((double)lg[next] - max) / sum);
		}

		rc = tq_ac_enc_token(&e, &g_pm, g_rt.logits, next);
		if (rc != TQ_OK) {
			return rc;
		}
		*charged -= log2((double)g_pm.bcnt[next / TQ_AC_BUCKET] /
				 (double)TQ_AC_TOT)
			  + log2((double)g_pm.scnt[next % TQ_AC_BUCKET] /
				 (double)TQ_AC_TOT);
	}

	*len = tq_ac_enc_finish(&e);
	return e.overflow ? TQ_ERR_NOMEM : TQ_OK;
}

static int sem_decode(const uint8_t *in, size_t len, int32_t *toks, int max)
{
	TqAcDec  d;
	uint32_t target;
	int      n = 0, want, rc;

	tq_ac_dec_init(&d, in, len);
	toks[n++] = (int32_t)g_model.hdr.bos_token;

	target = tq_ac_dec_target(&d, TQ_AC_TOT);
	want   = (int)(target / SEM_LEN_FREQ);
	tq_ac_dec_update(&d, (uint32_t)want * SEM_LEN_FREQ, SEM_LEN_FREQ);

	if (want + 1 > max) {
		return TQ_ERR_NOMEM;
	}

	while (n < want + 1) {
		int next;

		rc = tq_forward(&g_rt, toks[n - 1], n - 1);
		if (rc != TQ_OK) {
			return rc;
		}
		next = tq_ac_dec_token(&d, &g_pm, g_rt.logits);
		if (next < 0) {
			return next;
		}
		toks[n++] = (int32_t)next;
	}
	return n;
}

static void print_text(const int32_t *toks, int n)
{
	char buf[64];
	int  i;

	/* tq_decode returns the byte count, not TQ_OK — a zero-length piece is
	 * legal, so the test is >= 0. */
	for (i = 1; i < n; i++) {
		if (tq_decode(&g_tok, toks[i - 1], toks[i], buf, sizeof(buf)) >= 0) {
			fputs(buf, stdout);
		}
	}
	putchar('\n');
}

/* ----------------------------------------------------- classical baselines */

/*
 * Raw deflate, no zlib header — the honest framing for a payload riding inside
 * a radio protocol that already carries its own. On a 60-byte message it barely
 * compresses at all, because there is no room to build a dictionary.
 *
 * Which is why the dictionary-primed variant is the baseline that actually
 * matters: both endpoints already share a model, so it is only fair to let the
 * classical coder share something too. That is the strongest thing deflate can
 * do here, and beating it is the real claim.
 *
 * The dictionary MUST be held out. tools/semaphore_probe.py originally primed
 * it with paraphrases of the test messages and deflate appeared to win at 2.19
 * bits/char — test-set leakage, not a result. tools/corpora/dict.txt is the
 * corrected corpus: same register, no shared phrasing.
 */
static uint8_t g_dict[32768];
static size_t  g_dict_len;

static void load_dict(const char *path)
{
	FILE *f = fopen(path, "rb");

	if (!f) {
		perror(path);
		exit(1);
	}
	g_dict_len = fread(g_dict, 1, sizeof(g_dict), f);
	fclose(f);
}

static size_t deflate_bytes(const char *text, int use_dict)
{
	z_stream zs;
	uint8_t  out[MAX_WIRE];
	size_t   n;

	memset(&zs, 0, sizeof(zs));
	deflateInit2(&zs, 9, Z_DEFLATED, -15, 9, Z_DEFAULT_STRATEGY);
	if (use_dict && g_dict_len > 0) {
		deflateSetDictionary(&zs, g_dict, (uInt)g_dict_len);
	}
	zs.next_in   = (Bytef *)(uintptr_t)text;
	zs.avail_in  = (uInt)strlen(text);
	zs.next_out  = out;
	zs.avail_out = sizeof(out);
	deflate(&zs, Z_FINISH);
	n = sizeof(out) - zs.avail_out;
	deflateEnd(&zs);
	return n;
}

/* --------------------------------------------------------------- the modes */

struct msg_result {
	size_t raw, deflate, defdict, wire;
	double ideal, charged, term, secs;
	int    tokens, ok;
};

static int do_message(const char *text, struct msg_result *r, int verbose)
{
	int32_t  toks[MAX_MSG_TOKENS];
	int32_t  back[MAX_MSG_TOKENS];
	uint8_t  wire[MAX_WIRE];
	size_t   len = 0;
	double   ideal, charged, term, t0;
	int      n, m, i, ok;

	memset(r, 0, sizeof(*r));
	r->raw     = strlen(text);
	r->deflate = deflate_bytes(text, 0);
	r->defdict = deflate_bytes(text, 1);

	n = tq_encode(&g_tok, text, 1, 0, toks, MAX_MSG_TOKENS);
	if (n < 0) {
		fprintf(stderr, "encode: %s\n", tq_strerror(n));
		return 1;
	}

	t0 = now_s();
	if (sem_encode(toks, n, wire, sizeof(wire), &len, &ideal,
		       &charged, &term) != TQ_OK) {
		fprintf(stderr, "coder failed\n");
		return 1;
	}
	t0 = now_s() - t0;

	/* The round trip is the only honest verification: a codec that compresses
	 * beautifully and decodes to something else has compressed nothing. */
	m  = sem_decode(wire, len, back, MAX_MSG_TOKENS);
	ok = (m == n);
	for (i = 0; ok && i < n; i++) {
		ok = (back[i] == toks[i]);
	}

	r->wire    = len;
	r->ideal   = ideal;
	r->charged = charged;
	r->term    = term;
	r->secs    = t0;
	r->tokens  = n;
	r->ok      = ok;

	if (verbose) {
		printf("text        : %s\n", text);
		printf("tokens      : %d (+ 8-bit length)\n", n - 1);
		printf("wire        : ");
		for (i = 0; i < (int)len; i++) {
			printf("%02x", wire[i]);
		}
		printf("\n");
		printf("raw         : %zu bytes\n", r->raw);
		printf("deflate     : %zu bytes (%zu with a shared dictionary)\n",
		       r->deflate, r->defdict);
		printf("semaphore   : %zu bytes  (%.2f bits/char, %.1fx vs raw)\n",
		       len, (double)len * 8.0 / (double)r->raw,
		       (double)r->raw / (double)len);
		/*
		 * The table can charge *less* than the model's own cross-entropy on
		 * a single message, which is not an error: the minimum count of 1
		 * per symbol prices a genuinely surprising token — an EOS the model
		 * did not see coming — below its true improbability. Over a corpus
		 * the KL divergence makes the average come out above the ideal,
		 * which is why the bench totals are the number to quote.
		 */
		printf("  model ideal   %6.1f bits  (%.2f bits/char)\n",
		       ideal, ideal / (double)r->raw);
		printf("  table charges %6.1f bits  (%+.1f%% vs ideal, integer counts)\n",
		       charged, 100.0 * (charged - ideal) / ideal);
		printf("  on the wire   %6.1f bits  (%+.1f%% vs table, coder + flush)\n",
		       (double)len * 8.0, 100.0 * ((double)len * 8.0 - charged) /
		       charged);
		printf("  length field  %6.1f bits of that (was 25.4 as an in-band EOS)\n",
		       term);
		printf("round trip  : %s\n", ok ? "exact" : "*** MISMATCH ***");
		printf("encode time : %.2f s host (%d forward passes)\n", t0, n);
	} else {
		printf("%4zu %5zu %5zu %5zu  %7.2f  %s  %s\n",
		       r->raw, r->deflate, r->defdict, len,
		       (double)len * 8.0 / (double)r->raw,
		       ok ? "ok  " : "FAIL", text);
	}
	return ok ? 0 : 1;
}

static int do_bench(const char *path)
{
	FILE  *f = fopen(path, "r");
	char   line[1024];
	size_t tot_raw = 0, tot_def = 0, tot_dict = 0, tot_sem = 0;
	double tot_ideal = 0.0, tot_term = 0.0, tot_secs = 0.0;
	int    n = 0, bad = 0;

	if (!f) {
		perror(path);
		return 1;
	}
	printf(" raw  defl +dict   sem  bits/ch  ok    message\n");
	printf("---- ----- ----- -----  -------  ----  -------\n");
	while (fgets(line, sizeof(line), f)) {
		struct msg_result r;
		size_t l = strlen(line);

		while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) {
			line[--l] = '\0';
		}
		if (l == 0) {
			continue;
		}
		if (do_message(line, &r, 0) != 0) {
			bad++;
		}
		tot_raw   += r.raw;
		tot_def   += r.deflate;
		tot_dict  += r.defdict;
		tot_sem   += r.wire;
		tot_ideal += r.ideal;
		tot_term  += r.term;
		tot_secs  += r.secs;
		n++;
	}
	fclose(f);

	if (n == 0) {
		fprintf(stderr, "no messages in %s\n", path);
		return 1;
	}
	printf("\n%d messages, %zu chars\n", n, tot_raw);
	printf("  %-12s %8.3f bits/char  %6.1f bytes/msg\n", "raw",
	       8.0, (double)tot_raw / n);
	printf("  %-12s %8.3f bits/char  %6.1f bytes/msg  %5.2fx\n", "deflate",
	       (double)tot_def * 8.0 / (double)tot_raw, (double)tot_def / n,
	       (double)tot_raw / (double)tot_def);
	printf("  %-12s %8.3f bits/char  %6.1f bytes/msg  %5.2fx\n", "deflate+dict",
	       (double)tot_dict * 8.0 / (double)tot_raw, (double)tot_dict / n,
	       (double)tot_raw / (double)tot_dict);
	printf("  %-12s %8.3f bits/char  %6.1f bytes/msg  %5.2fx\n", "SEMAPHORE",
	       (double)tot_sem * 8.0 / (double)tot_raw, (double)tot_sem / n,
	       (double)tot_raw / (double)tot_sem);
	printf("  %-12s %8.3f bits/char   (cross-entropy floor, unreachable)\n",
	       "model ideal", tot_ideal / (double)tot_raw);
	printf("\nsemaphore is %.2fx smaller than the best classical coder, and\n"
	       "%.1f%% above the model's own cross-entropy — that overhead is the\n"
	       "coder, the integer count tables, and one flush per message.\n",
	       (double)(tot_dict < tot_def ? tot_dict : tot_def) / (double)tot_sem,
	       100.0 * ((double)tot_sem * 8.0 - tot_ideal) / tot_ideal);
	printf("framing         %8.3f bits/char  %6.1f bits/msg  = %.0f%% of the wire\n",
	       tot_term / (double)tot_raw, tot_term / n,
	       100.0 * tot_term / ((double)tot_sem * 8.0));
	printf("host encode time: %.1f s total, %.2f s/message\n",
	       tot_secs, tot_secs / n);
	if (bad) {
		printf("*** %d of %d round trips FAILED ***\n", bad, n);
	}
	return bad ? 1 : 0;
}

static int do_decode(const char *hex)
{
	uint8_t wire[MAX_WIRE];
	int32_t back[MAX_MSG_TOKENS];
	size_t  len = 0;
	int     n;

	while (hex[0] && hex[1] && len < sizeof(wire)) {
		unsigned v;

		if (sscanf(hex, "%2x", &v) != 1) {
			break;
		}
		wire[len++] = (uint8_t)v;
		hex += 2;
	}

	n = sem_decode(wire, len, back, MAX_MSG_TOKENS);
	if (n < 0) {
		fprintf(stderr, "decode: %s\n", tq_strerror(n));
		return 1;
	}
	print_text(back, n);
	return 0;
}

/*
 * The counterpart to `tinyllm sem hash` on the board. Same fixed token
 * sequence, derived from the vocabulary size so it needs no tokenizer, and the
 * same FNV-1a over the raw logit bytes.
 *
 * If these two values agree, the laptop and the Teensy compute byte-identical
 * probability tables and can be the two ends of one arithmetic-coded message.
 * If they disagree, no amount of protocol work will make them interoperate —
 * see core/include/tq/tq_math.h.
 */
/*
 * Per-token cost, as JSON, for the visual walkthrough in docs/.
 *
 * This is the number that actually explains Semaphore. "70 characters became 5
 * bytes" is a result; "'Once upon a' cost 1.4 bits because the model had
 * already all but written it, and '42' cost 15 because it had not" is the
 * mechanism. It also shows directly where a fine-tune would pay: the expensive
 * tokens are the ones outside the model's register.
 */
static int do_explain(const char *text)
{
	int32_t toks[MAX_MSG_TOKENS];
	uint8_t wire[MAX_WIRE];
	TqAcEnc e;
	char    piece[64];
	size_t  len;
	int     n, i, rc;
	double  total = 0.0;

	n = tq_encode(&g_tok, text, 1, 0, toks, MAX_MSG_TOKENS);
	if (n < 1 || n - 1 >= SEM_LEN_SYMBOLS) {
		fprintf(stderr, "encode: bad length\n");
		return 1;
	}

	tq_ac_enc_init(&e, wire, sizeof(wire));
	tq_ac_enc_sym(&e, (uint32_t)(n - 1) * SEM_LEN_FREQ, SEM_LEN_FREQ, TQ_AC_TOT);

	printf("{\n  \"text\": \"");
	for (i = 0; text[i]; i++) {
		if (text[i] == '"' || text[i] == '\\') {
			putchar('\\');
		}
		putchar(text[i]);
	}
	printf("\",\n  \"raw_bytes\": %zu,\n", strlen(text));
	printf("  \"deflate_bytes\": %zu,\n", deflate_bytes(text, 0));
	printf("  \"deflate_dict_bytes\": %zu,\n", deflate_bytes(text, 1));
	printf("  \"length_field_bits\": 8,\n  \"tokens\": [\n");

	for (i = 0; i + 1 < n; i++) {
		int    next = toks[i + 1];
		double bits;
		int    k;

		rc = tq_forward(&g_rt, toks[i], i);
		if (rc != TQ_OK) {
			fprintf(stderr, "forward: %s\n", tq_strerror(rc));
			return 1;
		}
		tq_ac_enc_token(&e, &g_pm, g_rt.logits, next);
		bits = -log2((double)g_pm.bcnt[next / TQ_AC_BUCKET] / (double)TQ_AC_TOT)
		       - log2((double)g_pm.scnt[next % TQ_AC_BUCKET] / (double)TQ_AC_TOT);
		total += bits;

		if (tq_decode(&g_tok, toks[i], next, piece, sizeof(piece)) < 0) {
			piece[0] = '\0';
		}
		printf("    {\"piece\": \"");
		for (k = 0; piece[k]; k++) {
			if (piece[k] == '"' || piece[k] == '\\') {
				putchar('\\');
			}
			putchar(piece[k]);
		}
		printf("\", \"bits\": %.2f}%s\n", bits, (i + 2 < n) ? "," : "");
	}

	len = tq_ac_enc_finish(&e);
	printf("  ],\n  \"token_bits_total\": %.2f,\n", total);
	printf("  \"wire_bytes\": %zu\n}\n", len);
	return 0;
}

#define SEM_HASH_STEPS 8

static int do_hash(void)
{
	uint64_t h = 1469598103934665603ull;
	int      step, i, k, rc;

	for (step = 0; step < SEM_HASH_STEPS; step++) {
		int tok = (step * 4099 + 17) % g_model.vocab_size;

		rc = tq_forward(&g_rt, tok, step);
		if (rc != TQ_OK) {
			fprintf(stderr, "forward: %s\n", tq_strerror(rc));
			return 1;
		}
		for (i = 0; i < g_model.vocab_size; i++) {
			unsigned char b[sizeof(float)];

			memcpy(b, &g_rt.logits[i], sizeof(b));
			for (k = 0; k < (int)sizeof(b); k++) {
				h ^= (uint64_t)b[k];
				h *= 1099511628211ull;
			}
		}
	}
	printf("logit bit hash: %016llx  [%s]\n", (unsigned long long)h,
	       tq_kernel_backend());
	return 0;
}

static void usage(void)
{
	fprintf(stderr,
		"usage: tq_sem MODEL.etq <mode> [arg]\n"
		"  encode TEXT     compress one message, show the wire bytes\n"
		"  decode HEX      decompress wire bytes back to text\n"
		"  bench FILE      one message per line, totals at the end\n"
		"  hash            logit fingerprint; must equal `tinyllm sem hash`\n"
		"  explain TEXT    per-token bit costs, as JSON\n"
		"\noptions:\n"
		"  --dict FILE     prime the deflate baseline with a shared\n"
		"                  dictionary. Must be held-out text, or the\n"
		"                  baseline is measuring leakage, not compression.\n");
}

int main(int argc, char **argv)
{
	int i;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--dict") && i + 1 < argc) {
			load_dict(argv[i + 1]);
			memmove(&argv[i], &argv[i + 2],
				(size_t)(argc - i - 2) * sizeof(char *));
			argc -= 2;
			i--;
		}
	}

	if (argc < 3) {
		usage();
		return 2;
	}
	if (open_model(argv[1]) != 0) {
		return 1;
	}

	if (!strcmp(argv[2], "encode") && argc > 3) {
		struct msg_result r;

		return do_message(argv[3], &r, 1);
	}
	if (!strcmp(argv[2], "decode") && argc > 3) {
		return do_decode(argv[3]);
	}
	if (!strcmp(argv[2], "bench") && argc > 3) {
		return do_bench(argv[3]);
	}
	if (!strcmp(argv[2], "hash")) {
		return do_hash();
	}
	if (!strcmp(argv[2], "explain") && argc > 3) {
		return do_explain(argv[3]);
	}
	usage();
	return 2;
}

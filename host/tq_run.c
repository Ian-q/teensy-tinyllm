/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Host runner: the same engine the Teensy runs, driven from a shell.
 *
 * The point is not to generate text on a laptop — llama.cpp does that better.
 * The point is that when the board produces garbage you can run the identical
 * code on the identical model file here, with a debugger, in a second. The
 * mmap'd file looks exactly like PSRAM at 0x70000000 to the core, and
 * --stream makes it look exactly like an SD card.
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "tq/tq.h"

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* --- streaming backend: pretend the model is on an SD card, not in PSRAM --- */

struct file_ctx {
	int fd;
	uint64_t reads;
	uint64_t bytes;
};

static int file_read(void *vctx, uint64_t off, void *dst, uint32_t n)
{
	struct file_ctx *c = (struct file_ctx *)vctx;
	ssize_t r = pread(c->fd, dst, n, (off_t)off);

	c->reads++;
	c->bytes += n;
	return (r == (ssize_t)n) ? 0 : -1;
}

static void usage(void)
{
	fprintf(stderr,
		"usage: tq_run MODEL.etq [options]\n"
		"  -i TEXT      prompt (default: empty, generates unconditionally)\n"
		"  -n N         tokens to generate (default 96)\n"
		"  -t T         temperature, 0 = greedy (default 0.9)\n"
		"  -p P         top-p nucleus threshold (default 0.9)\n"
		"  -s SEED      RNG seed (default 42)\n"
		"  -c N         context length (default: model seq_len)\n"
		"  --kv8        int8 KV cache instead of fp32\n"
		"  --stream KB  simulate SD streaming with a KB tile\n"
		"  --quiet      suppress generated text, print stats only\n");
}

int main(int argc, char **argv)
{
	const char *path = NULL, *prompt = "";
	int steps = 96, ctx = 0, stream_kb = 0, quiet = 0, kv8 = 0;
	float temp = 0.9f, topp = 0.9f;
	uint64_t seed = 42;
	int i, rc;

	int fd;
	struct stat st;
	void *map = MAP_FAILED;
	uint8_t *tile = NULL;
	struct file_ctx fctx = { 0 };

	TqStore store;
	TqModel model;
	TqRuntime rt;
	TqTokenizer tok;
	TqSampler sampler;
	int have_tok;

	void *marena, *fast, *cache;
	size_t fast_bytes, cache_bytes;
	int32_t *toks;
	int n_prompt = 0, pos, next, token;
	char tbuf[64], pbuf[64];
	double t0, t_first = 0.0;

	for (i = 1; i < argc; i++) {
		if (argv[i][0] != '-' && path == NULL) {
			path = argv[i];
		} else if (!strcmp(argv[i], "-i") && i + 1 < argc) {
			prompt = argv[++i];
		} else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
			steps = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
			temp = (float)atof(argv[++i]);
		} else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
			topp = (float)atof(argv[++i]);
		} else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
			seed = strtoull(argv[++i], NULL, 10);
		} else if (!strcmp(argv[i], "-c") && i + 1 < argc) {
			ctx = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "--kv8")) {
			kv8 = 1;
		} else if (!strcmp(argv[i], "--stream") && i + 1 < argc) {
			stream_kb = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "--quiet")) {
			quiet = 1;
		} else {
			usage();
			return 2;
		}
	}
	if (path == NULL) {
		usage();
		return 2;
	}

	fd = open(path, O_RDONLY);
	if (fd < 0 || fstat(fd, &st) != 0) {
		perror(path);
		return 1;
	}

	if (stream_kb > 0) {
		size_t tb = (size_t)stream_kb * 1024u;

		tile = malloc(tb);
		fctx.fd = fd;
		tq_store_init_stream(&store, (uint64_t)st.st_size, file_read, &fctx,
				     tile, (uint32_t)tb);
	} else {
		map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
		if (map == MAP_FAILED) {
			perror("mmap");
			return 1;
		}
		tq_store_init_mapped(&store, map, (uint64_t)st.st_size);
	}

	marena = malloc(64 * 1024);
	rc = tq_model_open(&model, &store, marena, 64 * 1024);
	if (rc != TQ_OK) {
		fprintf(stderr, "open: %s\n", tq_strerror(rc));
		return 1;
	}

	if (ctx <= 0 || ctx > model.seq_len) {
		ctx = model.seq_len;
	}
	fast_bytes  = tq_runtime_bytes(&model, ctx);
	cache_bytes = tq_kv_bytes(&model, ctx, kv8 ? TQ_KV_Q8 : TQ_KV_F32);
	fast  = malloc(fast_bytes);
	cache = malloc(cache_bytes);
	rc = tq_runtime_init(&rt, &model, ctx, kv8 ? TQ_KV_Q8 : TQ_KV_F32,
			     fast, fast_bytes, cache, cache_bytes);
	if (rc != TQ_OK) {
		fprintf(stderr, "runtime: %s\n", tq_strerror(rc));
		return 1;
	}

	fprintf(stderr,
		"model   : %s\n"
		"arch    : dim=%d hidden=%d layers=%d heads=%d/%d vocab=%d\n"
		"quant   : %s   kernel: %s\n"
		"context : %d   kv: %s\n"
		"memory  : weights %.2f MB   activations %.0f KB   kv %.2f MB\n"
		"store   : %s\n",
		path, model.dim, model.hidden_dim, model.n_layers,
		model.n_heads, model.n_kv_heads, model.vocab_size,
		model.qtype == TQ_DT_Q4_0 ? "q4_0" : "q8_0", tq_kernel_backend(),
		ctx, kv8 ? "int8" : "fp32",
		(double)st.st_size / 1e6, (double)fast_bytes / 1024.0,
		(double)cache_bytes / 1e6,
		stream_kb ? "streamed" : "mapped");

	toks = malloc(sizeof(int32_t) * (size_t)ctx);
	have_tok = (tq_tokenizer_init(&tok, &model, pbuf, sizeof(pbuf)) == TQ_OK);
	if (have_tok) {
		n_prompt = tq_encode(&tok, prompt, 1, 0, toks, ctx);
		if (n_prompt < 0) {
			fprintf(stderr, "encode: %s\n", tq_strerror(n_prompt));
			return 1;
		}
	} else {
		fprintf(stderr, "note   : no tokenizer in model, emitting raw ids\n");
		toks[0] = (int32_t)model.hdr.bos_token;
		n_prompt = 1;
	}

	{
		int32_t *sidx = malloc(sizeof(int32_t) * (size_t)model.vocab_size);
		float *sprob = malloc(sizeof(float) * (size_t)model.vocab_size);

		tq_sampler_init(&sampler, model.vocab_size, temp, topp, seed,
				sidx, sprob);
	}

	if (steps > ctx) {
		steps = ctx;
	}

	token = toks[0];
	t0 = now_s();
	for (pos = 0; pos < steps; pos++) {
		rc = tq_forward(&rt, token, pos);
		if (rc != TQ_OK) {
			fprintf(stderr, "\nforward: %s\n", tq_strerror(rc));
			return 1;
		}
		if (pos == 0) {
			t_first = now_s() - t0;
		}

		if (pos + 1 < n_prompt) {
			next = (int)toks[pos + 1];
		} else {
			next = tq_sample(&sampler, rt.logits);
		}
		if (next == (int)model.hdr.eos_token) {
			pos++;
			break;
		}

		if (!quiet) {
			if (have_tok) {
				tq_decode(&tok, token, next, tbuf, sizeof(tbuf));
				fputs(tbuf, stdout);
			} else {
				printf("%d ", next);
			}
			fflush(stdout);
		}
		token = next;
	}
	{
		double dt = now_s() - t0;

		if (!quiet) {
			printf("\n");
		}
		fprintf(stderr,
			"\n--\n%d tokens in %.3f s = %.2f tok/s "
			"(first token %.3f s)\n"
			"weight bytes read: %.1f MB total, %.2f MB/token\n",
			pos, dt, (double)pos / dt, t_first,
			(double)rt.bytes_read / 1e6,
			(double)rt.bytes_read / 1e6 / (pos ? pos : 1));
		if (stream_kb) {
			fprintf(stderr, "stream: %llu reads, %.1f MB, avg %.1f KB/read\n",
				(unsigned long long)fctx.reads,
				(double)fctx.bytes / 1e6,
				(double)fctx.bytes / 1024.0 /
					(double)(fctx.reads ? fctx.reads : 1));
		}
	}

	if (map != MAP_FAILED) {
		munmap(map, (size_t)st.st_size);
	}
	close(fd);
	return 0;
}

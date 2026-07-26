/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * teensy41-tinyllm — a quantized language model running on a Teensy 4.1.
 *
 * Everything is driven from the Zephyr shell over USB CDC-ACM. The intended
 * first session after soldering is:
 *
 *     tinyllm psram sweep      find the fastest stable FlexSPI2 clock
 *     tinyllm load stories15M.etq
 *     tinyllm gen Once upon a time
 *
 * Memory plan (32 MB PSRAM, 1 MB on-chip):
 *
 *   DTCM   activations + logits    ~150 KB   touched thousands of times/token
 *   PSRAM  model weights           8-30 MB   streamed once per token
 *   PSRAM  KV cache                1-9 MB    random access, but small
 *   PSRAB  sampler scratch          256 KB   touched once per token
 *
 * The split is the whole design. Weights get read exactly once per token and
 * never reused, so they belong in the big slow memory; activations get hit
 * constantly, so they belong in the fast one.
 */

#include <stdlib.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/shell/shell.h>

#include "model_store.h"
#include "psram_flexspi2.h"
#include "tq/tq.h"

LOG_MODULE_REGISTER(tinyllm, CONFIG_TINYLLM_LOG_LEVEL);

/* ------------------------------------------------------------------ state */

/* Activations live here. Static, so the linker puts it in whatever region the
 * board's `zephyr,sram` chosen node points at — DTCM on the Teensy 4.1. */
static uint8_t fast_arena[CONFIG_TINYLLM_FAST_ARENA_BYTES] __aligned(32);
static uint8_t model_arena[4096];
static uint8_t stream_tile[CONFIG_TINYLLM_STREAM_TILE_BYTES] __aligned(32);
static char    tok_scratch[64];

static TqStore     g_store;
static TqModel     g_model;
static TqRuntime   g_rt;
static TqTokenizer g_tok;
static bool g_loaded, g_have_tok;
static uint64_t g_model_bytes;
static uint32_t g_kv_off, g_scratch_off;

/* PSRAM allocation cursor, in bytes from PSRAM_BASE. */
static uint32_t psram_alloc(uint32_t *cursor, uint32_t bytes)
{
	uint32_t at = (*cursor + 63u) & ~63u;

	*cursor = at + bytes;
	return at;
}

/* --------------------------------------------------------------- psram cmd */

static int cmd_psram(const struct shell *sh, size_t argc, char **argv)
{
	const struct psram_info *pi;
	const char *sub = (argc > 1) ? argv[1] : "info";

	if (!strcmp(sub, "sweep")) {
		int best = psram_clock_sweep(CONFIG_TINYLLM_PSRAM_MAX_CLOCK_INDEX);

		if (best < 0) {
			shell_error(sh, "no usable clock — check the solder joints "
					"on the eight QSPI pads");
			return -EIO;
		}
		shell_print(sh, "settled at %u.%u MHz",
			    psram_clock_hz(best) / 1000000u,
			    (psram_clock_hz(best) / 100000u) % 10u);
		g_loaded = false;   /* the sweep just scribbled over the array */
		return 0;
	}

	if (!strcmp(sub, "test")) {
		uint32_t mb = psram_info()->mbytes;
		uint32_t bad;

		if (mb == 0u) {
			shell_error(sh, "no PSRAM detected");
			return -ENODEV;
		}
		shell_print(sh, "full memtest over %u MB, this takes a while...", mb);
		bad = psram_memtest(0u, (size_t)mb << 20, false);
		shell_print(sh, "%s (%u bad words)", bad ? "FAIL" : "PASS", bad);
		g_loaded = false;
		return bad ? -EIO : 0;
	}

	if (!strcmp(sub, "bench")) {
		uint32_t r = psram_bench_read(0u, 1024u * 1024u);
		uint32_t w;

		g_loaded = false;
		w = psram_bench_write(0u, 1024u * 1024u);
		shell_print(sh, "sequential read  %u.%02u MB/s",
			    r / 1000u, (r / 10u) % 100u);
		shell_print(sh, "sequential write %u.%02u MB/s",
			    w / 1000u, (w / 10u) % 100u);
		shell_print(sh, "at this read rate a Q4_0 model of N million "
				"parameters decodes at about %u.%02u/N tok/s",
			    (r * 16u) / 9000u, ((r * 16u) / 90u) % 100u);
		return 0;
	}

	pi = psram_info();
	if (pi->mbytes == 0u) {
		shell_print(sh, "no PSRAM detected");
		shell_print(sh, "  chip 1 id: 0x%08x", pi->chip_id[0]);
		shell_print(sh, "  an id of 0x00000000 or 0xffffffff usually "
				"means an open joint on DATA0-3 or SCLK");
		return 0;
	}
	shell_print(sh, "%u MB across %u chip%s @ %u.%u MHz",
		    pi->mbytes, pi->chips, pi->chips == 1 ? "" : "s",
		    pi->clock_hz / 1000000u, (pi->clock_hz / 100000u) % 10u);
	shell_print(sh, "  CS0 (pad EMC_24): %u MB, id 0x%08x",
		    pi->chip_mbytes[0], pi->chip_id[0]);
	shell_print(sh, "  CS1 (pad EMC_22): %u MB, id 0x%08x",
		    pi->chip_mbytes[1], pi->chip_id[1]);
	return 0;
}

/* ---------------------------------------------------------------- loading */

static const struct shell *g_progress_sh;

static void load_progress(uint64_t done, uint64_t total)
{
	static uint32_t last_pct = 0xFFFFFFFFu;
	uint32_t pct = (uint32_t)((done * 100u) / (total ? total : 1u));

	if (pct / 10u != last_pct / 10u) {
		last_pct = pct;
		if (g_progress_sh != NULL) {
			shell_fprintf(g_progress_sh, SHELL_NORMAL, "%u%% ", pct);
		}
	}
}

static int finish_open(const struct shell *sh, int max_seq, int kv8)
{
	uint32_t cursor;
	size_t fast_need, kv_need, scratch_need;
	int rc;

	rc = tq_model_open(&g_model, &g_store, model_arena, sizeof(model_arena));
	if (rc != TQ_OK) {
		shell_error(sh, "model open: %s", tq_strerror(rc));
		return -EINVAL;
	}

	if (max_seq <= 0 || max_seq > g_model.seq_len) {
		max_seq = g_model.seq_len;
	}

	fast_need = tq_runtime_bytes(&g_model, max_seq);
	if (fast_need > sizeof(fast_arena)) {
		shell_error(sh,
			    "activations need %u KB but the DTCM arena is %u KB",
			    (unsigned)(fast_need / 1024u),
			    (unsigned)(sizeof(fast_arena) / 1024u));
		shell_error(sh, "shorten the context with -c, or raise "
				"CONFIG_TINYLLM_FAST_ARENA_BYTES and rebuild");
		return -ENOMEM;
	}

	/* Weights already occupy the front of PSRAM when resident; when
	 * streaming they occupy none of it. */
	cursor = (g_store.base != NULL) ? (uint32_t)g_model_bytes : 0u;

	kv_need = tq_kv_bytes(&g_model, max_seq, kv8 ? TQ_KV_Q8 : TQ_KV_F32);
	scratch_need = (size_t)g_model.vocab_size * (sizeof(int32_t) + sizeof(float));

	if (cursor + kv_need + scratch_need > ((uint64_t)psram_info()->mbytes << 20)) {
		shell_error(sh, "KV cache (%u KB) will not fit alongside the "
				"model; try -c %d or --kv8",
			    (unsigned)(kv_need / 1024u), max_seq / 2);
		return -ENOMEM;
	}

	g_kv_off = psram_alloc(&cursor, (uint32_t)kv_need);
	g_scratch_off = psram_alloc(&cursor, (uint32_t)scratch_need);

	rc = tq_runtime_init(&g_rt, &g_model, max_seq,
			     kv8 ? TQ_KV_Q8 : TQ_KV_F32,
			     fast_arena, sizeof(fast_arena),
			     (void *)(uintptr_t)(PSRAM_BASE + g_kv_off), kv_need);
	if (rc != TQ_OK) {
		shell_error(sh, "runtime: %s", tq_strerror(rc));
		return -ENOMEM;
	}

	g_have_tok = (tq_tokenizer_init(&g_tok, &g_model, tok_scratch,
					sizeof(tok_scratch)) == TQ_OK);
	g_loaded = true;

	shell_print(sh, "");
	shell_print(sh, "dim %d  hidden %d  layers %d  heads %d/%d  vocab %d",
		    g_model.dim, g_model.hidden_dim, g_model.n_layers,
		    g_model.n_heads, g_model.n_kv_heads, g_model.vocab_size);
	shell_print(sh, "quant %s  kernel %s  context %d  kv %s%s",
		    g_model.qtype == TQ_DT_Q4_0 ? "q4_0" : "q8_0",
		    tq_kernel_backend(), max_seq, kv8 ? "int8" : "fp32",
		    g_have_tok ? "" : "  (no tokenizer)");
	shell_print(sh, "psram: weights %u KB + kv %u KB + scratch %u KB",
		    (unsigned)(g_model_bytes / 1024u), (unsigned)(kv_need / 1024u),
		    (unsigned)(scratch_need / 1024u));
	shell_print(sh, "dtcm : activations %u KB of %u KB",
		    (unsigned)(fast_need / 1024u),
		    (unsigned)(sizeof(fast_arena) / 1024u));
	return 0;
}

static void parse_common(size_t argc, char **argv, size_t from,
			 int *max_seq, int *kv8)
{
	size_t i;

	for (i = from; i < argc; i++) {
		if (!strcmp(argv[i], "-c") && i + 1 < argc) {
			*max_seq = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "--kv8")) {
			*kv8 = 1;
		}
	}
}

static int cmd_load(const struct shell *sh, size_t argc, char **argv)
{
	char path[96];
	int max_seq = 0, kv8 = 0;
	int64_t t0;
	int rc;

	if (argc < 2) {
		shell_error(sh, "usage: tinyllm load <file.etq> [-c CTX] [--kv8]");
		return -EINVAL;
	}
	if (psram_info()->mbytes == 0u) {
		shell_error(sh, "no PSRAM — use `tinyllm stream` for SD-resident "
				"models");
		return -ENODEV;
	}
	parse_common(argc, argv, 2, &max_seq, &kv8);

	if (argv[1][0] == '/') {
		strncpy(path, argv[1], sizeof(path) - 1);
	} else {
		snprintk(path, sizeof(path), "/SD:/%s", argv[1]);
	}
	path[sizeof(path) - 1] = '\0';

	g_loaded = false;
	model_close_streaming();

	shell_fprintf(sh, SHELL_NORMAL, "loading %s: ", path);
	g_progress_sh = sh;
	t0 = k_uptime_get();
	rc = model_load_to_psram(path, 0u, &g_store, &g_model_bytes, load_progress);
	g_progress_sh = NULL;
	if (rc != 0) {
		shell_error(sh, "\nload failed: %d", rc);
		return rc;
	}
	{
		int64_t dt = k_uptime_get() - t0;

		shell_print(sh, "done  (%llu KB in %lld ms, %llu KB/s)",
			    (unsigned long long)(g_model_bytes / 1024u), dt,
			    (unsigned long long)(g_model_bytes /
						 (uint64_t)(dt ? dt : 1)));
	}
	return finish_open(sh, max_seq, kv8);
}

static int cmd_stream(const struct shell *sh, size_t argc, char **argv)
{
	char path[96];
	int max_seq = 0, kv8 = 1;   /* streaming implies a big model: default int8 kv */
	int rc;

	if (argc < 2) {
		shell_error(sh, "usage: tinyllm stream <file.etq> [-c CTX]");
		return -EINVAL;
	}
	parse_common(argc, argv, 2, &max_seq, &kv8);

	if (argv[1][0] == '/') {
		strncpy(path, argv[1], sizeof(path) - 1);
	} else {
		snprintk(path, sizeof(path), "/SD:/%s", argv[1]);
	}
	path[sizeof(path) - 1] = '\0';

	g_loaded = false;
	rc = model_open_streaming(path, &g_store, stream_tile,
				  sizeof(stream_tile), &g_model_bytes);
	if (rc != 0) {
		return rc;
	}
	return finish_open(sh, max_seq, kv8);
}

/* ------------------------------------------------------------- generation */

static int cmd_gen(const struct shell *sh, size_t argc, char **argv)
{
	TqSampler sampler;
	int32_t *toks;
	char prompt[192];
	char piece[64];
	int steps = 128, n_prompt, pos, token, next;
	float temp = 0.9f, topp = 0.9f;
	uint32_t seed = 42;
	size_t i, used = 0;
	int64_t t0, t_first = 0;
	uint64_t bytes0;

	if (!g_loaded) {
		shell_error(sh, "no model loaded — `tinyllm load <file.etq>`");
		return -ENODEV;
	}

	prompt[0] = '\0';
	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-n") && i + 1 < argc) {
			steps = atoi(argv[++i]);
		} else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
			temp = (float)atof(argv[++i]);
		} else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
			topp = (float)atof(argv[++i]);
		} else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
			seed = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else {
			size_t l = strlen(argv[i]);

			if (used + l + 2 < sizeof(prompt)) {
				if (used) {
					prompt[used++] = ' ';
				}
				memcpy(prompt + used, argv[i], l);
				used += l;
				prompt[used] = '\0';
			}
		}
	}

	if (steps > g_rt.max_seq) {
		steps = g_rt.max_seq;
	}

	/* Sampler scratch lives in PSRAM. It is 256 KB at a 32k vocab, which
	 * would be a quarter of DTCM, and it is touched exactly once per
	 * token — the cheapest thing in the whole step to make slow. */
	tq_sampler_init(&sampler, g_model.vocab_size, temp, topp, seed,
			(int32_t *)(uintptr_t)(PSRAM_BASE + g_scratch_off),
			(float *)(uintptr_t)(PSRAM_BASE + g_scratch_off +
					     (uint32_t)g_model.vocab_size * 4u));

	toks = (int32_t *)(uintptr_t)(PSRAM_BASE + g_scratch_off +
				      (uint32_t)g_model.vocab_size * 8u);

	if (g_have_tok) {
		n_prompt = tq_encode(&g_tok, prompt, 1, 0, toks, g_rt.max_seq);
		if (n_prompt < 0) {
			shell_error(sh, "encode: %s", tq_strerror(n_prompt));
			return -EINVAL;
		}
	} else {
		toks[0] = (int32_t)g_model.hdr.bos_token;
		n_prompt = 1;
	}

	bytes0 = g_rt.bytes_read;
	token = (int)toks[0];
	t0 = k_uptime_get();

	for (pos = 0; pos < steps; pos++) {
		int rc = tq_forward(&g_rt, token, pos);

		if (rc != TQ_OK) {
			shell_error(sh, "\nforward at pos %d: %s", pos,
				    tq_strerror(rc));
			return -EIO;
		}
		if (pos == 0) {
			t_first = k_uptime_get() - t0;
		}

		next = (pos + 1 < n_prompt) ? (int)toks[pos + 1]
					    : tq_sample(&sampler, g_rt.logits);
		if (next == (int)g_model.hdr.eos_token) {
			pos++;
			break;
		}
		if (g_have_tok) {
			tq_decode(&g_tok, token, next, piece, sizeof(piece));
			shell_fprintf(sh, SHELL_NORMAL, "%s", piece);
		} else {
			shell_fprintf(sh, SHELL_NORMAL, "%d ", next);
		}
		token = next;
	}

	{
		int64_t dt = k_uptime_get() - t0;
		uint64_t read = g_rt.bytes_read - bytes0;
		uint32_t mtps = (uint32_t)(((uint64_t)pos * 100000u) /
					   (uint64_t)(dt ? dt : 1));   /* tok/s * 100 */
		uint32_t bw = (uint32_t)((read * 1000u) /
					 (uint64_t)(dt ? dt : 1) / 1000u);  /* KB/s */

		shell_print(sh, "\n--");
		shell_print(sh, "%d tokens in %lld ms = %u.%02u tok/s "
				"(first token %lld ms)",
			    pos, dt, mtps / 100u, mtps % 100u, t_first);
		shell_print(sh, "weights read %llu KB total, %llu KB/token, "
				"effective %u.%02u MB/s",
			    (unsigned long long)(read / 1024u),
			    (unsigned long long)(read / 1024u /
						 (uint64_t)(pos ? pos : 1)),
			    bw / 1000u, (bw / 10u) % 100u);
	}
	return 0;
}

/* ------------------------------------------------------------------- info */

static int cmd_info(const struct shell *sh, size_t argc, char **argv)
{
	const struct psram_info *pi = psram_info();

	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	shell_print(sh, "teensy41-tinyllm");
	shell_print(sh, "  cpu     : %u MHz", (unsigned)(sys_clock_hw_cycles_per_sec() / 1000000u));
	shell_print(sh, "  kernel  : %s", tq_kernel_backend());
	shell_print(sh, "  psram   : %u MB @ %u.%u MHz", pi->mbytes,
		    pi->clock_hz / 1000000u, (pi->clock_hz / 100000u) % 10u);
	shell_print(sh, "  arena   : %u KB dtcm, %u KB stream tile",
		    (unsigned)(sizeof(fast_arena) / 1024u),
		    (unsigned)(sizeof(stream_tile) / 1024u));
	if (g_loaded) {
		shell_print(sh, "  model   : %llu KB, %s, ctx %d",
			    (unsigned long long)(g_model_bytes / 1024u),
			    g_model.qtype == TQ_DT_Q4_0 ? "q4_0" : "q8_0",
			    g_rt.max_seq);
	} else {
		shell_print(sh, "  model   : none loaded");
	}
	return 0;
}

SHELL_STATIC_SUBCMD_SET_CREATE(tinyllm_cmds,
	SHELL_CMD_ARG(psram,  NULL,
		      "psram [info|sweep|test|bench]", cmd_psram, 1, 1),
	SHELL_CMD_ARG(load,   NULL,
		      "load <file.etq> [-c CTX] [--kv8]  — copy into PSRAM",
		      cmd_load, 2, 4),
	SHELL_CMD_ARG(stream, NULL,
		      "stream <file.etq> [-c CTX]  — run from SD, no PSRAM copy",
		      cmd_stream, 2, 4),
	SHELL_CMD_ARG(gen,    NULL,
		      "gen [-n N] [-t T] [-p P] [-s SEED] <prompt...>",
		      cmd_gen, 1, 24),
	SHELL_CMD_ARG(info,   NULL, "show configuration", cmd_info, 1, 0),
	SHELL_SUBCMD_SET_END
);
SHELL_CMD_REGISTER(tinyllm, &tinyllm_cmds, "Tiny LLM on Teensy 4.1", NULL);

int main(void)
{
	int mb;

	LOG_INF("teensy41-tinyllm starting");

	/*
	 * Bring PSRAM up at the conservative clock. `tinyllm psram sweep` will
	 * push it as fast as this particular pair of chips and this particular
	 * solder job can go — that is a per-board property, not a per-design
	 * one, so it is not safe to assume at boot.
	 */
	mb = psram_init(CONFIG_TINYLLM_PSRAM_BOOT_CLOCK_INDEX);
	if (mb <= 0) {
		LOG_WRN("no PSRAM found — SD streaming still works");
	}

	if (model_sd_mount() != 0) {
		LOG_WRN("no SD card mounted; insert one and re-run "
			"`tinyllm load`");
	}

	LOG_INF("ready — try `tinyllm info`");
	return 0;
}

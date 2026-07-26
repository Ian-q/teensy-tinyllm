/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Getting a model from the microSD card into somewhere the engine can read it.
 */

#ifndef MODEL_STORE_H_
#define MODEL_STORE_H_

#include <stddef.h>
#include <stdint.h>

#include "tq/tq.h"

/* Mount the Teensy's native microSD slot (usdhc1, 4-bit SDIO) at /SD:. */
int model_sd_mount(void);

/*
 * Copy `path` into PSRAM at `psram_off` and point `store` at it.
 *
 * The copy goes through a DTCM staging buffer rather than reading straight
 * into PSRAM: the SD stack DMAs into the destination, and PSRAM is a
 * cacheable AHB region, so a direct read would leave the D-cache holding
 * stale lines over freshly DMA'd data. Staging in DTCM and doing a CPU memcpy
 * keeps the cache coherent with no explicit maintenance, and costs nothing
 * measurable next to the SD read itself.
 *
 * Reports progress through `progress` (may be NULL).
 */
int model_load_to_psram(const char *path, uint32_t psram_off, TqStore *store,
			uint64_t *out_bytes,
			void (*progress)(uint64_t done, uint64_t total));

/*
 * Leave the model on the card and stream it.
 *
 * The only way to run a model bigger than the PSRAM fitted. Roughly half the
 * token rate of the resident path, but it is the difference between running a
 * 62 MB model on a 32 MB board and not running it.
 */
int model_open_streaming(const char *path, TqStore *store, uint8_t *tile,
			 uint32_t tile_bytes, uint64_t *out_bytes);

/* Release whatever the streaming store has open. */
void model_close_streaming(void);

#endif /* MODEL_STORE_H_ */

/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * FlexSPI2 PSRAM driver for the Teensy 4.1 bottom-side QSPI pads.
 */

#ifndef PSRAM_FLEXSPI2_H_
#define PSRAM_FLEXSPI2_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* FlexSPI2 AHB window. Both chip selects map here back to back, so two 16 MB
 * parts appear as one flat 32 MB span. */
#define PSRAM_BASE 0x70000000u

/* Clock options, in the order psram_clock_hz()/psram_init() index them.
 * Derived from the FlexSPI2 PODF/CLK_SEL table in the RT1062 reference manual
 * (CCM_CBCMR). Index 3 (105.6 MHz) is what Teensyduino 1.60 ships as default;
 * index 6 (132 MHz) is the fastest the 16 MB ISSI parts are documented to hold
 * and is what `psram sweep` usually lands on. */
enum {
	PSRAM_CLK_88   = 0,
	PSRAM_CLK_99   = 1,
	PSRAM_CLK_103  = 2,
	PSRAM_CLK_106  = 3,
	PSRAM_CLK_111  = 4,
	PSRAM_CLK_120  = 5,
	PSRAM_CLK_132  = 6,
	PSRAM_CLK_144  = 7,
	PSRAM_CLK_166  = 8,
	PSRAM_CLK_176  = 9,
	PSRAM_CLK_COUNT
};

struct psram_info {
	uint8_t  chips;          /* 0, 1 or 2 parts detected           */
	uint32_t mbytes;         /* total capacity                     */
	uint32_t chip_mbytes[2];
	uint32_t chip_id[2];     /* raw JEDEC ID words                 */
	uint32_t clock_hz;
	uint8_t  clock_index;
};

uint32_t psram_clock_hz(int index);

/*
 * Bring up FlexSPI2 and detect what is soldered on.
 *
 * Returns total megabytes (0 if nothing answers). Safe to call repeatedly —
 * `psram sweep` re-runs it at each clock.
 */
int psram_init(int clock_index);

const struct psram_info *psram_info(void);

/* Non-destructive read bandwidth in KB/s over `bytes` starting at `off`. */
uint32_t psram_bench_read(uint32_t off, size_t bytes);

/* Write-then-read bandwidth in KB/s. DESTROYS the region under test. */
uint32_t psram_bench_write(uint32_t off, size_t bytes);

/*
 * Memory test. Destroys the region. Returns 0 on success, or the number of
 * failing words. `quick` samples one word per 4 KB page instead of every word,
 * which is what the clock sweep uses so it does not take a minute per step.
 */
uint32_t psram_memtest(uint32_t off, size_t bytes, bool quick);

/*
 * Walk the clock table upward, memtesting at each step, and leave the part
 * configured at the fastest clock that passes. Returns the chosen index.
 *
 * This is the point of the whole exercise: token rate is proportional to PSRAM
 * bandwidth, and the difference between the 88 MHz default and a part that
 * happens to be stable at 166 MHz is roughly a 1.9x speedup for free.
 */
int psram_clock_sweep(int max_index);

#endif /* PSRAM_FLEXSPI2_H_ */

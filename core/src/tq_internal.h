/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Cross-translation-unit helpers that are not part of the public API.
 */

#ifndef TQ_INTERNAL_H_
#define TQ_INTERNAL_H_

#include "tq/tq.h"

/* Map a byte range of the model for direct reading.
 *
 * For a mapped store this is pointer arithmetic. For a streaming store the
 * bytes land in the store's single tile buffer, so the returned pointer is
 * INVALIDATED by the next call — copy anything you need to keep. */
const void *tq_view(const TqModel *m, uint64_t off, uint32_t nbytes);

/* Dequantize row `row` of a [rows, n] quantized matrix into `out`. */
int tq_row_to_float(const TqModel *m, uint64_t base_off, uint32_t dtype,
		    int n, int row, float *out);

#endif /* TQ_INTERNAL_H_ */

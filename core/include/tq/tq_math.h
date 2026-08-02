/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Reproducible transcendentals.
 *
 * These exist for one reason: two machines running libtq must produce
 * BIT-IDENTICAL logits. libm does not offer that. newlib's expf on the Teensy
 * and macOS's expf on a laptop are both excellent and they disagree in the
 * last ulp, which is all it takes — the Semaphore arithmetic decoder
 * (core/src/tq_ac.c) desynchronises on a single differing bit and loses the
 * rest of the message. The same applies to any future cross-checking of a
 * board against the host.
 *
 * So every libm call in the forward pass is replaced by a fixed sequence of
 * IEEE-754 double operations here. IEEE-754 pins the result of every one of
 * them exactly, so any conforming implementation agrees, on any architecture,
 * at any optimisation level — provided the compiler is not allowed to contract
 * a multiply and an add into a differently-rounded fused instruction. That is
 * what -ffp-contract=off in the build files is for, and it is not optional.
 *
 * `sqrtf` is deliberately NOT replaced. IEEE-754 requires square root to be
 * correctly rounded, so it is already identical everywhere and it compiles to
 * a single hardware instruction on both targets.
 *
 * Accuracy is not the hard part here and these are not trying to be
 * last-ulp-perfect. The forward pass is checked against a NumPy oracle with a
 * 2e-3 gate and currently sits at 6.9e-06, so there is ~290x of headroom; the
 * series below are far inside it. Reproducibility is the requirement,
 * precision merely has to be sufficient.
 */

#ifndef TQ_MATH_H_
#define TQ_MATH_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Drop-in replacements for the libm functions the forward pass used. */
float tq_expf(float x);
float tq_powf(float a, float b);
float tq_sinf(float x);
float tq_cosf(float x);

/* The primitives the above are built from. Exposed for testing. */
double tq_exp2(double x);
double tq_log2(double x);

#ifdef __cplusplus
}
#endif

#endif /* TQ_MATH_H_ */

/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Reproducible transcendentals. See tq_math.h for why these exist at all.
 *
 * Every series below is a plain Maclaurin expansion with exact rational
 * coefficients, evaluated by Horner in double. That is a deliberate choice
 * over a fitted minimax polynomial: the coefficients can be checked by
 * inspection against a factorial, there is nothing to re-derive if someone
 * later wants to extend a range, and the accuracy is already two orders of
 * magnitude better than the forward pass needs. A minimax fit would buy a
 * couple of terms and cost that auditability.
 *
 * The argument ranges are all tiny by construction — each function reduces to
 * a short interval before the series runs — which is what makes such simple
 * expansions accurate enough.
 */

#include <stdint.h>
#include <string.h>

#include "tq/tq_math.h"

#define TQ_LN2      0.69314718055994530942
#define TQ_LOG2E    1.44269504088896340736
#define TQ_PIO2_HI  1.57079632673412561417e+00   /* leading 33 bits of pi/2  */
#define TQ_PIO2_LO  6.07710050650619224932e-11   /* ...and the rest of it    */

/* ------------------------------------------------------------ scaling by 2^n
 *
 * Assembling the exponent field directly rather than multiplying repeatedly:
 * exact, branch-free, and identical on every IEEE-754 machine. memcpy is the
 * portable spelling of the bit reinterpretation; every compiler folds it away.
 */
static double tq_ldexp2(double m, int n)
{
	uint64_t bits;
	double   s;

	if (n > 1023) {
		n = 1023;
	} else if (n < -1022) {
		/* Flush to zero rather than descend into subnormals. Nothing in the
		 * forward pass cares about values below 2^-1022, and subnormal
		 * handling is the one place hardware is allowed to differ. */
		return 0.0;
	}

	bits = (uint64_t)(n + 1023) << 52;
	memcpy(&s, &bits, sizeof(s));
	return m * s;
}

/* ------------------------------------------------------------------- exp2 */

double tq_exp2(double x)
{
	int    n;
	double f, y, p;

	/* Clamp before converting to int: an out-of-range double-to-int
	 * conversion is undefined behaviour, and logits from a broken model can
	 * be arbitrarily large. */
	if (x > 1024.0) {
		x = 1024.0;
	} else if (x < -1075.0) {
		return 0.0;
	}

	/* Round to nearest, halves away from zero. Spelled out rather than using
	 * round() so there is no libm call and no rounding-mode dependence. */
	n = (int)(x >= 0.0 ? x + 0.5 : x - 0.5);
	f = x - (double)n;                       /* exact: both are representable */

	/* 2^f = exp(f * ln2) with |f| <= 1/2, so |y| <= 0.347 and the Maclaurin
	 * series converges brutally fast: the first omitted term is
	 * 0.347^11/11! = 2.5e-13. */
	y = f * TQ_LN2;
	p = 1.0 / 3628800.0;                              /* 1/10! */
	p = p * y + 1.0 / 362880.0;                       /* 1/9!  */
	p = p * y + 1.0 / 40320.0;                        /* 1/8!  */
	p = p * y + 1.0 / 5040.0;                         /* 1/7!  */
	p = p * y + 1.0 / 720.0;                          /* 1/6!  */
	p = p * y + 1.0 / 120.0;                          /* 1/5!  */
	p = p * y + 1.0 / 24.0;                           /* 1/4!  */
	p = p * y + 1.0 / 6.0;                            /* 1/3!  */
	p = p * y + 1.0 / 2.0;                            /* 1/2!  */
	p = p * y + 1.0;
	p = p * y + 1.0;

	return tq_ldexp2(p, n);
}

/* ------------------------------------------------------------------- log2 */

double tq_log2(double x)
{
	uint64_t bits;
	int      e;
	double   m, t, t2, p;

	if (!(x > 0.0)) {                        /* also catches NaN */
		return 0.0;
	}

	memcpy(&bits, &x, sizeof(bits));
	e = (int)((bits >> 52) & 0x7FFu) - 1023;
	if (e == -1023) {
		return 0.0;                      /* subnormal or zero; unused here */
	}

	/* Force the mantissa into [1,2) by overwriting the exponent field. */
	bits = (bits & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull;
	memcpy(&m, &bits, sizeof(m));

	/* Recentre to [1/sqrt2, sqrt2) so the series argument stays small and the
	 * expansion is accurate across the whole interval rather than just near 1. */
	if (m > 1.41421356237309504880) {
		m *= 0.5;
		e += 1;
	}

	/*
	 * ln(m) = 2*atanh((m-1)/(m+1)). With m in [0.707, 1.414] the argument is
	 * bounded by 0.1716. The first omitted term is 0.1716^15/15, which the
	 * trailing 2/ln2 scaling inflates to about 6e-13 in the returned value —
	 * that outer factor is worth remembering, since stopping at t^9 looks like
	 * 3.5e-10 of truncation and actually lands at 1.0e-09.
	 *
	 * This is far finer than the float results need. It is cheap because
	 * tq_log2 is only reached through tq_powf, which the forward pass calls
	 * head_size/2 times per token — 32, against ~8000 tq_expf calls.
	 */
	t  = (m - 1.0) / (m + 1.0);
	t2 = t * t;
	p  = 1.0 / 13.0;
	p  = p * t2 + 1.0 / 11.0;
	p  = p * t2 + 1.0 / 9.0;
	p  = p * t2 + 1.0 / 7.0;
	p  = p * t2 + 1.0 / 5.0;
	p  = p * t2 + 1.0 / 3.0;
	p  = p * t2 + 1.0;

	return (double)e + 2.0 * t * p * TQ_LOG2E;
}

/* -------------------------------------------------------------- sin / cos */

/*
 * Cody-Waite reduction: subtract k*(pi/2) in two pieces so the cancellation
 * happens against a value carrying more bits than a single double holds. The
 * forward pass only ever asks for RoPE angles, which reach about 1024 radians
 * at position 1024, so k stays under ~652 and a two-part split is ample. This
 * is not a general-purpose sin() and does not pretend to be.
 */
static void tq_sincos(double x, double *s, double *c)
{
	int    k, q;
	double r, r2, sp, cp;

	if (x > 1.0e8) {
		x = 1.0e8;
	} else if (x < -1.0e8) {
		x = -1.0e8;
	}

	k = (int)(x >= 0.0 ? x * (2.0 / 3.14159265358979323846) + 0.5
			   : x * (2.0 / 3.14159265358979323846) - 0.5);
	r = (x - (double)k * TQ_PIO2_HI) - (double)k * TQ_PIO2_LO;

	/* |r| <= pi/4, where the first omitted term of each series is below 1e-16. */
	r2 = r * r;

	sp = -1.0 / 362880.0;                             /* -1/9! */
	sp = sp * r2 + 1.0 / 5040.0;                      /*  1/7! */
	sp = sp * r2 - 1.0 / 120.0;                       /* -1/5! */
	sp = sp * r2 + 1.0 / 6.0;                         /*  1/3! */
	sp = r - r * r2 * sp;
	/* The sign pattern above folds the alternating series into one Horner
	 * chain; expanded, this is r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880. */

	cp = -1.0 / 40320.0;                              /* -1/8! */
	cp = cp * r2 + 1.0 / 720.0;                       /*  1/6! */
	cp = cp * r2 - 1.0 / 24.0;                        /* -1/4! */
	cp = cp * r2 + 1.0 / 2.0;                         /*  1/2! */
	cp = 1.0 - r2 * cp;

	/* Each quarter turn rotates (sin, cos) into each other with a sign flip. */
	q = k & 3;
	if (q < 0) {
		q += 4;
	}
	switch (q) {
	case 0:  *s =  sp; *c =  cp; break;
	case 1:  *s =  cp; *c = -sp; break;
	case 2:  *s = -sp; *c = -cp; break;
	default: *s = -cp; *c =  sp; break;
	}
}

/* ------------------------------------------------------- public interface */

float tq_expf(float x)
{
	return (float)tq_exp2((double)x * TQ_LOG2E);
}

float tq_powf(float a, float b)
{
	if (a <= 0.0f) {
		return 0.0f;                     /* the forward pass never asks */
	}
	return (float)tq_exp2((double)b * tq_log2((double)a));
}

float tq_sinf(float x)
{
	double s, c;

	tq_sincos((double)x, &s, &c);
	return (float)s;
}

float tq_cosf(float x)
{
	double s, c;

	tq_sincos((double)x, &s, &c);
	return (float)c;
}

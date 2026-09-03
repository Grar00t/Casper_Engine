/*
 * constraint_solver.c — NIYAH Linear Constraint Solver
 *
 * Simplex-inspired feasibility checker for small systems of
 * linear inequalities over exact rational arithmetic.
 *
 * Zero external dependencies. C11 clean.
 *
 * Standalone test:
 *   gcc -O2 -std=c11 -Wall -Wextra -Werror -Wcast-align
 *       -DCSP_STANDALONE_TEST constraint_solver.c -o test_csp
 *   ./test_csp
 */

#include "constraint_solver.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * §0  Rational arithmetic
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

static int64_t i64_abs(int64_t x) { return x < 0 ? -x : x; }

static int64_t gcd(int64_t a, int64_t b) {
    a = i64_abs(a);
    b = i64_abs(b);
    while (b) { int64_t t = b; b = a % b; a = t; }
    return a;
}

static NiyahCspRat rat_normalize(int64_t n, int64_t d) {
    assert(d != 0);
    if (d < 0) { n = -n; d = -d; }
    if (n == 0) { return (NiyahCspRat){0, 1}; }
    int64_t g = gcd(i64_abs(n), d);
    return (NiyahCspRat){n / g, d / g};
}

NiyahCspRat niyah_csp_rat(int64_t num, int64_t den) {
    return rat_normalize(num, den);
}

/*
 * Overflow-safe helpers for signed 64-bit arithmetic.
 *
 * Cross-reduce a/b op c/d before multiplying to keep products in range.
 * For add/sub we additionally pre-reduce the common denominator.
 *
 * For rat_cmp we use __int128 where the target has it to avoid the product
 * overflowing; elsewhere we fall back to double comparison, which is exact
 * only while both products fit in a 53-bit mantissa.
 */

NiyahCspRat niyah_csp_rat_add(NiyahCspRat a, NiyahCspRat b) {
    /* Reduce common denominator first: g = gcd(a.den, b.den)
     * a/b + c/d = (a*(d/g) + c*(b/g)) / (b*(d/g)) */
    int64_t g = gcd(a.den, b.den);
    int64_t bdog = b.den / g;   /* b.den / gcd */
    int64_t adog = a.den / g;   /* a.den / gcd */
    return rat_normalize(a.num * bdog + b.num * adog, a.den * bdog);
}

NiyahCspRat niyah_csp_rat_sub(NiyahCspRat a, NiyahCspRat b) {
    int64_t g = gcd(a.den, b.den);
    int64_t bdog = b.den / g;
    int64_t adog = a.den / g;
    return rat_normalize(a.num * bdog - b.num * adog, a.den * bdog);
}

NiyahCspRat niyah_csp_rat_mul(NiyahCspRat a, NiyahCspRat b) {
    /* Cross-reduce before multiplying to minimise magnitude */
    int64_t g1 = gcd(i64_abs(a.num), b.den);
    int64_t g2 = gcd(i64_abs(b.num), a.den);
    return rat_normalize((a.num / g1) * (b.num / g2),
                         (a.den / g2) * (b.den / g1));
}

NiyahCspRat niyah_csp_rat_div(NiyahCspRat a, NiyahCspRat b) {
    assert(b.num != 0);
    /* a/b ÷ c/d = a*d / (b*c); cross-reduce to stay in int64 */
    int64_t g1 = gcd(i64_abs(a.num), i64_abs(b.num));
    int64_t g2 = gcd(a.den, b.den);
    return rat_normalize((a.num / g1) * (b.den / g2),
                         (a.den / g2) * (b.num / g1));
}

NiyahCspRat niyah_csp_rat_neg(NiyahCspRat a) {
    return (NiyahCspRat){-a.num, a.den};
}

int niyah_csp_rat_cmp(NiyahCspRat a, NiyahCspRat b) {
    /*
     * Compare a.num/a.den vs b.num/b.den without overflow.
     * Cross-multiply:  a.num * b.den  vs  b.num * a.den
     *
     * __int128 is a GNU/Clang extension, not ISO C, so -Wpedantic rejects it
     * and this tree builds with -Werror. Exactness is the entire purpose of
     * this function, so the extension stays and the diagnostic is silenced
     * for these two declarations only. __SIZEOF_INT128__ is the feature test;
     * __GNUC__ is not, because 32-bit and clang-cl targets define the vendor
     * macro without providing a 128-bit integer.
     */
#if defined(__SIZEOF_INT128__)
#  if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wpedantic"
#  endif
    __int128 lhs = (__int128)a.num * b.den;
    __int128 rhs = (__int128)b.num * a.den;
    if (lhs < rhs) return -1;
    if (lhs > rhs) return  1;
    return 0;
#  if defined(__GNUC__)
#    pragma GCC diagnostic pop
#  endif
#else
    /* No 128-bit integer: use double; exact only within a 53-bit mantissa */
    double lhsd = (double)a.num * (double)b.den;
    double rhsd = (double)b.num * (double)a.den;
    if (lhsd < rhsd) return -1;
    if (lhsd > rhsd) return  1;
    return 0;
#endif
}

bool niyah_csp_rat_eq(NiyahCspRat a, NiyahCspRat b) {
    return a.num == b.num && a.den == b.den;
}

double niyah_csp_rat_to_double(NiyahCspRat r) {
    return (double)r.num / (double)r.den;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * §1  System management
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

void niyah_csp_init(NiyahCspSystem *sys, uint32_t n_vars) {
    assert(n_vars <= NIYAH_CSP_MAX_VARS);
    memset(sys, 0, sizeof(*sys));
    sys->n_vars = n_vars;
}

bool niyah_csp_add(NiyahCspSystem *sys, NiyahCspConstraint c) {
    if (sys->count >= NIYAH_CSP_MAX_CONSTRAINTS) return false;
    sys->constraints[sys->count++] = c;
    return true;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * §2  Constraint evaluation
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* Evaluate LHS = sum(coeff[i] * values[var_id[i]]) */
static NiyahCspRat eval_lhs(const NiyahCspConstraint *c,
                            const NiyahCspRat *values)
{
    NiyahCspRat sum = {0, 1};
    for (uint32_t i = 0; i < c->n_terms; i++) {
        NiyahCspRat term = niyah_csp_rat_mul(c->coeffs[i],
                                              values[c->var_ids[i]]);
        sum = niyah_csp_rat_add(sum, term);
    }
    return sum;
}

/* Check if a single constraint is satisfied by values[] */
static bool constraint_sat(const NiyahCspConstraint *c,
                           const NiyahCspRat *values)
{
    NiyahCspRat lhs = eval_lhs(c, values);
    int cmp = niyah_csp_rat_cmp(lhs, c->bound);
    switch (c->op) {
        case NIYAH_CSP_LE: return cmp <= 0;
        case NIYAH_CSP_GE: return cmp >= 0;
        case NIYAH_CSP_EQ: return cmp == 0;
    }
    return false;
}

/* Check if all constraints are satisfied */
static bool all_satisfied(const NiyahCspSystem *sys,
                          const NiyahCspRat *values)
{
    for (uint32_t i = 0; i < sys->count; i++) {
        if (!constraint_sat(&sys->constraints[i], values))
            return false;
    }
    return true;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * §3  Bounds propagation solver
 *
 * For small systems: compute tight bounds for each variable
 * from single-variable constraints, then check multi-variable
 * constraints at midpoints. Iteratively tighten bounds.
 *
 * This is sufficient for the constraint sizes used in symbolic
 * reasoning (~2-8 variables, ~4-16 constraints).
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* Large default bounds for unconstrained variables */
#define DEFAULT_LO ((NiyahCspRat){-10000, 1})
#define DEFAULT_HI ((NiyahCspRat){ 10000, 1})

/* Extract bounds for a single variable from a 1-term constraint */
static bool extract_bound(const NiyahCspConstraint *c,
                          NiyahCspRat *lo, NiyahCspRat *hi)
{
    if (c->n_terms != 1) return false;

    NiyahCspRat coeff = c->coeffs[0];
    if (coeff.num == 0) return false;

    /* coeff * x OP bound  =>  x OP bound/coeff */
    NiyahCspRat limit = niyah_csp_rat_div(c->bound, coeff);
    bool flip = coeff.num < 0; /* inequality flips when dividing by negative */

    NiyahCspOp op = c->op;
    if (flip) {
        if (op == NIYAH_CSP_LE) op = NIYAH_CSP_GE;
        else if (op == NIYAH_CSP_GE) op = NIYAH_CSP_LE;
    }

    switch (op) {
        case NIYAH_CSP_LE:
            if (niyah_csp_rat_cmp(limit, *hi) < 0) *hi = limit;
            break;
        case NIYAH_CSP_GE:
            if (niyah_csp_rat_cmp(limit, *lo) > 0) *lo = limit;
            break;
        case NIYAH_CSP_EQ:
            *lo = limit;
            *hi = limit;
            break;
    }
    return true;
}

bool niyah_csp_feasible(const NiyahCspSystem *sys) {
    NiyahCspRat values[NIYAH_CSP_MAX_VARS];
    return niyah_csp_solve(sys, values);
}

bool niyah_csp_solve(const NiyahCspSystem *sys, NiyahCspRat *values) {
    if (sys->count == 0) {
        for (uint32_t i = 0; i < sys->n_vars; i++)
            values[i] = (NiyahCspRat){0, 1};
        return true;
    }

    /* Initialize bounds */
    NiyahCspRat lo[NIYAH_CSP_MAX_VARS], hi[NIYAH_CSP_MAX_VARS];
    for (uint32_t i = 0; i < sys->n_vars; i++) {
        lo[i] = DEFAULT_LO;
        hi[i] = DEFAULT_HI;
    }

    /* Pass 1: extract tight bounds from single-variable constraints */
    for (uint32_t ci = 0; ci < sys->count; ci++) {
        const NiyahCspConstraint *c = &sys->constraints[ci];
        if (c->n_terms == 1) {
            uint32_t vid = c->var_ids[0];
            extract_bound(c, &lo[vid], &hi[vid]);
        }
    }

    /* Check bound consistency */
    for (uint32_t i = 0; i < sys->n_vars; i++) {
        if (niyah_csp_rat_cmp(lo[i], hi[i]) > 0)
            return false; /* infeasible: lo > hi */
    }

    /* Start at midpoint of bounds */
    for (uint32_t i = 0; i < sys->n_vars; i++) {
        values[i] = niyah_csp_rat_div(
            niyah_csp_rat_add(lo[i], hi[i]),
            (NiyahCspRat){2, 1}
        );
    }

    /* If the simple midpoint works, we're done */
    if (all_satisfied(sys, values))
        return true;

    /*
     * Pass 2: iterative constraint propagation for multi-variable
     * constraints. For each violated constraint, try adjusting each
     * variable in turn to move toward satisfaction.
     */
    for (uint32_t iter = 0; iter < 200; iter++) {
        bool changed = false;

        for (uint32_t ci = 0; ci < sys->count; ci++) {
            const NiyahCspConstraint *c = &sys->constraints[ci];
            if (constraint_sat(c, values)) continue;

            for (uint32_t ti = 0; ti < c->n_terms; ti++) {
                if (c->coeffs[ti].num == 0) continue;
                uint32_t vid = c->var_ids[ti];

                /* Compute: rest = LHS - coeff*var, needed = (bound - rest) / coeff */
                NiyahCspRat cur_contrib = niyah_csp_rat_mul(
                    c->coeffs[ti], values[vid]);
                NiyahCspRat rest = niyah_csp_rat_sub(
                    eval_lhs(c, values), cur_contrib);
                NiyahCspRat needed_val = niyah_csp_rat_div(
                    niyah_csp_rat_sub(c->bound, rest), c->coeffs[ti]);

                /* For inequality: only adjust if it helps */
                if (c->op == NIYAH_CSP_LE) {
                    if (niyah_csp_rat_cmp(needed_val, values[vid]) >= 0)
                        continue;
                } else if (c->op == NIYAH_CSP_GE) {
                    if (niyah_csp_rat_cmp(needed_val, values[vid]) <= 0)
                        continue;
                }

                /* Clamp to variable bounds */
                if (niyah_csp_rat_cmp(needed_val, lo[vid]) < 0)
                    needed_val = lo[vid];
                if (niyah_csp_rat_cmp(needed_val, hi[vid]) > 0)
                    needed_val = hi[vid];

                if (!niyah_csp_rat_eq(needed_val, values[vid])) {
                    values[vid] = needed_val;
                    changed = true;
                }

                if (constraint_sat(c, values)) break;
            }
        }

        if (all_satisfied(sys, values))
            return true;

        if (!changed)
            break;
    }

    return all_satisfied(sys, values);
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * §4  Smoke test
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

#define CSP_PASS(cond, label) do { \
    if (cond) { pass++; fprintf(stderr, "  [PASS] %s\n", label); } \
    else      { fail++; fprintf(stderr, "  [FAIL] %s\n", label); } \
} while(0)

/* Helper: build a 1-term constraint: coeff * var_id OP bound */
static NiyahCspConstraint csp1(int64_t coeff, uint32_t var_id,
                               NiyahCspOp op, int64_t bound)
{
    NiyahCspConstraint c;
    memset(&c, 0, sizeof(c));
    c.coeffs[0] = niyah_csp_rat(coeff, 1);
    c.var_ids[0] = var_id;
    c.n_terms = 1;
    c.op = op;
    c.bound = niyah_csp_rat(bound, 1);
    return c;
}

/* Helper: build a 2-term constraint: c1*v1 + c2*v2 OP bound */
static NiyahCspConstraint csp2(int64_t c1, uint32_t v1,
                               int64_t c2, uint32_t v2,
                               NiyahCspOp op, int64_t bound)
{
    NiyahCspConstraint c;
    memset(&c, 0, sizeof(c));
    c.coeffs[0] = niyah_csp_rat(c1, 1);
    c.var_ids[0] = v1;
    c.coeffs[1] = niyah_csp_rat(c2, 1);
    c.var_ids[1] = v2;
    c.n_terms = 2;
    c.op = op;
    c.bound = niyah_csp_rat(bound, 1);
    return c;
}

int niyah_csp_smoke(void) {
    int pass = 0, fail = 0;

    fprintf(stderr, "\n+--------------------------------------+\n");
    fprintf(stderr, "|  NIYAH Constraint Solver Smoke Test  |\n");
    fprintf(stderr, "+--------------------------------------+\n");

    /* §4.1 — Rational arithmetic basics */
    {
        NiyahCspRat a = niyah_csp_rat(1, 3);
        NiyahCspRat b = niyah_csp_rat(1, 6);
        NiyahCspRat sum = niyah_csp_rat_add(a, b);
        CSP_PASS(sum.num == 1 && sum.den == 2, "1/3 + 1/6 = 1/2");

        NiyahCspRat diff = niyah_csp_rat_sub(a, b);
        CSP_PASS(diff.num == 1 && diff.den == 6, "1/3 - 1/6 = 1/6");

        NiyahCspRat prod = niyah_csp_rat_mul(a, b);
        CSP_PASS(prod.num == 1 && prod.den == 18, "1/3 * 1/6 = 1/18");

        NiyahCspRat quot = niyah_csp_rat_div(a, b);
        CSP_PASS(quot.num == 2 && quot.den == 1, "(1/3) / (1/6) = 2");
    }

    /* §4.2 — Normalization */
    {
        NiyahCspRat r = niyah_csp_rat(6, -9);
        CSP_PASS(r.num == -2 && r.den == 3, "6/-9 normalizes to -2/3");

        NiyahCspRat z = niyah_csp_rat(0, 5);
        CSP_PASS(z.num == 0 && z.den == 1, "0/5 normalizes to 0/1");
    }

    /* §4.3 — Comparison */
    {
        NiyahCspRat a = niyah_csp_rat(1, 3);
        NiyahCspRat b = niyah_csp_rat(1, 2);
        CSP_PASS(niyah_csp_rat_cmp(a, b) < 0, "1/3 < 1/2");
        CSP_PASS(niyah_csp_rat_cmp(b, a) > 0, "1/2 > 1/3");
        CSP_PASS(niyah_csp_rat_cmp(a, a) == 0, "1/3 == 1/3");
    }

    /* §4.4 — Simple bounds: 0 <= x <= 10 */
    {
        NiyahCspSystem sys;
        niyah_csp_init(&sys, 1);
        niyah_csp_add(&sys, csp1(1, 0, NIYAH_CSP_GE, 0));   /* x >= 0 */
        niyah_csp_add(&sys, csp1(1, 0, NIYAH_CSP_LE, 10));  /* x <= 10 */

        NiyahCspRat vals[1];
        bool ok = niyah_csp_solve(&sys, vals);
        CSP_PASS(ok, "0 <= x <= 10 is feasible");
        CSP_PASS(niyah_csp_rat_cmp(vals[0], niyah_csp_rat(0, 1)) >= 0
              && niyah_csp_rat_cmp(vals[0], niyah_csp_rat(10, 1)) <= 0,
                 "solution x in [0, 10]");
        fprintf(stderr, "  x = %ld/%ld (%.2f)\n",
                (long)vals[0].num, (long)vals[0].den,
                niyah_csp_rat_to_double(vals[0]));
    }

    /* §4.5 — Infeasible: x >= 5 AND x <= 3 */
    {
        NiyahCspSystem sys;
        niyah_csp_init(&sys, 1);
        niyah_csp_add(&sys, csp1(1, 0, NIYAH_CSP_GE, 5));
        niyah_csp_add(&sys, csp1(1, 0, NIYAH_CSP_LE, 3));

        CSP_PASS(!niyah_csp_feasible(&sys), "x >= 5 AND x <= 3 infeasible");
    }

    /* §4.6 — Two variables: x + y <= 10, x >= 0, y >= 0 */
    {
        NiyahCspSystem sys;
        niyah_csp_init(&sys, 2);
        niyah_csp_add(&sys, csp1(1, 0, NIYAH_CSP_GE, 0));   /* x >= 0 */
        niyah_csp_add(&sys, csp1(1, 1, NIYAH_CSP_GE, 0));   /* y >= 0 */
        niyah_csp_add(&sys, csp2(1, 0, 1, 1, NIYAH_CSP_LE, 10)); /* x+y <= 10 */

        NiyahCspRat vals[2];
        bool ok = niyah_csp_solve(&sys, vals);
        CSP_PASS(ok, "x+y <= 10, x>=0, y>=0 feasible");
        if (ok) {
            NiyahCspRat sum = niyah_csp_rat_add(vals[0], vals[1]);
            CSP_PASS(niyah_csp_rat_cmp(sum, niyah_csp_rat(10, 1)) <= 0,
                     "solution: x+y <= 10");
            fprintf(stderr, "  x=%.2f y=%.2f sum=%.2f\n",
                    niyah_csp_rat_to_double(vals[0]),
                    niyah_csp_rat_to_double(vals[1]),
                    niyah_csp_rat_to_double(sum));
        }
    }

    /* §4.7 — Equality constraint: x = 7 */
    {
        NiyahCspSystem sys;
        niyah_csp_init(&sys, 1);
        niyah_csp_add(&sys, csp1(1, 0, NIYAH_CSP_EQ, 7));

        NiyahCspRat vals[1];
        bool ok = niyah_csp_solve(&sys, vals);
        CSP_PASS(ok, "x = 7 feasible");
        CSP_PASS(ok && niyah_csp_rat_eq(vals[0], niyah_csp_rat(7, 1)),
                 "solution: x = 7");
    }

    /* §4.8 — Empty system: always feasible */
    {
        NiyahCspSystem sys;
        niyah_csp_init(&sys, 3);
        CSP_PASS(niyah_csp_feasible(&sys), "empty system feasible");
    }

    /* §4.9 — Negative coefficient: -x <= -3  (i.e. x >= 3) */
    {
        NiyahCspSystem sys;
        niyah_csp_init(&sys, 1);
        niyah_csp_add(&sys, csp1(-1, 0, NIYAH_CSP_LE, -3)); /* -x <= -3 */
        niyah_csp_add(&sys, csp1(1, 0, NIYAH_CSP_LE, 8));   /* x <= 8 */

        NiyahCspRat vals[1];
        bool ok = niyah_csp_solve(&sys, vals);
        CSP_PASS(ok, "-x <= -3, x <= 8 feasible");
        CSP_PASS(ok && niyah_csp_rat_cmp(vals[0], niyah_csp_rat(3, 1)) >= 0
              && niyah_csp_rat_cmp(vals[0], niyah_csp_rat(8, 1)) <= 0,
                 "solution: 3 <= x <= 8");
    }

    fprintf(stderr, "\n  Results: %d passed, %d failed\n\n", pass, fail);
    return fail;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * §5  Standalone test entry point
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

#ifdef CSP_STANDALONE_TEST
int main(void) {
    int failed = niyah_csp_smoke();
    if (failed == 0)
        printf("CSP SMOKE PASS - 0 failed\n");
    else
        printf("CSP SMOKE FAIL - %d failed\n", failed);
    return failed;
}
#endif

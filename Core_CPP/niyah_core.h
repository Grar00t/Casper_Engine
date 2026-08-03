/*
 * niyah_core.h — NIYAH Inference Engine v3.0
 *
 * Zero external dependencies. C99 clean. C++17 compatible.
 * Targets: x86_64 (AVX2+FMA), aarch64 (NEON), scalar fallback.
 *
 * ABI version: 0x0005
 */
#ifndef NIYAH_CORE_H
#define NIYAH_CORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Constants
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
#define NIYAH_MAGIC     UINT32_C(0x4E595148)   /* "NYQH" */
#define NIYAH_VER       UINT32_C(0x0005)
#define NIYAH_MAX_CTX   UINT32_C(8192)
#define NIYAH_MAX_VOCAB UINT32_C(131072)

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Config — serialised verbatim to .bin header
 * Changing any field requires bumping NIYAH_VER.
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t embed_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;    /* GQA: kv heads (≤ n_heads) */
    uint32_t n_layers;
    uint32_t ffn_mult;      /* ffn_hidden = embed_dim * ffn_mult */
    uint32_t vocab_size;
    uint32_t ctx_len;
    float    rope_theta;
    float    rms_eps;
    uint32_t flags;         /* reserved, zero */
    uint8_t  _pad[16];      /* pad to 64 bytes total */
} NiyahConfig;             /* sizeof must be 64 */

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Layer weight layout (all pointers into pool)
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
typedef struct {
    float *wq;          /* [embed × embed]           Q projection */
    float *wk;          /* [kv_dim × embed]          K projection */
    float *wv;          /* [kv_dim × embed]          V projection */
    float *wo;          /* [embed × embed]           O projection */
    float *w_gate;      /* [ffn_hidden × embed]      SwiGLU gate  */
    float *w_up;        /* [ffn_hidden × embed]      SwiGLU up    */
    float *w_down;      /* [embed × ffn_hidden]      FFN down     */
    float *rms_att;     /* [embed]                   pre-attn norm*/
    float *rms_ffn;     /* [embed]                   pre-ffn norm */
} NiyahLayer;

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Model — single-pool allocation
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
typedef struct {
    NiyahConfig  cfg;
    NiyahLayer  *layers;       /* array[n_layers] in pool */

    float *token_embed;        /* [vocab × embed]  */
    float *rms_final;          /* [embed]          */
    float *lm_head;            /* [vocab × embed]  */

    /*
     * KV cache — head-major layout (best attention-loop locality):
     *   kv_k[layer][head][seq_pos][head_dim]
     *   kv_v[layer][head][seq_pos][head_dim]
     * stride: layer_stride = n_kv_heads * ctx_len * head_dim
     */
    float *kv_k;
    float *kv_v;

    /* Run-state scratch buffer (points inside pool) */
    float *scratch;
    float *_logits;            /* inside scratch, model-local */

    /* Single pool — one owner, zero fragmentation */
    void  *_pool;
    size_t _pool_bytes;

    /* Derived constants (computed at alloc, never serialised) */
    uint32_t head_dim;         /* embed_dim / n_heads     */
    uint32_t kv_dim;           /* n_kv_heads * head_dim   */
    uint32_t ffn_dim;          /* embed_dim * ffn_mult    */
} NiyahModel;

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Adam optimizer state
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
typedef struct {
    float   *m;            /* 1st moment  (same count as weights) */
    float   *v;            /* 2nd moment  */
    uint32_t step;
    float    lr;           /* learning rate          default 3e-4 */
    float    beta1;        /* momentum decay         default 0.9  */
    float    beta2;        /* variance decay         default 0.999*/
    float    eps;          /* numerical stability    default 1e-8 */
    float    wd;           /* weight decay           default 0.01 */
    size_t   n_weights;    /* total floats managed              */
} NiyahAdam;

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Sampler
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
typedef struct {
    float temperature;
    float top_p;
    uint64_t seed;
} NiyahSampler;

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Public API
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* Allocation / free */
NiyahModel *niyah_alloc(const NiyahConfig *cfg);
void        niyah_free (NiyahModel *m);

/* Persistence — returns 0 on success, -1 I/O error, -2 version mismatch */
int  niyah_save(const NiyahModel *m, const char *path);
int  niyah_load(NiyahModel **out,    const char *path);

/* Inference — KV-cache written at pos; pos must be monotonically increasing.
 * Returns pointer to logit buffer (owned by model, valid until next call).
 * Call with pos=0 to reset generation. */
float *niyah_forward(NiyahModel *m, uint32_t token, uint32_t pos);

/* Sampling */
uint32_t niyah_sample(const float *logits, uint32_t vocab_size,
                      NiyahSampler *s);

/* Training — output-layer Adam step; returns mean cross-entropy loss */
float niyah_train_step(NiyahModel *m, NiyahAdam *opt,
                       const uint32_t *tokens, uint32_t n);

/* Adam lifecycle */
NiyahAdam *niyah_adam_alloc(const NiyahModel *m);
void       niyah_adam_free (NiyahAdam *opt);

/* Introspection */
const char *niyah_simd_name(void);      /* "AVX2+FMA" | "NEON" | "Scalar" */
size_t      niyah_param_count(const NiyahModel *m);

/* Smoke test — returns failed-assertion count (0 = all pass) */
int niyah_smoke(void);

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Hybrid neuro-symbolic API
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* Forward declaration — full types in rule_parser.h */
typedef struct NiyahRuleKBTag NiyahRuleKBOpaque;

/* Hybrid generation options */
typedef struct {
    void        *rules;         /* NiyahRuleKB* or NULL for pure neural */
    uint32_t     max_retries;   /* re-sample attempts on violation (default 3) */
    bool         generate_proof;/* compute proof hash */
} NiyahHybridOpts;

/* Why generation produced — or failed to produce — an output.
 *
 * A total rejection used to be papered over with the literal string
 * "[Output rejected]", which is indistinguishable from a model that
 * genuinely emitted that text. The verdict is now reported out of
 * band, so a caller can tell verification failure from content. */
typedef enum {
    NIYAH_HYBRID_OK = 0,            /* passed KHZ_Q and all rules       */
    NIYAH_HYBRID_RULE_REPLACED,     /* a rule substituted the output    */
    NIYAH_HYBRID_RULE_REJECTED,     /* rule verdict REJECTED, no text   */
    NIYAH_HYBRID_KHZ_INCOHERENT,    /* every attempt failed the SVD gate*/
    NIYAH_HYBRID_ERR_DECODE         /* tokenizer produced nothing usable*/
} NiyahHybridOutcome;

typedef struct {
    NiyahHybridOutcome outcome;
    uint32_t    attempts;         /* attempts actually consumed          */
    float       khz_energy;       /* energy_preserved of the last attempt*/
    float       khz_penalty_nasl; /* nasl penalty of the last attempt    */
    const char *rule_violation;   /* borrowed from the rule KB, or NULL  */
} NiyahHybridStatus;

/*
 * Generate text with optional symbolic verification.
 *
 * Returns a malloc'd string (caller frees) for NIYAH_HYBRID_OK and
 * NIYAH_HYBRID_RULE_REPLACED. Returns NULL for every rejection
 * outcome: NULL means "no output survived verification" and must not
 * be smoothed over with substitute text.
 *
 * status_out, when non-NULL, always receives the outcome — including
 * on success. proof_out, when non-NULL, receives a 32-byte SHA-256
 * hash, and is only written when text was actually produced.
 * If no rules are provided, runs pure neural generation.
 */
char *niyah_hybrid_generate(NiyahModel *m, const char *prompt,
                            const NiyahHybridOpts *opts,
                            NiyahSampler *sampler,
                            uint8_t proof_out[32],
                            NiyahHybridStatus *status_out);

/* Stable, human-readable outcome name. Static storage, never NULL. */
const char *niyah_hybrid_outcome_str(NiyahHybridOutcome o);

#ifdef __cplusplus
}
#endif
#endif /* NIYAH_CORE_H */

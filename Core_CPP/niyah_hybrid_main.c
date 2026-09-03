/*
 * niyah_hybrid_main.c — NIYAH Hybrid Neuro-Symbolic CLI
 */

#include "niyah_core.h"
#include "rule_parser.h"
#include "proof_generator.h"
#include "khz_q_svd.h"
#include "casper_rag.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

void tokenizer_init(void);
uint32_t tokenizer_encode(const char *text, uint32_t *tokens, uint32_t max_len);
char *tokenizer_decode(const uint32_t *tokens, uint32_t n);
void tokenizer_free(void);

int niyah_sym_smoke(void);
int niyah_csp_smoke(void);
int niyah_rule_smoke(void);
int niyah_proof_smoke(void);

static uint32_t clamp_token(uint32_t token, uint32_t vocab) {
    return vocab ? token % vocab : 0u;
}

static uint32_t generate_tokens(NiyahModel *m, const uint32_t *prompt_tokens,
                                uint32_t prompt_len, uint32_t *out_tokens,
                                uint32_t max_out, NiyahSampler *sampler)
{
    if (!m || !prompt_tokens || !out_tokens || !sampler || m->cfg.vocab_size == 0u) return 0u;
    uint32_t ctx = m->cfg.ctx_len;
    if (ctx == 0u || prompt_len == 0u || prompt_len > ctx) return 0u;

    uint32_t pos = 0u;
    uint32_t n_out = 0u;
    for (uint32_t i = 0u; i < prompt_len && pos < ctx; ++i, ++pos)
        (void)niyah_forward(m, clamp_token(prompt_tokens[i], m->cfg.vocab_size), pos);

    uint32_t last_tok = clamp_token(prompt_tokens[prompt_len - 1u], m->cfg.vocab_size);
    for (uint32_t i = 0u; i < max_out && pos < ctx; ++i, ++pos) {
        float *logits = niyah_forward(m, last_tok, pos);
        if (!logits) break;
        uint32_t tok = clamp_token(niyah_sample(logits, m->cfg.vocab_size, sampler), m->cfg.vocab_size);
        if (tok == 1u) break;
        if (n_out >= max_out) break;
        out_tokens[n_out++] = tok;
        last_tok = tok;
    }
    return n_out;
}

char *niyah_hybrid_generate(NiyahModel *m, const char *prompt,
                            const NiyahHybridOpts *opts,
                            NiyahSampler *sampler,
                            uint8_t proof_out[32])
{
    if (!m || !prompt || !sampler || m->cfg.vocab_size == 0u || m->cfg.ctx_len == 0u) return NULL;

    tokenizer_init();
    uint32_t prompt_tokens[512];
    uint32_t prompt_len = tokenizer_encode(prompt, prompt_tokens, 512u);
    if (prompt_len == 0u || prompt_len > m->cfg.ctx_len) {
        tokenizer_free();
        return NULL;
    }

    for (uint32_t i = 0u; i < prompt_len; ++i)
        prompt_tokens[i] = clamp_token(prompt_tokens[i], m->cfg.vocab_size);

    uint32_t max_retries = (opts && opts->max_retries > 0u) ? opts->max_retries : 3u;
    NiyahRuleKB *rules = opts ? (NiyahRuleKB *)opts->rules : NULL;
    bool generate_proof = opts ? opts->generate_proof : false;
    uint32_t out_tokens[512];
    char *result = NULL;

    for (uint32_t attempt = 0u; attempt <= max_retries; ++attempt) {
        if (attempt > 0u) sampler->seed += UINT64_C(12345) * attempt;
        uint32_t n_out = generate_tokens(m, prompt_tokens, prompt_len, out_tokens, 512u, sampler);
        char *text = tokenizer_decode(out_tokens, n_out);
        if (!text) continue;

        KHZQ_Result khz = khz_q_verify_output(text, 0.85f);
        if (!khz.is_coherent) { free(text); continue; }

        if (!rules) { result = text; break; }
        const char *violation = niyah_rule_check(rules, prompt, text);
        if (!violation) { result = text; break; }

        if (attempt == max_retries) {
            if (strcmp(violation, "REJECTED") == 0) {
                free(text);
                result = (char *)malloc(64u);
                if (result) (void)snprintf(result, 64u, "[Output rejected by rules]");
            } else {
                size_t len = strlen(violation) + 1u;
                char *replacement = (char *)malloc(len);
                if (replacement) memcpy(replacement, violation, len);
                free(text);
                result = replacement;
            }
            break;
        }
        free(text);
    }

    if (!result) {
        result = (char *)malloc(32u);
        if (result) (void)snprintf(result, 32u, "[Output rejected]");
    }

    if (proof_out) memset(proof_out, 0, 32u);
    if (proof_out && generate_proof && result)
        niyah_proof_generate(prompt, result, NULL, proof_out);

    tokenizer_free();
    return result;
}

static int run_all_smoke(void) {
    int total_fail = 0;
    /* niyah_smoke() was merged into Niyah.Engine / NiyahKernel and is defined
     * nowhere here; the core self-check is build/niyah. */
    total_fail += niyah_sym_smoke();
    total_fail += niyah_csp_smoke();
    total_fail += niyah_rule_smoke();
    total_fail += niyah_proof_smoke();
    {
        int pass = 0, fail = 0;
#define KHZQ_PASS(cond, label) do { if (cond) { ++pass; } else { ++fail; (void)fprintf(stderr, "[FAIL] %s\n", label); } } while (0)
        KHZQ_Result r1 = khz_q_verify_output("bismillah bismillah bismillah bismillah", 0.85f);
        KHZQ_PASS(r1.energy_preserved >= 0.85f, "coherent text energy");
        KHZQ_Result r2 = khz_q_verify_output("", 0.85f);
        KHZQ_PASS(!r2.is_coherent, "empty text rejected");
        KHZQ_Result r3 = khz_q_verify_output("test", 1.5f);
        KHZQ_PASS(r3.chi_e > 0 && r3.chi_e <= KHZ_MAX_N, "chi_e range");
        KHZQ_Result r4 = khz_q_verify_output("\xd8\xa8\xd8\xb3\xd9\x85 \xd8\xa7\xd9\x84\xd9\x84\xd9\x87", 0.80f);
        KHZQ_PASS(r4.chi_e >= 1, "Arabic UTF-8");
        total_fail += fail;
#undef KHZQ_PASS
        (void)pass;
    }
    {
        NiyahConfig cfg = {.magic=NIYAH_MAGIC,.version=NIYAH_VER,.embed_dim=64,.n_heads=4,.n_kv_heads=4,.n_layers=2,.ffn_mult=4,.vocab_size=256,.ctx_len=32,.rope_theta=10000.f,.rms_eps=1e-5f};
        NiyahModel *m = niyah_alloc(&cfg);
        if (!m) ++total_fail;
        else {
            float *wp = (float *)m->_pool;
            size_t nw = niyah_param_count(m);
            for (size_t i=0;i<nw;++i) wp[i]=((float)(i%37)-18.f)*0.005f;
            for (uint32_t l=0;l<cfg.n_layers;++l) for (uint32_t j=0;j<cfg.embed_dim;++j) { m->layers[l].rms_att[j]=1.f; m->layers[l].rms_ffn[j]=1.f; }
            for (uint32_t j=0;j<cfg.embed_dim;++j) m->rms_final[j]=1.f;
            NiyahSampler s={.temperature=0.8f,.top_p=0.9f,.seed=42};
            NiyahHybridOpts opts={.rules=NULL,.max_retries=0,.generate_proof=false};
            char *out=niyah_hybrid_generate(m,"hello",&opts,&s,NULL); if(!out) ++total_fail; free(out);
            niyah_free(m);
        }
    }
    return total_fail;
}

static void rag_loop(RagBackend backend, NiyahRuleKB *rules) {
    char line[2048];
    while (1) {
        (void)printf("[RAG] > "); (void)fflush(stdout);
        if (!fgets(line,sizeof(line),stdin)) break;
        size_t len=strlen(line); while(len && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len]='\0';
        if (!len) continue;
        if (!strcmp(line,"quit") || !strcmp(line,"exit")) break;
        RagCtx *ctx=casper_rag_query(line,backend,NULL);
        if (!ctx) { (void)printf("[RAG] query failed\n"); continue; }
        (void)printf("sources=%d confidence=%.3f elapsed=%u\n",ctx->n_results,(double)ctx->confidence,ctx->elapsed_ms);
        for(int i=0;i<ctx->n_results;++i) (void)printf("[%d] %s\n    %s\n",i+1,ctx->results[i].title,ctx->results[i].url);
        if(rules && ctx->context[0]) { const char *v=niyah_rule_check(rules,line,ctx->context); if(v) (void)printf("rule=%s\n",v); }
        casper_rag_free(ctx);
    }
}

static void interactive_loop(NiyahModel *m, NiyahRuleKB *rules) {
    NiyahSampler sampler={.temperature=0.8f,.top_p=0.9f,.seed=12345};
    NiyahHybridOpts opts={.rules=rules,.max_retries=3,.generate_proof=false};
    char line[4096];
    while(1){
        (void)printf("> "); (void)fflush(stdout);
        if(!fgets(line,sizeof(line),stdin))break;
        size_t len=strlen(line); while(len&&(line[len-1]=='\n'||line[len-1]=='\r'))line[--len]='\0';
        if(!len)continue; if(!strcmp(line,"quit")||!strcmp(line,"exit"))break;
        char *response=niyah_hybrid_generate(m,line,&opts,&sampler,NULL);
        if(response){(void)printf("%s\n",response);free(response);} else (void)printf("[no response]\n");
        sampler.seed+=7919u;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { (void)fprintf(stderr,"usage: %s --smoke | --rag | --model <file> | --interactive\n",argv[0]); return 3; }
    if (!strcmp(argv[1],"--smoke")) return run_all_smoke();
    if (!strcmp(argv[1],"--rag")) { rag_loop(RAG_BACKEND_DDG,NULL); return 0; }
    if (!strcmp(argv[1],"--model")) {
        if (argc < 3) return 3;
        NiyahModel *m=NULL;
        int rc=niyah_load(&m,argv[2]);
        if(rc!=0 || !m) return 1;
        interactive_loop(m,NULL);
        niyah_free(m); return 0;
    }
    if (!strcmp(argv[1],"--interactive")) {
        NiyahConfig cfg={.magic=NIYAH_MAGIC,.version=NIYAH_VER,.embed_dim=128,.n_heads=8,.n_kv_heads=8,.n_layers=4,.ffn_mult=4,.vocab_size=8192,.ctx_len=64,.rope_theta=10000.f,.rms_eps=1e-5f};
        NiyahModel *m=niyah_alloc(&cfg); if(!m)return 1;
        interactive_loop(m,NULL); niyah_free(m); return 0;
    }
    return 3;
}

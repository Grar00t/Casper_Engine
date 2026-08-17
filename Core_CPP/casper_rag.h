/*
 * casper_rag.h — Retrieval-Augmented Generation pipe for Casper
 * C11, zero external dependencies in the public API.
 */
#ifndef CASPER_RAG_H
#define CASPER_RAG_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAG_MAX_RESULTS 8
#define RAG_URL_MAX 512
#define RAG_TITLE_MAX 256
#define RAG_SNIPPET_MAX 1024
#define RAG_CONTEXT_MAX 8192
#define RAG_TRACE_MAX 32
#define RAG_TIMEOUT_MS 7000

typedef struct {
    char url[RAG_URL_MAX];
    char title[RAG_TITLE_MAX];
    char snippet[RAG_SNIPPET_MAX];
    float score;
    uint8_t sha256[32];
} RagResult;

typedef enum {
    TRACE_PARSE = 0,
    TRACE_SEARCH = 1,
    TRACE_FETCH = 2,
    TRACE_RANK = 3,
    TRACE_CONTEXT = 4,
    TRACE_SYMBOLIC = 5,
    TRACE_COMPOSE = 6,
    TRACE_WARN = 7
} TraceKind;

typedef struct {
    TraceKind kind;
    uint32_t elapsed_ms;
    float confidence;
    char detail[256];
} TraceStep;

typedef struct {
    char query[512];
    RagResult results[RAG_MAX_RESULTS];
    int n_results;
    char context[RAG_CONTEXT_MAX];
    TraceStep trace[RAG_TRACE_MAX];
    int n_steps;
    float confidence;
    bool contradiction;
    uint8_t chain_hash[32];
    uint32_t elapsed_ms;
} RagCtx;

typedef enum {
    RAG_BACKEND_DDG = 0,
    RAG_BACKEND_SEARXNG = 1,
    RAG_BACKEND_BING = 2
} RagBackend;

RagCtx *casper_rag_query(const char *query,
                         RagBackend backend,
                         const char *rules_path);
void casper_rag_free(RagCtx *ctx);
char *casper_rag_to_json(const RagCtx *ctx);
bool casper_rag_online(RagBackend backend);

#ifdef __cplusplus
}
#endif
#endif

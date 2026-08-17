/* casper_rag.c - deterministic RAG transport/parser/ranker. C11. */
#include "casper_rag.h"
#include "proof_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <time.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <winhttp.h>
#endif

#define HBUF_MAX (256u * 1024u)

typedef struct { char *buf; size_t len; size_t cap; } HBuf;

static uint32_t ms_now(void) {
#ifdef _WIN32
    return (uint32_t)GetTickCount();
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return (uint32_t)((double)clock() * 1000.0 / (double)CLOCKS_PER_SEC);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u);
#endif
}

static HBuf *hbuf_new(void) {
    HBuf *b = (HBuf *)calloc(1u, sizeof(*b));
    if (!b) return NULL;
    b->cap = 8192u;
    b->buf = (char *)malloc(b->cap);
    if (!b->buf) { free(b); return NULL; }
    b->buf[0] = '\0';
    return b;
}

static void hbuf_free(HBuf *b) { if (b) { free(b->buf); free(b); } }

static int hbuf_app(HBuf *b, const char *data, size_t n) {
    if (!b || (!data && n)) return -1;
    if (b->len > HBUF_MAX - 1u || n > HBUF_MAX - b->len - 1u) return -1;
    size_t need = b->len + n + 1u;
    if (need > b->cap) {
        size_t next = b->cap;
        while (next < need && next < HBUF_MAX) {
            if (next > HBUF_MAX / 2u) { next = HBUF_MAX; break; }
            next *= 2u;
        }
        if (next < need) return -1;
        char *p = (char *)realloc(b->buf, next);
        if (!p) return -1;
        b->buf = p; b->cap = next;
    }
    if (n) memcpy(b->buf + b->len, data, n);
    b->len += n; b->buf[b->len] = '\0';
    return 0;
}

static void url_enc(const char *in, char *out, size_t max) {
    static const char hex[] = "0123456789ABCDEF";
    size_t o = 0u;
    if (!out || max == 0u) return;
    if (!in) { out[0] = '\0'; return; }
    while (*in && o + 3u < max) {
        unsigned char c = (unsigned char)*in++;
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out[o++] = (char)c;
        else { out[o++] = '%'; out[o++] = hex[c >> 4]; out[o++] = hex[c & 0x0Fu]; }
    }
    out[o] = '\0';
}

#ifdef _WIN32
static HBuf *http_get(const char *host, const char *path) {
    HBuf *out = hbuf_new();
    if (!out) return NULL;
    int host_wlen = MultiByteToWideChar(CP_UTF8, 0, host, -1, NULL, 0);
    int path_wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    if (host_wlen <= 0 || path_wlen <= 0) { hbuf_free(out); return NULL; }
    wchar_t *whost = (wchar_t *)calloc((size_t)host_wlen, sizeof(wchar_t));
    wchar_t *wpath = (wchar_t *)calloc((size_t)path_wlen, sizeof(wchar_t));
    if (!whost || !wpath) { free(whost); free(wpath); hbuf_free(out); return NULL; }
    if (!MultiByteToWideChar(CP_UTF8, 0, host, -1, whost, host_wlen) ||
        !MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, path_wlen)) {
        free(whost); free(wpath); hbuf_free(out); return NULL;
    }
    HINTERNET session = WinHttpOpen(L"Casper/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) { free(whost); free(wpath); hbuf_free(out); return NULL; }
    HINTERNET connection = WinHttpConnect(session, whost, INTERNET_DEFAULT_HTTPS_PORT, 0);
    if (!connection) { WinHttpCloseHandle(session); free(whost); free(wpath); hbuf_free(out); return NULL; }
    HINTERNET request = WinHttpOpenRequest(connection, L"GET", wpath, NULL, WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
    free(whost); free(wpath);
    if (!request) { WinHttpCloseHandle(connection); WinHttpCloseHandle(session); hbuf_free(out); return NULL; }
    WinHttpAddRequestHeaders(request,
        L"User-Agent: Casper/1.0\r\nAccept: text/html,application/xhtml+xml\r\n",
        (DWORD)-1, WINHTTP_ADDREQ_FLAG_ADD);
    DWORD timeout = RAG_TIMEOUT_MS;
    WinHttpSetTimeouts(request, timeout, timeout, timeout, timeout);
    BOOL ok = WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
        WINHTTP_NO_REQUEST_DATA, 0, 0, 0) && WinHttpReceiveResponse(request, NULL);
    if (ok) {
        DWORD available = 0;
        while (WinHttpQueryDataAvailable(request, &available) && available > 0) {
            char chunk[8192]; DWORD read = 0;
            DWORD want = available < (DWORD)sizeof(chunk) ? available : (DWORD)sizeof(chunk);
            if (!WinHttpReadData(request, chunk, want, &read) || read == 0) break;
            if (hbuf_app(out, chunk, (size_t)read) != 0) { ok = FALSE; break; }
        }
    }
    WinHttpCloseHandle(request); WinHttpCloseHandle(connection); WinHttpCloseHandle(session);
    if (!ok || out->len == 0u) { hbuf_free(out); return NULL; }
    return out;
}
#endif

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void url_decode(char *s) {
    if (!s) return;
    char *r = s, *w = s;
    while (*r) {
        if (r[0] == '%' && r[1] && r[2]) {
            int hi = hexval(r[1]), lo = hexval(r[2]);
            if (hi >= 0 && lo >= 0) { *w++ = (char)((hi << 4) | lo); r += 3; continue; }
        }
        *w++ = (*r == '+') ? ' ' : *r; ++r;
    }
    *w = '\0';
}

static void strip_tags(const char *html, char *out, size_t max) {
    size_t n = 0u; int in_tag = 0;
    if (!out || !max) return;
    out[0] = '\0'; if (!html) return;
    for (const char *p = html; *p && n + 1u < max; ++p) {
        if (*p == '<') { in_tag = 1; continue; }
        if (*p == '>') { in_tag = 0; if (n && out[n-1] != ' ') out[n++] = ' '; continue; }
        if (in_tag) continue;
        if (*p == '&') {
            if (!strncmp(p, "&amp;", 5)) { out[n++]='&'; p+=4; }
            else if (!strncmp(p, "&lt;", 4)) { out[n++]='<'; p+=3; }
            else if (!strncmp(p, "&gt;", 4)) { out[n++]='>'; p+=3; }
            else if (!strncmp(p, "&quot;", 6)) { out[n++]='"'; p+=5; }
            else if (!strncmp(p, "&#39;", 5)) { out[n++]='\''; p+=4; }
            else if (!strncmp(p, "&nbsp;", 6)) { out[n++]=' '; p+=5; }
            else out[n++]='&';
            continue;
        }
        if (*p == '\r' || *p == '\n' || *p == '\t') { if (n && out[n-1] != ' ') out[n++]=' '; }
        else out[n++] = *p;
    }
    while (n && out[n-1] == ' ') --n;
    out[n] = '\0';
}

static int parse_ddg(const char *html, RagResult *res, int max) {
    int n = 0; const char *p = html;
    while (n < max && p && *p) {
        const char *h = strstr(p, "result__a\" href=\"");
        if (!h) { h = strstr(p, "/l/?uddg="); if (!h) break; h += 9; }
        else h += 17;
        const char *end = strchr(h, '"');
        const char *amp = strchr(h, '&');
        if (amp && (!end || amp < end)) end = amp;
        if (!end) break;
        size_t len = (size_t)(end - h); if (len >= RAG_URL_MAX) len = RAG_URL_MAX - 1u;
        memcpy(res[n].url, h, len); res[n].url[len] = '\0';
        url_decode(res[n].url);
        p = end;
        const char *gt = strchr(p, '>'); if (!gt) break;
        const char *title_end = strstr(gt + 1, "</a>"); if (!title_end) title_end = strstr(gt + 1, "</");
        if (title_end) {
            char raw[RAG_TITLE_MAX * 2u]; size_t tl = (size_t)(title_end - (gt + 1));
            if (tl >= sizeof(raw)) tl = sizeof(raw)-1u;
            memcpy(raw, gt + 1, tl); raw[tl] = '\0'; strip_tags(raw, res[n].title, sizeof(res[n].title)); p = title_end;
        } else res[n].title[0] = '\0';
        const char *sn = strstr(p, "result__snippet");
        if (sn) {
            const char *s0 = strchr(sn, '>');
            if (s0) { ++s0; const char *s1 = strstr(s0, "</");
                if (s1) {
                    char raw[RAG_SNIPPET_MAX * 2u]; size_t sl=(size_t)(s1-s0);
                    if (sl >= sizeof(raw)) sl=sizeof(raw)-1u;
                    memcpy(raw,s0,sl); raw[sl]='\0'; strip_tags(raw,res[n].snippet,sizeof(res[n].snippet)); p=s1;
                }
            }
        } else res[n].snippet[0] = '\0';
        if (res[n].url[0] && (!strncmp(res[n].url,"http://",7) || !strncmp(res[n].url,"https://",8))) ++n;
        else ++p;
    }
    return n;
}

static float score_rel(const char *query, const RagResult *r) {
    char q[512]; size_t len=query?strlen(query):0u; if (len>=sizeof(q)) len=sizeof(q)-1u;
    memcpy(q,query?query:"",len); q[len]='\0'; for(size_t i=0;i<len;++i)q[i]=(char)tolower((unsigned char)q[i]);
    int total=0,hits=0; char *tok=strtok(q," \t\r\n");
    while(tok){ if(strlen(tok)>=3u){ ++total; char title[RAG_TITLE_MAX],snippet[RAG_SNIPPET_MAX];
        strncpy(title,r->title,sizeof(title)-1u); title[sizeof(title)-1u]='\0';
        strncpy(snippet,r->snippet,sizeof(snippet)-1u); snippet[sizeof(snippet)-1u]='\0';
        for(char *c=title;*c;++c)*c=(char)tolower((unsigned char)*c);
        for(char *c=snippet;*c;++c)*c=(char)tolower((unsigned char)*c);
        if(strstr(title,tok)||strstr(snippet,tok))++hits; }
        tok=strtok(NULL," \t\r\n"); }
    return total ? (float)hits/(float)total : 0.0f;
}

static void tr_add(RagCtx *ctx, TraceKind kind, float conf, uint32_t t0, const char *fmt, ...) {
    if (!ctx || ctx->n_steps >= RAG_TRACE_MAX) return;
    TraceStep *step=&ctx->trace[ctx->n_steps++]; step->kind=kind; step->elapsed_ms=ms_now()-t0; step->confidence=conf;
    va_list ap; va_start(ap,fmt); vsnprintf(step->detail,sizeof(step->detail),fmt,ap); va_end(ap);
}

bool casper_rag_online(RagBackend backend) {
#ifdef _WIN32
    const wchar_t *host = backend == RAG_BACKEND_BING ? L"www.bing.com" : L"html.duckduckgo.com";
    HINTERNET session=WinHttpOpen(L"Casper/1.0",WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,WINHTTP_NO_PROXY_NAME,WINHTTP_NO_PROXY_BYPASS,0);
    if(!session)return false; HINTERNET connection=WinHttpConnect(session,host,INTERNET_DEFAULT_HTTPS_PORT,0);
    bool ok=connection!=NULL; if(connection)WinHttpCloseHandle(connection); WinHttpCloseHandle(session); return ok;
#else
    (void)backend; return false;
#endif
}

RagCtx *casper_rag_query(const char *q, RagBackend backend, const char *rules_path) {
    (void)rules_path; if(!q||!q[0])return NULL; RagCtx *ctx=(RagCtx*)calloc(1u,sizeof(*ctx)); if(!ctx)return NULL;
    uint32_t t0=ms_now(); strncpy(ctx->query,q,sizeof(ctx->query)-1u); tr_add(ctx,TRACE_PARSE,0.0f,t0,"parsed %zu",strlen(q));
#ifdef _WIN32
    char enc[512]; url_enc(q,enc,sizeof(enc)); const char *host="html.duckduckgo.com"; if(backend==RAG_BACKEND_BING)host="www.bing.com";
    const char *fmt=backend==RAG_BACKEND_BING?"/search?q=%s&setlang=en":"/html/?q=%s"; char path[640]; (void)snprintf(path,sizeof(path),fmt,enc);
    tr_add(ctx,TRACE_SEARCH,0.0f,t0,"GET https://%s%s",host,path); HBuf *resp=http_get(host,path);
#else
    HBuf *resp=NULL;
#endif
    if(!resp||resp->len==0u){tr_add(ctx,TRACE_WARN,0.0f,t0,"offline");hbuf_free(resp);ctx->elapsed_ms=ms_now()-t0;return ctx;}
    tr_add(ctx,TRACE_FETCH,1.0f,t0,"received %zu",resp->len); ctx->n_results=parse_ddg(resp->buf,ctx->results,RAG_MAX_RESULTS); hbuf_free(resp);
    float total_score=0.0f; for(int i=0;i<ctx->n_results;++i){ctx->results[i].score=score_rel(q,&ctx->results[i]); total_score+=ctx->results[i].score;
        size_t ul=strlen(ctx->results[i].url),sl=strlen(ctx->results[i].snippet),n=ul+1u+sl; uint8_t *anchor=(uint8_t*)malloc(n?n:1u);
        if(!anchor){memset(ctx->results[i].sha256,0,sizeof(ctx->results[i].sha256));continue;}
        memcpy(anchor,ctx->results[i].url,ul); anchor[ul]=0; memcpy(anchor+ul+1u,ctx->results[i].snippet,sl); niyah_sha256(anchor,n,ctx->results[i].sha256); free(anchor);
    }
    ctx->confidence=ctx->n_results?(total_score/(float)ctx->n_results):0.0f; tr_add(ctx,TRACE_RANK,ctx->confidence,t0,"results %d",ctx->n_results);
    size_t cp=0u; for(int i=0;i<ctx->n_results;++i){if(cp>=RAG_CONTEXT_MAX-1u)break;int wrote=snprintf(ctx->context+cp,RAG_CONTEXT_MAX-cp,"[%d] %s\n%s\n\n",i+1,ctx->results[i].title,ctx->results[i].snippet);if(wrote<0)break;size_t add=(size_t)wrote;if(add>=RAG_CONTEXT_MAX-cp){cp=RAG_CONTEXT_MAX-1u;break;}cp+=add;}
    tr_add(ctx,TRACE_CONTEXT,ctx->confidence,t0,"context %zu",cp); niyah_sha256((const uint8_t*)ctx->context,cp,ctx->chain_hash); tr_add(ctx,TRACE_COMPOSE,ctx->confidence,t0,"conf=%.3f",(double)ctx->confidence); ctx->elapsed_ms=ms_now()-t0; return ctx;
}

void casper_rag_free(RagCtx *ctx){free(ctx);}

static void json_escape(char *out,size_t max,const char *s){size_t o=0u;if(!out||!max)return;if(!s){out[0]='\0';return;}for(const unsigned char*p=(const unsigned char*)s;*p&&o+2u<max;++p){unsigned char c=*p;if(c=='"'||c=='\\'){out[o++]='\\';out[o++]=(char)c;}else if(c=='\n'){out[o++]='\\';out[o++]='n';}else if(c=='\r'){out[o++]='\\';out[o++]='r';}else if(c=='\t'){out[o++]='\\';out[o++]='t';}else if(c<0x20u){if(o+6u>=max)break;o+=(size_t)snprintf(out+o,max-o,"\\u%04x",c);}else out[o++]=(char)c;}out[o]='\0';}

char *casper_rag_to_json(const RagCtx *ctx){if(!ctx)return NULL;size_t cap=32768u;char*out=(char*)malloc(cap);if(!out)return NULL;char query[1024],chain[65];json_escape(query,sizeof(query),ctx->query);niyah_hash_to_hex(ctx->chain_hash,chain);int n=snprintf(out,cap,"{\"query\":\"%s\",\"confidence\":%.3f,\"elapsed_ms\":%u,\"chain_hash\":\"%s\",\"n_sources\":%d,\"n_steps\":%d}",query,(double)ctx->confidence,ctx->elapsed_ms,chain,ctx->n_results,ctx->n_steps);if(n<0||(size_t)n>=cap){free(out);return NULL;}return out;}

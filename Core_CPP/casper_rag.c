/* casper_rag.c — deterministic RAG transport/parser/ranker. C11. */

/*
 * Feature macro must precede every include.
 *
 * scripts/build.sh compiles with -std=c11 -pedantic. Under strict C11 glibc
 * does not declare clock_gettime(), nanosleep() or popen(), so this file
 * fails with implicit-declaration errors under -Werror without this. The
 * previous build script passed no -std= at all, so the compiler default
 * (gnu23) exposed them and the dependency stayed invisible.
 */
#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#endif

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

/*
 * Politeness controls.
 *
 * The previous revision announced itself as "Casper/1.0" and issued a
 * request the instant it was called, with no interval and no status check.
 * That combination is what got the source IP blocked by DuckDuckGo.
 */
#define RAG_MIN_INTERVAL_MS  2000u
#define RAG_MAX_ATTEMPTS     3
#define RAG_BACKOFF_BASE_MS  1500u

static const char rag_user_agent[] =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

typedef struct { char *buf; size_t len; size_t cap; } HBuf;

static uint32_t ms_now(void) {
#ifdef _WIN32
    return (uint32_t)GetTickCount();
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (uint32_t)((double)clock() * 1000.0 / (double)CLOCKS_PER_SEC);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u);
#endif
}

static void sleep_ms(uint32_t ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec  = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)((ms % 1000u) * 1000000u);
    (void)nanosleep(&ts, NULL);
#endif
}

/* Enforce a floor on the gap between outbound requests. */
static void throttle(void) {
    static uint32_t last_ms  = 0u;
    static int      have_last = 0;

    if (have_last) {
        uint32_t elapsed = ms_now() - last_ms;   /* unsigned wrap is intended */
        if (elapsed < RAG_MIN_INTERVAL_MS) sleep_ms(RAG_MIN_INTERVAL_MS - elapsed);
    }
    last_ms   = ms_now();
    have_last = 1;
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

/*
 * Hosts reach a command line on POSIX, so restrict them to the characters a
 * hostname and port can legitimately contain. url_enc already guarantees the
 * path side carries nothing but [A-Za-z0-9-_.~%] and the fixed query keys.
 */
static int host_is_safe(const char *host) {
    const unsigned char *p;
    if (!host || !host[0] || strlen(host) > 255u) return 0;
    for (p = (const unsigned char *)host; *p; ++p) {
        if (!(isalnum(*p) || *p == '.' || *p == '-' || *p == ':')) return 0;
    }
    return 1;
}

/*
 * SearXNG endpoint, host[:port] with no scheme, e.g. 127.0.0.1:8888.
 * A self-hosted instance is the only backend here that cannot rate-limit
 * you. The instance must list json under search.formats in settings.yml;
 * upstream ships that disabled.
 */
static const char *searxng_host(void) {
    const char *h = getenv("SEARXNG_HOST");
    return (h && host_is_safe(h)) ? h : NULL;
}

#ifdef _WIN32
static HBuf *http_get(const char *host, const char *path, unsigned *status_out) {
    HBuf *out = hbuf_new(); if (!out) return NULL;
    if (status_out) *status_out = 0u;
    int host_wlen = MultiByteToWideChar(CP_UTF8, 0, host, -1, NULL, 0);
    int path_wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    if (host_wlen <= 0 || path_wlen <= 0) { hbuf_free(out); return NULL; }
    wchar_t *whost = (wchar_t *)calloc((size_t)host_wlen, sizeof(wchar_t));
    wchar_t *wpath = (wchar_t *)calloc((size_t)path_wlen, sizeof(wchar_t));
    if (!whost || !wpath) { free(whost); free(wpath); hbuf_free(out); return NULL; }
    if (!MultiByteToWideChar(CP_UTF8, 0, host, -1, whost, host_wlen) || !MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, path_wlen)) {
        free(whost); free(wpath); hbuf_free(out); return NULL;
    }
    /* Browser UA here too: WinHttpOpen's agent string is what goes on the
       wire when no explicit header overrides it. */
    HINTERNET session = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                                   WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) { free(whost); free(wpath); hbuf_free(out); return NULL; }
    HINTERNET connection = WinHttpConnect(session, whost, INTERNET_DEFAULT_HTTPS_PORT, 0);
    if (!connection) { WinHttpCloseHandle(session); free(whost); free(wpath); hbuf_free(out); return NULL; }
    HINTERNET request = WinHttpOpenRequest(connection, L"GET", wpath, NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
    free(whost); free(wpath);
    if (!request) { WinHttpCloseHandle(connection); WinHttpCloseHandle(session); hbuf_free(out); return NULL; }
    WinHttpAddRequestHeaders(request,
        L"Accept: text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8\r\n"
        L"Accept-Language: en-US,en;q=0.9,ar;q=0.8\r\n",
        (DWORD)-1, WINHTTP_ADDREQ_FLAG_ADD);
    DWORD timeout = RAG_TIMEOUT_MS;
    WinHttpSetTimeouts(request, timeout, timeout, timeout, timeout);
    BOOL ok = WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) && WinHttpReceiveResponse(request, NULL);
    if (ok) {
        /* Read the status code. Without this a 403 or 429 body was handed to
           the parser, which reported "offline" and hid the block. */
        DWORD code = 0, code_sz = (DWORD)sizeof(code);
        if (WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                                WINHTTP_HEADER_NAME_BY_INDEX, &code, &code_sz, WINHTTP_NO_HEADER_INDEX)) {
            if (status_out) *status_out = (unsigned)code;
        }
        DWORD available = 0;
        while (WinHttpQueryDataAvailable(request, &available) && available > 0) {
            char chunk[8192]; DWORD read = 0; DWORD want = available < (DWORD)sizeof(chunk) ? available : (DWORD)sizeof(chunk);
            if (!WinHttpReadData(request, chunk, want, &read) || read == 0) { ok = FALSE; break; }
            if (hbuf_app(out, chunk, (size_t)read) != 0) { ok = FALSE; break; }
        }
    }
    WinHttpCloseHandle(request); WinHttpCloseHandle(connection); WinHttpCloseHandle(session);
    if (!ok || out->len == 0u) { hbuf_free(out); return NULL; }
    return out;
}
#else
/*
 * POSIX transport.
 *
 * The old #else branch was literally "resp = NULL", so every non-Windows
 * build reported "offline" regardless of the network. curl is used rather
 * than libcurl so the link line gains no dependency. Injection is not
 * reachable: host_is_safe() restricts the host and url_enc() percent-encodes
 * the query, so the constructed URL holds no shell metacharacters.
 */
static HBuf *http_get(const char *host, const char *path, unsigned *status_out) {
    char   cmd[2048];
    HBuf  *out;
    FILE  *fp;
    char   chunk[8192];
    size_t got;
    char  *marker;

    if (status_out) *status_out = 0u;
    if (!host_is_safe(host) || !path) return NULL;

    if (snprintf(cmd, sizeof(cmd),
                 "curl -sS -L --max-time %u --compressed "
                 "-A '%s' "
                 "-H 'Accept: text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8' "
                 "-H 'Accept-Language: en-US,en;q=0.9,ar;q=0.8' "
                 "-w '\\n__STATUS__%%{http_code}' "
                 "'https://%s%s' 2>/dev/null",
                 (unsigned)(RAG_TIMEOUT_MS / 1000u), rag_user_agent, host, path)
        >= (int)sizeof(cmd)) {
        return NULL;
    }

    fp = popen(cmd, "r");
    if (!fp) return NULL;

    out = hbuf_new();
    if (!out) { (void)pclose(fp); return NULL; }

    while ((got = fread(chunk, 1u, sizeof(chunk), fp)) > 0u) {
        if (hbuf_app(out, chunk, got) != 0) break;
    }
    (void)pclose(fp);

    /* Split the trailing "\n__STATUS__<code>" that -w appended. */
    marker = out->len ? strstr(out->buf, "\n__STATUS__") : NULL;
    if (marker) {
        if (status_out) *status_out = (unsigned)strtoul(marker + 11, NULL, 10);
        *marker  = '\0';
        out->len = (size_t)(marker - out->buf);
    }

    if (out->len == 0u) { hbuf_free(out); return NULL; }
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
        if (*p == '>') { in_tag = 0; if (n && out[n-1] != ' ') out[n++]=' '; continue; }
        if (in_tag) continue;
        if (*p == '&') {
            if (!strncmp(p,"&amp;",5)) { out[n++]='&'; p+=4; }
            else if (!strncmp(p,"&lt;",4)) { out[n++]='<'; p+=3; }
            else if (!strncmp(p,"&gt;",4)) { out[n++]='>'; p+=3; }
            else if (!strncmp(p,"&quot;",6)) { out[n++]='"'; p+=5; }
            else if (!strncmp(p,"&#39;",5)) { out[n++]='\''; p+=4; }
            else if (!strncmp(p,"&nbsp;",6)) { out[n++]=' '; p+=5; }
            else out[n++]='&';
            continue;
        }
        if (*p=='\r'||*p=='\n'||*p=='\t') { if(n&&out[n-1]!=' ') out[n++]=' '; }
        else out[n++]=*p;
    }
    while(n&&out[n-1]==' ') --n;
    out[n]='\0';
}

static int parse_ddg(const char *html, RagResult *res, int max) {
    int n=0; const char *p=html;
    while(n<max&&p&&*p){
        const char *h=strstr(p,"result__a\" href=\"");
        if(!h){h=strstr(p,"/l/?uddg=");if(!h)break;h+=9;}else h+=17;
        const char *end=strchr(h,'"'); const char *amp=strchr(h,'&'); if(amp&&(!end||amp<end))end=amp; if(!end)break;
        size_t len=(size_t)(end-h); if(len>=RAG_URL_MAX)len=RAG_URL_MAX-1u; memcpy(res[n].url,h,len);res[n].url[len]='\0';url_decode(res[n].url);p=end;
        const char *gt=strchr(p,'>');if(!gt)break;const char *title_end=strstr(gt+1,"</a>");if(!title_end)title_end=strstr(gt+1,"</");
        if(title_end){char raw[RAG_TITLE_MAX*2u];size_t tl=(size_t)(title_end-(gt+1));if(tl>=sizeof(raw))tl=sizeof(raw)-1u;memcpy(raw,gt+1,tl);raw[tl]='\0';strip_tags(raw,res[n].title,sizeof(res[n].title));p=title_end;}else res[n].title[0]='\0';
        const char *sn=strstr(p,"result__snippet");
        if(sn){const char *s0=strchr(sn,'>');if(s0){++s0;const char *s1=strstr(s0,"</");if(s1){char raw[RAG_SNIPPET_MAX*2u];size_t sl=(size_t)(s1-s0);if(sl>=sizeof(raw))sl=sizeof(raw)-1u;memcpy(raw,s0,sl);raw[sl]='\0';strip_tags(raw,res[n].snippet,sizeof(res[n].snippet));p=s1;}}}else res[n].snippet[0]='\0';
        if(res[n].url[0]&&(!strncmp(res[n].url,"http://",7)||!strncmp(res[n].url,"https://",8)))++n;else ++p;
    }
    return n;
}

/*
 * Minimal JSON string reader: finds "key":" after *cursor and copies the
 * value up to the closing quote, resolving the escapes SearXNG emits.
 * Advances *cursor past the value. Returns 0 when the key is not found.
 */
static int json_field(const char **cursor, const char *key, char *out, size_t max) {
    char        pattern[32];
    const char *p;
    size_t      o = 0u;

    if (!cursor || !*cursor || !out || max == 0u) return 0;
    out[0] = '\0';
    if (snprintf(pattern, sizeof(pattern), "\"%s\":", key) >= (int)sizeof(pattern)) return 0;

    p = strstr(*cursor, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') ++p;
    if (*p != '"') return 0;
    ++p;

    while (*p && *p != '"' && o + 1u < max) {
        if (*p == '\\' && p[1]) {
            ++p;
            switch (*p) {
                case 'n':  out[o++] = '\n'; break;
                case 'r':  out[o++] = '\r'; break;
                case 't':  out[o++] = '\t'; break;
                case 'b':  out[o++] = '\b'; break;
                case 'f':  out[o++] = '\f'; break;
                case '/':  out[o++] = '/';  break;
                case '\\': out[o++] = '\\'; break;
                case '"':  out[o++] = '"';  break;
                case 'u': {
                    /* Re-encode \uXXXX as UTF-8 so Arabic survives. */
                    int h0 = p[1] ? hexval(p[1]) : -1, h1 = p[2] ? hexval(p[2]) : -1;
                    int h2 = p[3] ? hexval(p[3]) : -1, h3 = p[4] ? hexval(p[4]) : -1;
                    if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) { out[o++] = 'u'; break; }
                    {
                        unsigned uc = (unsigned)((h0 << 12) | (h1 << 8) | (h2 << 4) | h3);
                        if (uc < 0x80u && o + 1u < max) {
                            out[o++] = (char)uc;
                        } else if (uc < 0x800u && o + 2u < max) {
                            out[o++] = (char)(0xC0u | (uc >> 6));
                            out[o++] = (char)(0x80u | (uc & 0x3Fu));
                        } else if (o + 3u < max) {
                            out[o++] = (char)(0xE0u | (uc >> 12));
                            out[o++] = (char)(0x80u | ((uc >> 6) & 0x3Fu));
                            out[o++] = (char)(0x80u | (uc & 0x3Fu));
                        }
                        p += 4;
                    }
                    break;
                }
                default: out[o++] = *p; break;
            }
            ++p;
            continue;
        }
        out[o++] = *p++;
    }

    out[o] = '\0';
    *cursor = *p ? p + 1 : p;
    return 1;
}

/* SearXNG /search?format=json: {"results":[{"url":..,"title":..,"content":..}]} */
static int parse_searxng(const char *json, RagResult *res, int max) {
    const char *cursor = json ? strstr(json, "\"results\"") : NULL;
    int         n      = 0;

    if (!cursor) return 0;

    while (n < max) {
        const char *probe = cursor;
        if (!json_field(&probe, "url", res[n].url, sizeof(res[n].url))) break;
        cursor = probe;
        (void)json_field(&cursor, "title",   res[n].title,   sizeof(res[n].title));
        (void)json_field(&cursor, "content", res[n].snippet, sizeof(res[n].snippet));
        if (!strncmp(res[n].url, "http://", 7) || !strncmp(res[n].url, "https://", 8)) ++n;
    }
    return n;
}

static float score_rel(const char *query, const RagResult *r) {
    char q[512]; size_t len=query?strlen(query):0u; if(len>=sizeof(q))len=sizeof(q)-1u;memcpy(q,query?query:"",len);q[len]='\0';for(size_t i=0;i<len;++i)q[i]=(char)tolower((unsigned char)q[i]);
    int total=0,hits=0;char *tok=strtok(q," \t\r\n");
    while(tok){if(strlen(tok)>=3u){++total;char title[RAG_TITLE_MAX],snippet[RAG_SNIPPET_MAX];strncpy(title,r->title,sizeof(title)-1u);title[sizeof(title)-1u]='\0';strncpy(snippet,r->snippet,sizeof(snippet)-1u);snippet[sizeof(snippet)-1u]='\0';for(char*c=title;*c;++c)*c=(char)tolower((unsigned char)*c);for(char*c=snippet;*c;++c)*c=(char)tolower((unsigned char)*c);if(strstr(title,tok)||strstr(snippet,tok))++hits;}tok=strtok(NULL," \t\r\n");}
    return total?(float)hits/(float)total:0.0f;
}

static int result_cmp(const void *a,const void *b){const RagResult*ra=(const RagResult*)a,*rb=(const RagResult*)b;if(ra->score<rb->score)return 1;if(ra->score>rb->score)return -1;int u=strcmp(ra->url,rb->url);if(u)return u;return strcmp(ra->title,rb->title);}

static void tr_add(RagCtx *ctx,TraceKind kind,float conf,uint32_t t0,const char *fmt,...){if(!ctx||ctx->n_steps>=RAG_TRACE_MAX)return;TraceStep*step=&ctx->trace[ctx->n_steps++];step->kind=kind;step->elapsed_ms=ms_now()-t0;step->confidence=conf;va_list ap;va_start(ap,fmt);vsnprintf(step->detail,sizeof(step->detail),fmt,ap);va_end(ap);}

/*
 * Throttled GET with backoff. 403/429/503 mean "slow down or you are
 * blocked", which is a different condition from "no network" and is now
 * reported as such instead of collapsing into "offline".
 */
static HBuf *http_get_retry(const char *host, const char *path, RagCtx *ctx, uint32_t t0) {
    int attempt;

    for (attempt = 1; attempt <= RAG_MAX_ATTEMPTS; ++attempt) {
        unsigned status = 0u;
        HBuf    *resp;

        throttle();
        resp = http_get(host, path, &status);

        if (resp && (status == 200u || status == 0u)) return resp;
        hbuf_free(resp);

        if (status == 403u || status == 429u || status == 503u) {
            uint32_t wait = RAG_BACKOFF_BASE_MS << (attempt - 1);
            tr_add(ctx, TRACE_WARN, 0.0f, t0, "http %u rate-limited, waiting %u ms (%d/%d)",
                   status, wait, attempt, RAG_MAX_ATTEMPTS);
            sleep_ms(wait);
            continue;
        }

        if (status != 0u) tr_add(ctx, TRACE_WARN, 0.0f, t0, "http %u", status);
        return NULL;
    }

    tr_add(ctx, TRACE_WARN, 0.0f, t0, "blocked after %d attempts, try CASPER_BACKEND=searxng",
           RAG_MAX_ATTEMPTS);
    return NULL;
}

bool casper_rag_online(RagBackend backend){
    if(backend==RAG_BACKEND_SEARXNG) return searxng_host()!=NULL;
#ifdef _WIN32
    if(backend!=RAG_BACKEND_DDG&&backend!=RAG_BACKEND_BING)return false;
    const wchar_t*host=backend==RAG_BACKEND_BING?L"www.bing.com":L"html.duckduckgo.com";HINTERNET session=WinHttpOpen(L"Mozilla/5.0",WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,WINHTTP_NO_PROXY_NAME,WINHTTP_NO_PROXY_BYPASS,0);if(!session)return false;HINTERNET connection=WinHttpConnect(session,host,INTERNET_DEFAULT_HTTPS_PORT,0);bool ok=connection!=NULL;if(connection)WinHttpCloseHandle(connection);WinHttpCloseHandle(session);return ok;
#else
    /* Reachability is decided by the transport, which now exists on POSIX. */
    return backend==RAG_BACKEND_DDG||backend==RAG_BACKEND_BING;
#endif
}

RagCtx *casper_rag_query(const char *q, RagBackend backend, const char *rules_path) {
    RagCtx     *ctx;
    uint32_t    t0;
    const char *host;
    const char *fmt;
    char        enc[512];
    char        path[640];
    HBuf       *resp;
    int         i;
    float       total_score = 0.0f;
    size_t      cp = 0u;

    (void)rules_path;
    if (!q || !q[0]) return NULL;

    ctx = (RagCtx *)calloc(1u, sizeof(*ctx));
    if (!ctx) return NULL;
    t0 = ms_now();
    strncpy(ctx->query, q, sizeof(ctx->query) - 1u);
    tr_add(ctx, TRACE_PARSE, 0.0f, t0, "parsed %zu", strlen(q));

    switch (backend) {
        case RAG_BACKEND_SEARXNG:
            host = searxng_host();
            if (!host) {
                tr_add(ctx, TRACE_WARN, 0.0f, t0, "SEARXNG_HOST unset or invalid");
                ctx->elapsed_ms = ms_now() - t0;
                return ctx;
            }
            fmt = "/search?q=%s&format=json&language=en&safesearch=0";
            break;
        case RAG_BACKEND_BING:
            host = "www.bing.com";
            fmt  = "/search?q=%s&setlang=en";
            break;
        case RAG_BACKEND_DDG:
            host = "html.duckduckgo.com";
            fmt  = "/html/?q=%s";
            break;
        default:
            tr_add(ctx, TRACE_WARN, 0.0f, t0, "unsupported backend");
            ctx->elapsed_ms = ms_now() - t0;
            return ctx;
    }

    url_enc(q, enc, sizeof(enc));
    if (snprintf(path, sizeof(path), fmt, enc) >= (int)sizeof(path)) {
        tr_add(ctx, TRACE_WARN, 0.0f, t0, "query too long");
        ctx->elapsed_ms = ms_now() - t0;
        return ctx;
    }
    tr_add(ctx, TRACE_SEARCH, 0.0f, t0, "GET https://%s%s", host, path);

    resp = http_get_retry(host, path, ctx, t0);
    if (!resp || resp->len == 0u) {
        tr_add(ctx, TRACE_WARN, 0.0f, t0, "no response");
        hbuf_free(resp);
        ctx->elapsed_ms = ms_now() - t0;
        return ctx;
    }

    tr_add(ctx, TRACE_FETCH, 1.0f, t0, "received %zu", resp->len);
    ctx->n_results = (backend == RAG_BACKEND_SEARXNG)
                   ? parse_searxng(resp->buf, ctx->results, RAG_MAX_RESULTS)
                   : parse_ddg(resp->buf, ctx->results, RAG_MAX_RESULTS);
    hbuf_free(resp);

    for (i = 0; i < ctx->n_results; ++i) {
        size_t   ul, sl, n;
        uint8_t *anchor;

        ctx->results[i].score = score_rel(q, &ctx->results[i]);
        total_score += ctx->results[i].score;

        ul = strlen(ctx->results[i].url);
        sl = strlen(ctx->results[i].snippet);
        n  = ul + 1u + sl;
        anchor = (uint8_t *)malloc(n ? n : 1u);
        if (!anchor) { memset(ctx->results[i].sha256, 0, sizeof(ctx->results[i].sha256)); continue; }
        memcpy(anchor, ctx->results[i].url, ul);
        anchor[ul] = 0;
        memcpy(anchor + ul + 1u, ctx->results[i].snippet, sl);
        niyah_sha256(anchor, n, ctx->results[i].sha256);
        free(anchor);
    }

    qsort(ctx->results, (size_t)ctx->n_results, sizeof(ctx->results[0]), result_cmp);
    ctx->confidence = ctx->n_results ? total_score / (float)ctx->n_results : 0.0f;
    tr_add(ctx, TRACE_RANK, ctx->confidence, t0, "results %d", ctx->n_results);

    for (i = 0; i < ctx->n_results; ++i) {
        int    wrote;
        size_t add;
        if (cp >= RAG_CONTEXT_MAX - 1u) break;
        wrote = snprintf(ctx->context + cp, RAG_CONTEXT_MAX - cp, "[%d] %s\n%s\n\n",
                         i + 1, ctx->results[i].title, ctx->results[i].snippet);
        if (wrote < 0) break;
        add = (size_t)wrote;
        if (add >= RAG_CONTEXT_MAX - cp) { cp = RAG_CONTEXT_MAX - 1u; break; }
        cp += add;
    }
    tr_add(ctx, TRACE_CONTEXT, ctx->confidence, t0, "context %zu", cp);
    niyah_sha256((const uint8_t *)ctx->context, cp, ctx->chain_hash);
    tr_add(ctx, TRACE_COMPOSE, ctx->confidence, t0, "conf=%.3f", (double)ctx->confidence);
    ctx->elapsed_ms = ms_now() - t0;
    return ctx;
}

void casper_rag_free(RagCtx*ctx){free(ctx);}

static void json_escape(char*out,size_t max,const char*s){size_t o=0u;if(!out||!max)return;if(!s){out[0]='\0';return;}for(const unsigned char*p=(const unsigned char*)s;*p&&o+2u<max;++p){unsigned char c=*p;if(c=='"'||c=='\\'){out[o++]='\\';out[o++]=(char)c;}else if(c=='\n'){out[o++]='\\';out[o++]='n';}else if(c=='\r'){out[o++]='\\';out[o++]='r';}else if(c=='\t'){out[o++]='\\';out[o++]='t';}else if(c<0x20u){if(o+6u>=max)break;o+=(size_t)snprintf(out+o,max-o,"\\u%04x",c);}else out[o++]=(char)c;}out[o]='\0';}

char*casper_rag_to_json(const RagCtx*ctx){if(!ctx)return NULL;size_t cap=32768u;char*out=(char*)malloc(cap);if(!out)return NULL;char query[1024],chain[65];json_escape(query,sizeof(query),ctx->query);niyah_hash_to_hex(ctx->chain_hash,chain);int n=snprintf(out,cap,"{\"query\":\"%s\",\"confidence\":%.3f,\"elapsed_ms\":%u,\"chain_hash\":\"%s\",\"n_sources\":%d,\"n_steps\":%d}",query,(double)ctx->confidence,ctx->elapsed_ms,chain,ctx->n_results,ctx->n_steps);if(n<0||(size_t)n>=cap){free(out);return NULL;}return out;}

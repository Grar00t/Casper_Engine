#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

static int is_arabic(uint32_t c) {
    return (c >= 0x0600 && c <= 0x06FF) ||
           (c >= 0x0750 && c <= 0x077F) ||
           (c >= 0x08A0 && c <= 0x08FF) ||
           (c >= 0xFB50 && c <= 0xFDFF) ||
           (c >= 0xFE70 && c <= 0xFEFF);
}

#define MAX_VOCAB 65536u
typedef struct { char token[64]; uint32_t id; } TokenEntry;

static TokenEntry vocab[MAX_VOCAB];
static uint32_t vocab_size = 0u;
static int initialized = 0;

static int add_token(const char *s) {
    if (!s || vocab_size >= MAX_VOCAB) return -1;
    strncpy(vocab[vocab_size].token, s, sizeof(vocab[vocab_size].token) - 1u);
    vocab[vocab_size].token[sizeof(vocab[vocab_size].token) - 1u] = '\0';
    vocab[vocab_size].id = vocab_size;
    ++vocab_size;
    return 0;
}

void tokenizer_init(void) {
    if (initialized) return;
    vocab_size = 0u;

    add_token("<BOS>"); add_token("<EOS>"); add_token("<PAD>"); add_token("<UNK>");

    for (int i = 0; i < 10; ++i) {
        char buf[4];
        (void)snprintf(buf, sizeof(buf), "%d", i);
        add_token(buf);
    }

    const char *puncts = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    for (const char *p = puncts; *p; ++p) {
        char buf[3] = {*p, '\0', '\0'};
        add_token(buf);
    }

    for (char c = 'a'; c <= 'z'; ++c) { char buf[2] = {c, '\0'}; add_token(buf); }
    for (char c = 'A'; c <= 'Z'; ++c) { char buf[2] = {c, '\0'}; add_token(buf); }

    const char *words[] = {
        "the","a","an","and","or","is","in","of","to","for","with","on","at","by","from",
        "that","this","it","are","was","be","as","not","but","have","has","had","we","i","you",
        "they","he","she","can","will","would","do","does","did","if","when","then","so","all",
        "no","up","out","one","two","three","four","five","six","seven","eight","nine","ten","than",
        "more","less","very","also","only","model","data","train","training","layer","layers","weight",
        "weights","token","tokens","embed","embedding","head","heads","attention","output","input","loss",
        "gradient","optimizer","matrix","vector","kernel","cpu","gpu","memory","float","int","size","context",
        "vocab","local","zero","code","file","build","run","test","hash","proof","rule","query","fact",
        "function","class","struct","type","return","void","static","const","malloc","calloc","free","pointer",
        "buffer","stack","heap","pool","forward","backward","sample","generate","decode","encode","norm","softmax",
        "relu","silu","gelu","linear","bias","scale","sum","dot","compute","algorithm","system","engine","core",
        "base","key","value","arabic","quran","bismillah","inference","symbolic","logic","constraint","solver",
        "rational","arithmetic","sha","cryptographic","niyah","casper","khwarizmi","adam","rope","swiglu","rmsnorm",
        "gqa","zero","one","two","three","four","five","six","seven","eight","nine","ten","hundred","thousand",
        "million","billion", NULL
    };
    for (size_t i = 0; words[i]; ++i) add_token(words[i]);
    initialized = 1;
}

uint32_t tokenizer_encode(const char *text, uint32_t *tokens, uint32_t max_len) {
    if (!initialized) tokenizer_init();
    if (!tokens || max_len < 2u) return 0u;
    if (!text) text = "";

    uint32_t pos = 0u;
    tokens[pos++] = 0u;
    const unsigned char *p = (const unsigned char *)text;

    while (*p && pos + 1u < max_len) {
        if (isspace(*p)) { ++p; continue; }

        if (*p >= 0xC0u) {
            uint32_t uc = 0u;
            size_t width = 1u;
            if ((*p & 0xE0u) == 0xC0u && p[1]) {
                uc = ((uint32_t)(p[0] & 0x1Fu) << 6) | (uint32_t)(p[1] & 0x3Fu); width = 2u;
            } else if ((*p & 0xF0u) == 0xE0u && p[1] && p[2]) {
                uc = ((uint32_t)(p[0] & 0x0Fu) << 12) | ((uint32_t)(p[1] & 0x3Fu) << 6) | (uint32_t)(p[2] & 0x3Fu); width = 3u;
            } else if ((*p & 0xF8u) == 0xF0u && p[1] && p[2] && p[3]) {
                uc = ((uint32_t)(p[0] & 0x07u) << 18) | ((uint32_t)(p[1] & 0x3Fu) << 12) | ((uint32_t)(p[2] & 0x3Fu) << 6) | (uint32_t)(p[3] & 0x3Fu); width = 4u;
            } else {
                uc = 0xFFFDu;
            }
            p += width;
            tokens[pos++] = is_arabic(uc) ? 1000u + (uc % 5000u) : 6000u + (uc % 10000u);
            continue;
        }

        if (ispunct(*p)) {
            char sym[2] = {(char)*p, '\0'};
            uint32_t id = 3u;
            for (uint32_t j = 0u; j < vocab_size; ++j) {
                if (strcmp(vocab[j].token, sym) == 0) { id = vocab[j].id; break; }
            }
            tokens[pos++] = id;
            ++p;
            continue;
        }

        char word[64] = {0};
        size_t i = 0u;
        while (*p && !isspace(*p) && !ispunct(*p) && i + 1u < sizeof(word)) word[i++] = (char)*p++;
        if (i != 0u) {
            uint32_t id = 3u;
            for (uint32_t j = 0u; j < vocab_size; ++j) {
                if (strcmp(vocab[j].token, word) == 0) { id = vocab[j].id; break; }
            }
            tokens[pos++] = id;
        }
    }
    tokens[pos++] = 1u;
    return pos;
}

char *tokenizer_decode(const uint32_t *tokens, uint32_t n) {
    if (!tokens && n != 0u) return NULL;
    size_t cap = (size_t)n * 66u + 1u;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t pos = 0u;

    if (!initialized) tokenizer_init();
    for (uint32_t i = 0u; i < n; ++i) {
        uint32_t id = tokens[i];
        if (id == 0u || id == 1u || id == 2u) continue;
        const char *word = NULL;
        for (uint32_t j = 0u; j < vocab_size; ++j) {
            if (vocab[j].id == id) { word = vocab[j].token; break; }
        }
        if (!word) word = "?";
        if (pos && pos + 1u < cap) out[pos++] = ' ';
        size_t wlen = strlen(word);
        if (pos + wlen >= cap) break;
        memcpy(out + pos, word, wlen);
        pos += wlen;
    }
    out[pos] = '\0';
    return out;
}

void tokenizer_free(void) {
    vocab_size = 0u;
    initialized = 0;
}

#ifdef TOKENIZER_TEST
int main(void) {
    tokenizer_init();
    uint32_t tokens[1024];
    uint32_t n = tokenizer_encode("malloc allocates heap memory", tokens, 1024);
    printf("English: %u tokens\n", n);
    n = tokenizer_encode("بِسۡمِ ٱللَّهِ", tokens, 1024);
    printf("Arabic: %u tokens\n", n);
    printf("Vocab size: %u\n", vocab_size);
    tokenizer_free();
    return 0;
}
#endif

#ifdef __cplusplus
}
#endif

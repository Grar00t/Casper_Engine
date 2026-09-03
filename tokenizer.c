/*
 * tokenizer.c - deterministic UTF-8 tokenizer for the Casper/NIYAH engine.
 *
 * Vocabulary layout. Ids are positional and stable across builds, so a model
 * trained against one build stays valid against the next:
 *
 *      0 ..    3   <BOS> <EOS> <PAD> <UNK>
 *      4 ..   13   digits '0'..'9'
 *     14 ..   45   ASCII punctuation (32 entries)
 *     46 ..   71   'a'..'z'
 *     72 ..   97   'A'..'Z'
 *     98 ..  267   English / domain word list (170 unique entries)
 *    268 .. 1499   Arabic codepoints, one token each (1232 entries)
 *
 * Why the codepoint block exists: the previous revision synthesized ids as
 * 1000 + (uc % 5000) for Arabic and 6000 + (uc % 10000) otherwise. Those ids
 * were never inserted into the vocabulary, so tokenizer_decode() found no
 * matching entry and wrote "?" for every Arabic character. The mapping was
 * also lossy: 0x0600-0x06FF and 0xFB50-0xFDFF collided after the modulo.
 *
 * Every Arabic codepoint now owns a real entry, so encode -> decode is an
 * identity for Arabic text.
 */

#include "tokenizer.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char     token[TOK_STR_MAX];
    uint32_t id;
} TokenEntry;

typedef struct {
    uint32_t first;
    uint32_t last;
} CodepointRange;

/* Arabic blocks represented as single-codepoint tokens. */
static const CodepointRange arabic_ranges[] = {
    { 0x0600u, 0x06FFu },   /* Arabic                       - 256 */
    { 0x0750u, 0x077Fu },   /* Arabic Supplement            -  48 */
    { 0x08A0u, 0x08FFu },   /* Arabic Extended-A            -  96 */
    { 0xFB50u, 0xFDFFu },   /* Arabic Presentation Forms-A  - 688 */
    { 0xFE70u, 0xFEFFu }    /* Arabic Presentation Forms-B  - 144 */
};
#define ARABIC_RANGE_COUNT (sizeof(arabic_ranges) / sizeof(arabic_ranges[0]))

static TokenEntry vocab[TOK_MAX_VOCAB];
static uint32_t   vocab_size  = 0u;
static uint32_t   word_end    = 0u;  /* one past the last string token   */
static uint32_t   char_base   = 0u;  /* first id in the codepoint block  */
static uint32_t   char_count  = 0u;  /* entries in the codepoint block   */
static int        initialized = 0;

static uint32_t add_token(const char *s)
{
    if (!s || !s[0] || vocab_size >= TOK_MAX_VOCAB) return TOK_UNK;
    strncpy(vocab[vocab_size].token, s, TOK_STR_MAX - 1u);
    vocab[vocab_size].token[TOK_STR_MAX - 1u] = '\0';
    vocab[vocab_size].id = vocab_size;
    return vocab_size++;
}

/* Write uc as UTF-8 into out. Returns bytes written, excluding terminator. */
static size_t utf8_put(uint32_t uc, char *out, size_t max)
{
    if (!out || max < 5u) return 0u;
    if (uc < 0x80u) {
        out[0] = (char)uc;
        out[1] = '\0';
        return 1u;
    }
    if (uc < 0x800u) {
        out[0] = (char)(0xC0u | (uc >> 6));
        out[1] = (char)(0x80u | (uc & 0x3Fu));
        out[2] = '\0';
        return 2u;
    }
    if (uc < 0x10000u) {
        out[0] = (char)(0xE0u | (uc >> 12));
        out[1] = (char)(0x80u | ((uc >> 6) & 0x3Fu));
        out[2] = (char)(0x80u | (uc & 0x3Fu));
        out[3] = '\0';
        return 3u;
    }
    out[0] = (char)(0xF0u | (uc >> 18));
    out[1] = (char)(0x80u | ((uc >> 12) & 0x3Fu));
    out[2] = (char)(0x80u | ((uc >> 6) & 0x3Fu));
    out[3] = (char)(0x80u | (uc & 0x3Fu));
    out[4] = '\0';
    return 4u;
}

/* Codepoint -> id in O(1). Returns TOK_UNK outside the covered ranges. */
static uint32_t codepoint_id(uint32_t uc)
{
    uint32_t offset = 0u;
    size_t   r;

    if (char_count == 0u) return TOK_UNK;

    for (r = 0u; r < ARABIC_RANGE_COUNT; ++r) {
        uint32_t span = arabic_ranges[r].last - arabic_ranges[r].first + 1u;
        if (uc >= arabic_ranges[r].first && uc <= arabic_ranges[r].last) {
            return char_base + offset + (uc - arabic_ranges[r].first);
        }
        offset += span;
    }
    return TOK_UNK;
}

/* Exact match, then a lowercase retry. Scans string tokens only. */
static uint32_t lookup(const char *s)
{
    char     folded[TOK_STR_MAX];
    size_t   i;
    uint32_t j;

    if (!s || !s[0]) return TOK_UNK;

    for (j = 0u; j < word_end; ++j) {
        if (strcmp(vocab[j].token, s) == 0) return vocab[j].id;
    }

    for (i = 0u; s[i] && i + 1u < sizeof(folded); ++i) {
        folded[i] = (char)tolower((unsigned char)s[i]);
    }
    folded[i] = '\0';

    for (j = 0u; j < word_end; ++j) {
        if (strcmp(vocab[j].token, folded) == 0) return vocab[j].id;
    }
    return TOK_UNK;
}

void tokenizer_init(void)
{
    /*
     * De-duplicated. The previous list repeated "zero" and "one"."ten",
     * 11 dead entries that could never be reached by lookup.
     */
    static const char *const words[] = {
        "the","a","an","and","or","is","in","of","to","for","with","on","at","by","from",
        "that","this","it","are","was","be","as","not","but","have","has","had","we","i","you",
        "they","he","she","can","will","would","do","does","did","if","when","then","so","all",
        "no","up","out","than",
        "more","less","very","also","only","model","data","train","training","layer","layers","weight",
        "weights","token","tokens","embed","embedding","head","heads","attention","output","input","loss",
        "gradient","optimizer","matrix","vector","kernel","cpu","gpu","memory","float","int","size","context",
        "vocab","local","code","file","build","run","test","hash","proof","rule","query","fact",
        "function","class","struct","type","return","void","static","const","malloc","calloc","free","pointer",
        "buffer","stack","heap","pool","forward","backward","sample","generate","decode","encode","norm","softmax",
        "relu","silu","gelu","linear","bias","scale","sum","dot","compute","algorithm","system","engine","core",
        "base","key","value","arabic","quran","bismillah","inference","symbolic","logic","constraint","solver",
        "rational","arithmetic","sha","cryptographic","niyah","casper","khwarizmi","adam","rope","swiglu","rmsnorm",
        "gqa","zero","one","two","three","four","five","six","seven","eight","nine","ten","hundred","thousand",
        "million","billion", NULL
    };
    static const char punctuation[] = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    const char *p;
    size_t      i;
    size_t      r;
    int         d;
    char        c;

    if (initialized) return;

    vocab_size = 0u;
    word_end   = 0u;
    char_base  = 0u;
    char_count = 0u;

    /* 0..3 - reserved, in the order the TOK_* macros declare. */
    (void)add_token("<BOS>");
    (void)add_token("<EOS>");
    (void)add_token("<PAD>");
    (void)add_token("<UNK>");

    /* 4..13 - digits. */
    for (d = 0; d < 10; ++d) {
        char buf[4];
        (void)snprintf(buf, sizeof(buf), "%d", d);
        (void)add_token(buf);
    }

    /* 14..45 - ASCII punctuation. */
    for (p = punctuation; *p; ++p) {
        char buf[2] = { '\0', '\0' };
        buf[0] = *p;
        (void)add_token(buf);
    }

    /* 46..97 - ASCII letters. */
    for (c = 'a'; c <= 'z'; ++c) { char buf[2] = { '\0', '\0' }; buf[0] = c; (void)add_token(buf); }
    for (c = 'A'; c <= 'Z'; ++c) { char buf[2] = { '\0', '\0' }; buf[0] = c; (void)add_token(buf); }

    /* 98..267 - word list. */
    for (i = 0u; words[i]; ++i) (void)add_token(words[i]);

    word_end = vocab_size;

    /* 268..1499 - one entry per Arabic codepoint, in declared range order. */
    char_base = vocab_size;
    for (r = 0u; r < ARABIC_RANGE_COUNT; ++r) {
        uint32_t uc;
        for (uc = arabic_ranges[r].first; uc <= arabic_ranges[r].last; ++uc) {
            char utf8[8];
            if (utf8_put(uc, utf8, sizeof(utf8)) == 0u) continue;
            if (add_token(utf8) == TOK_UNK && vocab_size >= TOK_MAX_VOCAB) break;
        }
    }
    char_count = vocab_size - char_base;

    initialized = 1;
}

uint32_t tokenizer_vocab_size(void)
{
    return vocab_size;
}

uint32_t tokenizer_encode(const char *text, uint32_t *tokens, uint32_t max_len)
{
    const unsigned char *p;
    uint32_t             pos = 0u;

    if (!initialized) tokenizer_init();
    if (!tokens || max_len < 2u) return 0u;
    if (!text) text = "";

    tokens[pos++] = TOK_BOS;
    p = (const unsigned char *)text;

    while (*p && pos + 1u < max_len) {
        if (isspace(*p)) { ++p; continue; }

        if (*p >= 0xC0u) {
            uint32_t uc    = 0xFFFDu;
            size_t   width = 1u;

            if ((*p & 0xE0u) == 0xC0u && p[1]) {
                uc = ((uint32_t)(p[0] & 0x1Fu) << 6)
                   |  (uint32_t)(p[1] & 0x3Fu);
                width = 2u;
            } else if ((*p & 0xF0u) == 0xE0u && p[1] && p[2]) {
                uc = ((uint32_t)(p[0] & 0x0Fu) << 12)
                   | ((uint32_t)(p[1] & 0x3Fu) << 6)
                   |  (uint32_t)(p[2] & 0x3Fu);
                width = 3u;
            } else if ((*p & 0xF8u) == 0xF0u && p[1] && p[2] && p[3]) {
                uc = ((uint32_t)(p[0] & 0x07u) << 18)
                   | ((uint32_t)(p[1] & 0x3Fu) << 12)
                   | ((uint32_t)(p[2] & 0x3Fu) << 6)
                   |  (uint32_t)(p[3] & 0x3Fu);
                width = 4u;
            }

            p += width;
            tokens[pos++] = codepoint_id(uc);
            continue;
        }

        if (*p >= 0x80u) {          /* stray continuation byte */
            ++p;
            tokens[pos++] = TOK_UNK;
            continue;
        }

        if (ispunct(*p)) {
            char sym[2] = { '\0', '\0' };
            sym[0] = (char)*p;
            tokens[pos++] = lookup(sym);
            ++p;
            continue;
        }

        {
            char   word[TOK_STR_MAX];
            size_t i = 0u;

            while (*p && *p < 0x80u && !isspace(*p) && !ispunct(*p)
                   && i + 1u < sizeof(word)) {
                word[i++] = (char)*p++;
            }
            word[i] = '\0';
            if (i != 0u) tokens[pos++] = lookup(word);
        }
    }

    tokens[pos++] = TOK_EOS;
    return pos;
}

char *tokenizer_decode(const uint32_t *tokens, uint32_t n)
{
    size_t   cap;
    size_t   pos           = 0u;
    int      prev_was_char = 0;
    char    *out;
    uint32_t i;

    if (!initialized) tokenizer_init();
    if (!tokens && n != 0u) return NULL;

    cap = (size_t)n * (TOK_STR_MAX + 1u) + 1u;
    out = (char *)malloc(cap);
    if (!out) return NULL;

    for (i = 0u; i < n; ++i) {
        uint32_t    id = tokens[i];
        const char *word;
        size_t      wlen;
        int         is_char;

        if (id == TOK_BOS || id == TOK_EOS || id == TOK_PAD) continue;

        /* Ids are positional, so this is a direct index, not a search. */
        word = (id < vocab_size) ? vocab[id].token : vocab[TOK_UNK].token;
        is_char = (char_count != 0u && id >= char_base && id < char_base + char_count);

        /* Arabic characters concatenate; words are space-separated. */
        if (pos != 0u && !(is_char && prev_was_char) && pos + 1u < cap) {
            out[pos++] = ' ';
        }

        wlen = strlen(word);
        if (pos + wlen >= cap) break;
        memcpy(out + pos, word, wlen);
        pos += wlen;
        prev_was_char = is_char;
    }

    out[pos] = '\0';
    return out;
}

void tokenizer_free_string(char *s)
{
    free(s);
}

void tokenizer_free(void)
{
    vocab_size  = 0u;
    word_end    = 0u;
    char_base   = 0u;
    char_count  = 0u;
    initialized = 0;
}

#ifdef TOKENIZER_TEST
int main(void)
{
    static const char *const cases[] = {
        "malloc allocates heap memory",
        "\xd8\xa8\xd8\xb3\xd9\x85 \xd8\xa7\xd9\x84\xd9\x84\xd9\x87",
        "casper \xd9\x86\xd9\x8a\xd8\xa9 engine",
        NULL
    };
    uint32_t tokens[1024];
    size_t   i;
    int      failures = 0;

    tokenizer_init();
    printf("vocab_size = %u  (cap %u)\n", tokenizer_vocab_size(), TOK_MAX_VOCAB);

    for (i = 0u; cases[i]; ++i) {
        uint32_t n   = tokenizer_encode(cases[i], tokens, 1024u);
        char    *rt  = tokenizer_decode(tokens, n);
        int      bad = (rt == NULL) || (strchr(rt, '?') != NULL);

        printf("%-40s -> %3u tokens -> \"%s\"%s\n",
               cases[i], n, rt ? rt : "(null)", bad ? "   [FAIL]" : "");
        if (bad) ++failures;
        tokenizer_free_string(rt);
    }

    tokenizer_free();
    if (failures != 0) {
        printf("FAIL: %d case(s) lost characters\n", failures);
        return 1;
    }
    printf("PASS: no character decoded to '?'\n");
    return 0;
}
#endif

/*
 * tokenizer.h - public API for tokenizer.c
 *
 * These prototypes are mandatory, not decorative: scripts/build.sh compiles
 * with -Wstrict-prototypes -Wmissing-prototypes -Werror, so tokenizer.c does
 * not compile without this header.
 */
#ifndef CASPER_TOKENIZER_H
#define CASPER_TOKENIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reserved ids. Stable across builds. */
#define TOK_BOS 0u
#define TOK_EOS 1u
#define TOK_PAD 2u
#define TOK_UNK 3u

/* Upper bound on vocabulary entries. Live size is ~1500. */
#define TOK_MAX_VOCAB 2048u

/* Longest token string, including the terminator. */
#define TOK_STR_MAX 64u

/* Build the vocabulary. Idempotent; safe to call repeatedly. */
void tokenizer_init(void);

/* Live vocabulary entry count. Returns 0 before tokenizer_init(). */
uint32_t tokenizer_vocab_size(void);

/*
 * Encode UTF-8 text as TOK_BOS ... TOK_EOS.
 * Returns the number of ids written (<= max_len), or 0 if max_len < 2.
 */
uint32_t tokenizer_encode(const char *text, uint32_t *tokens, uint32_t max_len);

/*
 * Decode ids into a malloc'd UTF-8 string, or NULL on allocation failure.
 * The caller owns the result and must release it with tokenizer_free_string().
 */
char *tokenizer_decode(const uint32_t *tokens, uint32_t n);

/* Release a string returned by tokenizer_decode(). Accepts NULL. */
void tokenizer_free_string(char *s);

/* Reset vocabulary state. Does NOT free strings from tokenizer_decode(). */
void tokenizer_free(void);

#ifdef __cplusplus
}
#endif
#endif /* CASPER_TOKENIZER_H */

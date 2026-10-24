/* metal_attn.h — C ABI for Metal attention+KV (Slice A)
 *
 * Darwin arm64 builds link a real implementation (csrc/metal_attn.mm).
 * Other platforms link csrc/metal_attn_stub.c (always unavailable).
 *
 * Fail-open contract: every function returns 0 on success / non-zero on
 * failure (or 0 for "unavailable" queries). Callers must fall back to CPU.
 */
#ifndef TINYGIANT_METAL_ATTN_H
#define TINYGIANT_METAL_ATTN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 if a Metal device exists and shaders compiled; 0 otherwise. */
int tg_metal_available(void);

/* Create a context for n_layers of GQA attention. Returns NULL on failure. */
void *tg_metal_create(int n_layers, int embed_dim, int n_heads, int n_kv_heads,
                      int head_dim, int kv_max);

void tg_metal_destroy(void *ctx);

/* Upload FP16 row-major weights for one layer. Migrates ownership to Metal
 * shared buffers — caller should drop host copies after success (no double
 * residency). Layout: Wq [n_heads*head_dim, embed], etc.
 * Returns 0 on success. */
int tg_metal_upload_layer_f16(
    void *ctx, int layer,
    const uint16_t *wq, const uint16_t *wk,
    const uint16_t *wv, const uint16_t *wo,
    const float *q_norm_w, const float *k_norm_w);

/* RoPE cos/sin tables: [max_pos, head_dim/2] each. Returns 0 on success. */
int tg_metal_set_rope(void *ctx, const float *cos_table, const float *sin_table,
                      int max_pos);

void tg_metal_reset_kv(void *ctx);

/* Full attention decode matching tg_attention_decode semantics (FP16 weights).
 * input/output are embed_dim float32. pos == current KV length before append.
 * Returns 0 on success. */
int tg_metal_attention_decode(void *ctx, int layer, const float *input,
                              float *output, int pos);

/* Human-readable backend status for logging (static string, never NULL). */
const char *tg_metal_status(void);

#ifdef __cplusplus
}
#endif

#endif /* TINYGIANT_METAL_ATTN_H */

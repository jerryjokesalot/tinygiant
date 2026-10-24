/* metal_attn_stub.c — non-Darwin / no-Metal stand-in. Always unavailable. */
#include "metal_attn.h"

int tg_metal_available(void) { return 0; }

void *tg_metal_create(int n_layers, int embed_dim, int n_heads, int n_kv_heads,
                      int head_dim, int kv_max) {
    (void)n_layers; (void)embed_dim; (void)n_heads; (void)n_kv_heads;
    (void)head_dim; (void)kv_max;
    return 0;
}

void tg_metal_destroy(void *ctx) { (void)ctx; }

int tg_metal_upload_layer_f16(
    void *ctx, int layer,
    const uint16_t *wq, const uint16_t *wk,
    const uint16_t *wv, const uint16_t *wo,
    const float *q_norm_w, const float *k_norm_w) {
    (void)ctx; (void)layer; (void)wq; (void)wk; (void)wv; (void)wo;
    (void)q_norm_w; (void)k_norm_w;
    return -1;
}

int tg_metal_upload_layer_q4(
    void *ctx, int layer,
    const uint8_t *wq_q4, const uint8_t *wk_q4,
    const uint16_t *wv_f16, const uint8_t *wo_q4,
    const float *q_norm_w, const float *k_norm_w) {
    (void)ctx; (void)layer; (void)wq_q4; (void)wk_q4; (void)wv_f16;
    (void)wo_q4; (void)q_norm_w; (void)k_norm_w;
    return -1;
}

int tg_metal_set_rope(void *ctx, const float *cos_table, const float *sin_table,
                      int max_pos) {
    (void)ctx; (void)cos_table; (void)sin_table; (void)max_pos;
    return -1;
}

void tg_metal_reset_kv(void *ctx) { (void)ctx; }

int tg_metal_attention_decode(void *ctx, int layer, const float *input,
                              float *output, int pos) {
    (void)ctx; (void)layer; (void)input; (void)output; (void)pos;
    return -1;
}

void tg_metal_timing(void *ctx, double *decode_ms_sum, uint64_t *n_calls) {
    (void)ctx;
    if (decode_ms_sum) *decode_ms_sum = 0;
    if (n_calls) *n_calls = 0;
}

void tg_metal_timing_reset(void *ctx) { (void)ctx; }

const char *tg_metal_status(void) {
    return "Metal attn stub (not built for this platform)";
}

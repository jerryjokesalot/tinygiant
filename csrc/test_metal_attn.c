/* test_metal_attn.c — Metal vs CPU tg_attention_decode numerical check.
 *
 * Build (Darwin): make test-metal
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metal_attn.h"

/* Pull CPU reference from the library object via dynamic link — declare. */
void tg_attention_decode(
    const float *wq, const float *wk, const float *wv, const float *wo,
    const float *q_norm_w, const float *k_norm_w,
    float *kv_k, float *kv_v, int kv_len, int kv_max,
    const float *rope_cos, const float *rope_sin,
    const float *input, float *output,
    int embed_dim, int n_heads, int n_kv_heads, int head_dim, int pos);

static float frand(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.f - 1.f;
}

static void fill(float *a, int n) {
    for (int i = 0; i < n; i++) a[i] = frand() * 0.05f;
}

static uint16_t f32_to_f16(float f) {
    /* Simple round-to-nearest even-ish via truncating IEEE bits. */
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((int32_t)((x >> 23) & 0xff)) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

int main(void) {
    if (!tg_metal_available()) {
        printf("SKIP: Metal not available (%s)\n", tg_metal_status());
        return 0;
    }
    printf("Metal: %s\n", tg_metal_status());

    const int embed = 64;
    const int n_heads = 4;
    const int n_kv = 2;
    const int hd = 16;
    const int kv_max = 32;
    const int q_dim = n_heads * hd;
    const int kv_dim = n_kv * hd;
    const int max_pos = 64;

    srand(42);

    float *wq = calloc((size_t)q_dim * embed, sizeof(float));
    float *wk = calloc((size_t)kv_dim * embed, sizeof(float));
    float *wv = calloc((size_t)kv_dim * embed, sizeof(float));
    float *wo = calloc((size_t)embed * q_dim, sizeof(float));
    float *q_norm = calloc(hd, sizeof(float));
    float *k_norm = calloc(hd, sizeof(float));
    float *input = calloc(embed, sizeof(float));
    float *out_cpu = calloc(embed, sizeof(float));
    float *out_metal = calloc(embed, sizeof(float));
    float *kv_k = calloc((size_t)n_kv * kv_max * hd, sizeof(float));
    float *kv_v = calloc((size_t)n_kv * kv_max * hd, sizeof(float));
    float *rope_cos = calloc((size_t)max_pos * (hd / 2), sizeof(float));
    float *rope_sin = calloc((size_t)max_pos * (hd / 2), sizeof(float));

    fill(wq, q_dim * embed);
    fill(wk, kv_dim * embed);
    fill(wv, kv_dim * embed);
    fill(wo, embed * q_dim);
    for (int i = 0; i < hd; i++) { q_norm[i] = 1.f; k_norm[i] = 1.f; }
    fill(input, embed);
    for (int p = 0; p < max_pos; p++) {
        for (int i = 0; i < hd / 2; i++) {
            float ang = (float)p / powf(10000.f, (float)i / (float)(hd / 2));
            rope_cos[p * (hd / 2) + i] = cosf(ang);
            rope_sin[p * (hd / 2) + i] = sinf(ang);
        }
    }

    /* FP16 copies for Metal (quantization noise expected). */
    size_t n_wq = (size_t)q_dim * embed;
    size_t n_wk = (size_t)kv_dim * embed;
    size_t n_wo = (size_t)embed * q_dim;
    uint16_t *wq16 = malloc(n_wq * 2);
    uint16_t *wk16 = malloc(n_wk * 2);
    uint16_t *wv16 = malloc(n_wk * 2);
    uint16_t *wo16 = malloc(n_wo * 2);
    for (size_t i = 0; i < n_wq; i++) wq16[i] = f32_to_f16(wq[i]);
    for (size_t i = 0; i < n_wk; i++) {
        wk16[i] = f32_to_f16(wk[i]);
        wv16[i] = f32_to_f16(wv[i]);
    }
    for (size_t i = 0; i < n_wo; i++) wo16[i] = f32_to_f16(wo[i]);

    void *mctx = tg_metal_create(1, embed, n_heads, n_kv, hd, kv_max);
    if (!mctx) {
        fprintf(stderr, "FAIL: tg_metal_create\n");
        return 1;
    }
    if (tg_metal_set_rope(mctx, rope_cos, rope_sin, max_pos) != 0) {
        fprintf(stderr, "FAIL: set_rope\n");
        return 1;
    }
    if (tg_metal_upload_layer_f16(mctx, 0, wq16, wk16, wv16, wo16,
                                  q_norm, k_norm) != 0) {
        fprintf(stderr, "FAIL: upload\n");
        return 1;
    }

    const int steps = 4;
    float max_abs = 0.f, max_rel = 0.f;
    for (int pos = 0; pos < steps; pos++) {
        /* Fresh-ish input each step */
        fill(input, embed);
        memset(out_cpu, 0, (size_t)embed * sizeof(float));
        memset(out_metal, 0, (size_t)embed * sizeof(float));

        tg_attention_decode(wq, wk, wv, wo, q_norm, k_norm,
                            kv_k, kv_v, pos, kv_max,
                            rope_cos, rope_sin,
                            input, out_cpu,
                            embed, n_heads, n_kv, hd, pos);

        if (tg_metal_attention_decode(mctx, 0, input, out_metal, pos) != 0) {
            fprintf(stderr, "FAIL: metal decode pos=%d (%s)\n", pos,
                    tg_metal_status());
            return 1;
        }

        for (int i = 0; i < embed; i++) {
            float a = out_cpu[i], b = out_metal[i];
            float absd = fabsf(a - b);
            float rel = absd / fmaxf(1e-3f, fabsf(a));
            if (absd > max_abs) max_abs = absd;
            if (rel > max_rel) max_rel = rel;
        }
        printf("pos=%d  cpu[0]=%.5f metal[0]=%.5f\n", pos, out_cpu[0], out_metal[0]);
    }

    printf("max_abs=%.6f max_rel=%.6f\n", max_abs, max_rel);
    /* FP16 weight path: allow modest tolerance */
    if (max_abs > 5e-2f && max_rel > 0.15f) {
        fprintf(stderr, "FAIL: disagreement too large\n");
        tg_metal_destroy(mctx);
        return 1;
    }
    printf("PASS: Metal attn numerically close to CPU (FP16 weights)\n");
    tg_metal_destroy(mctx);
    return 0;
}

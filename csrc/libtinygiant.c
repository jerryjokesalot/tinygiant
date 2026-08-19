/*
 * libtinygiant.c — Fused Q4_K × Q8 inference library
 *
 * Correct Q4_K_M handling with sub-block scales (8 sub-blocks per 256-value block).
 * ARM NEON + vdotq_s32 for the hot path.
 *
 * Build (macOS):
 *   clang -shared -O3 -mcpu=apple-m1 -o libtinygiant.dylib tools/libtinygiant.c
 *
 * Build (Linux aarch64):
 *   gcc -shared -fPIC -O3 -march=armv8.2-a+dotprod -o libtinygiant.so tools/libtinygiant.c
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <arm_neon.h>

#define QK_K      256
#define Q4K_BSIZE 144   /* 2(d) + 2(dmin) + 12(scales) + 128(qs) */

/* ═══ f16 → f32 conversion ═══ */

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) {
        if (m == 0) { float r; uint32_t v = s; memcpy(&r, &v, 4); return r; }
        while (!(m & 0x400)) { m <<= 1; e--; }
        e++; m &= ~0x400;
    } else if (e == 31) { e = 255; }
    uint32_t f = s | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}

/* ═══ Q4_K scale extraction ═══
 *
 * 12-byte scales array encodes 8 (scale, min) pairs, each 6 bits.
 * Bytes 0-3: lower 6 bits = sc[0..3], upper 2 bits = extra for sc[4..7]
 * Bytes 4-7: lower 6 bits = m[0..3],  upper 2 bits = extra for m[4..7]
 * Bytes 8-11: lower 4 bits = sc[4..7] low4, upper 4 bits = m[4..7] low4
 */
static inline void get_scale_min_k4(int j, const uint8_t *scales,
                                     uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >>  4) | ((scales[j    ] >> 6) << 4);
    }
}

/* ═══ Q8 quantization ═══ */

typedef struct {
    float   scale;
    int32_t sum;
    int8_t  qs[QK_K];
} q8_block;

static void quantize_q8(const float *x, q8_block *out, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * QK_K;

        float32x4_t vmax = vdupq_n_f32(0);
        for (int i = 0; i < QK_K; i += 4) {
            float32x4_t v = vld1q_f32(xb + i);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        float amax = vmaxvq_f32(vmax);

        float sc = amax / 127.f;
        float inv = amax > 0 ? 127.f / amax : 0;
        out[b].scale = sc;

        int32x4_t vsum = vdupq_n_s32(0);
        float32x4_t vinv = vdupq_n_f32(inv);
        for (int i = 0; i < QK_K; i += 4) {
            float32x4_t v = vld1q_f32(xb + i);
            float32x4_t scaled = vmulq_f32(v, vinv);
            int32x4_t rounded = vcvtnq_s32_f32(scaled);
            int16x4_t n16 = vmovn_s32(rounded);
            int8x8_t n8 = vmovn_s16(vcombine_s16(n16, n16));
            vst1_lane_s32((int32_t *)(out[b].qs + i), vreinterpret_s32_s8(n8), 0);
            vsum = vaddq_s32(vsum, rounded);
        }
        out[b].sum = vaddvq_s32(vsum);
    }
}

/* ═══ Q4_K × Q8 dot product (correct sub-block scales) ═══
 *
 * Each Q4_K block: 256 values in 8 sub-blocks of 32.
 * Layout of qs[128]:
 *   bytes 0-31:   lo nibbles → values   0-31  (sub-block 0)
 *                 hi nibbles → values  32-63  (sub-block 1)
 *   bytes 32-63:  lo nibbles → values  64-95  (sub-block 2)
 *                 hi nibbles → values  96-127 (sub-block 3)
 *   bytes 64-95:  lo nibbles → values 128-159 (sub-block 4)
 *                 hi nibbles → values 160-191 (sub-block 5)
 *   bytes 96-127: lo nibbles → values 192-223 (sub-block 6)
 *                 hi nibbles → values 224-255 (sub-block 7)
 */
static float dot_q4k_q8(const uint8_t *q4, const q8_block *xq, int bpr) {
    float total = 0;
    const uint8x16_t m0f = vdupq_n_u8(0x0F);
    const int8x16_t ones = vdupq_n_s8(1);

    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q4 + b * Q4K_BSIZE;
        uint16_t dr, dmr;
        memcpy(&dr, bl, 2);
        memcpy(&dmr, bl + 2, 2);
        float d    = f16_to_f32(dr);
        float dmin = f16_to_f32(dmr);
        const uint8_t *scales = bl + 4;
        const uint8_t *qs = bl + 16;
        const int8_t *xqs = xq[b].qs;
        float sx = xq[b].scale;

        float block_sum = 0;

        /* 4 groups of 32 bytes, each producing 2 sub-blocks of 32 values */
        for (int g = 0; g < 4; g++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);

            const uint8_t *qs_g = qs + g * 32;
            /* lo nibbles correspond to values at offset g*64 */
            const int8_t *xq_lo = xqs + g * 64;
            /* hi nibbles correspond to values at offset g*64 + 32 */
            const int8_t *xq_hi = xqs + g * 64 + 32;

            int32x4_t idot_lo = vdupq_n_s32(0);
            int32x4_t idot_hi = vdupq_n_s32(0);
            int32x4_t isum_lo = vdupq_n_s32(0);
            int32x4_t isum_hi = vdupq_n_s32(0);

            /* Process 32 bytes in two 16-byte chunks */
            for (int j = 0; j < 32; j += 16) {
                uint8x16_t raw = vld1q_u8(qs_g + j);
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(raw, m0f));
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));

                int8x16_t x_lo = vld1q_s8(xq_lo + j);
                int8x16_t x_hi = vld1q_s8(xq_hi + j);

                idot_lo = vdotq_s32(idot_lo, lo, x_lo);
                idot_hi = vdotq_s32(idot_hi, hi, x_hi);
                isum_lo = vdotq_s32(isum_lo, x_lo, ones);
                isum_hi = vdotq_s32(isum_hi, x_hi, ones);
            }

            int32_t dot_lo = vaddvq_s32(idot_lo);
            int32_t dot_hi = vaddvq_s32(idot_hi);
            int32_t sum_lo = vaddvq_s32(isum_lo);
            int32_t sum_hi = vaddvq_s32(isum_hi);

            block_sum += d * (float)sc_lo * (float)dot_lo
                       - dmin * (float)m_lo * (float)sum_lo;
            block_sum += d * (float)sc_hi * (float)dot_hi
                       - dmin * (float)m_hi * (float)sum_hi;
        }

        total += sx * block_sum;
    }
    return total;
}

/* ═══ Matrix-vector multiply: Q4_K matrix × float vector ═══ */

static void matmul_q4k(const uint8_t *q4_matrix, const float *input,
                        float *output, int nrows, int ncols) {
    int bpr = ncols / QK_K;
    int n_q8 = bpr;

    /* Quantize input to Q8 once */
    q8_block xq[bpr];
    quantize_q8(input, xq, n_q8);

    for (int r = 0; r < nrows; r++) {
        output[r] = dot_q4k_q8(q4_matrix + (size_t)r * bpr * Q4K_BSIZE, xq, bpr);
    }
}

void tg_matmul_q4k(const uint8_t *q4_matrix, const float *input,
                    float *output, int nrows, int ncols) {
    matmul_q4k(q4_matrix, input, output, nrows, ncols);
}

/* ═══ Q6_K × Q8 fused kernel ═══ */

#define Q6K_BSIZE 210   /* 128(ql) + 64(qh) + 16(scales) + 2(d) */

static float dot_q6k_q8(const uint8_t *q6, const q8_block *xq, int bpr) {
    float total = 0;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q6 + b * Q6K_BSIZE;
        const uint8_t *ql = bl;
        const uint8_t *qh = bl + 128;
        const int8_t *sc = (const int8_t *)(bl + 192);
        uint16_t dr;
        memcpy(&dr, bl + 208, 2);
        float d = f16_to_f32(dr);
        float sx = xq[b].scale;
        const int8_t *xqs = xq[b].qs;

        float block_sum = 0;

        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const int8_t *sc_h = sc + half * 8;
            const int8_t *xq_h = xqs + half * 128;

            for (int l = 0; l < 32; l += 16) {
                int is_base = l / 16;

                int8_t v1[16], v2[16], v3[16], v4[16];
                for (int k = 0; k < 16; k++) {
                    int j = l + k;
                    uint8_t lo4_0 = ql_h[j] & 0xF;
                    uint8_t lo4_1 = ql_h[j + 32] & 0xF;
                    uint8_t hi4_0 = ql_h[j] >> 4;
                    uint8_t hi4_1 = ql_h[j + 32] >> 4;
                    uint8_t qh_byte = qh_h[j];
                    v1[k] = (int8_t)((lo4_0 | (((qh_byte >> 0) & 3) << 4)) - 32);
                    v2[k] = (int8_t)((lo4_1 | (((qh_byte >> 2) & 3) << 4)) - 32);
                    v3[k] = (int8_t)((hi4_0 | (((qh_byte >> 4) & 3) << 4)) - 32);
                    v4[k] = (int8_t)((hi4_1 | (((qh_byte >> 6) & 3) << 4)) - 32);
                }

                int8x16_t vv1 = vld1q_s8(v1);
                int8x16_t vv2 = vld1q_s8(v2);
                int8x16_t vv3 = vld1q_s8(v3);
                int8x16_t vv4 = vld1q_s8(v4);

                int8x16_t x1 = vld1q_s8(xq_h + l);
                int8x16_t x2 = vld1q_s8(xq_h + l + 32);
                int8x16_t x3 = vld1q_s8(xq_h + l + 64);
                int8x16_t x4 = vld1q_s8(xq_h + l + 96);

                int32x4_t d1 = vdotq_s32(vdupq_n_s32(0), vv1, x1);
                int32x4_t d2 = vdotq_s32(vdupq_n_s32(0), vv2, x2);
                int32x4_t d3 = vdotq_s32(vdupq_n_s32(0), vv3, x3);
                int32x4_t d4 = vdotq_s32(vdupq_n_s32(0), vv4, x4);

                block_sum += (float)sc_h[is_base + 0] * (float)vaddvq_s32(d1);
                block_sum += (float)sc_h[is_base + 2] * (float)vaddvq_s32(d2);
                block_sum += (float)sc_h[is_base + 4] * (float)vaddvq_s32(d3);
                block_sum += (float)sc_h[is_base + 6] * (float)vaddvq_s32(d4);
            }
        }
        total += d * sx * block_sum;
    }
    return total;
}

static void matmul_q6k(const uint8_t *q6_matrix, const float *input,
                        float *output, int nrows, int ncols) {
    int bpr = ncols / QK_K;
    q8_block xq[bpr];
    quantize_q8(input, xq, bpr);
    for (int r = 0; r < nrows; r++) {
        output[r] = dot_q6k_q8(q6_matrix + (size_t)r * bpr * Q6K_BSIZE, xq, bpr);
    }
}

void tg_matmul_q6k(const uint8_t *q6_matrix, const float *input,
                    float *output, int nrows, int ncols) {
    matmul_q6k(q6_matrix, input, output, nrows, ncols);
}

/* ═══ f16 matrix-vector multiply ═══ */

static void matvec_f16(const uint16_t *A, const float *x,
                        float *out, int nrows, int ncols) {
    for (int r = 0; r < nrows; r++) {
        const uint16_t *row = A + (size_t)r * ncols;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 7 < ncols; i += 8) {
            uint16x4_t h0 = vld1_u16(row + i);
            uint16x4_t h1 = vld1_u16(row + i + 4);
            float32x4_t a0 = vcvt_f32_f16(vreinterpret_f16_u16(h0));
            float32x4_t a1 = vcvt_f32_f16(vreinterpret_f16_u16(h1));
            acc0 = vfmaq_f32(acc0, a0, vld1q_f32(x + i));
            acc1 = vfmaq_f32(acc1, a1, vld1q_f32(x + i + 4));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; i < ncols; i++) sum += f16_to_f32(row[i]) * x[i];
        out[r] = sum;
    }
}

void tg_f16_matvec(const uint16_t *A, const float *x,
                    float *out, int nrows, int ncols) {
    matvec_f16(A, x, out, nrows, ncols);
}

/* ═══ f32 matrix-vector multiply ═══ */

void tg_f32_matvec(const float *A, const float *x,
                    float *out, int nrows, int ncols) {
    for (int r = 0; r < nrows; r++) {
        const float *row = A + (size_t)r * ncols;
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 7 < ncols; i += 8) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(row + i),     vld1q_f32(x + i));
            acc1 = vfmaq_f32(acc1, vld1q_f32(row + i + 4), vld1q_f32(x + i + 4));
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; i < ncols; i++) sum += row[i] * x[i];
        out[r] = sum;
    }
}

/* ═══ RMS Norm ═══ */

void tg_rms_norm(const float *x, const float *weight, float *out,
                  int dim, float eps) {
    float32x4_t vsum = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vsum = vfmaq_f32(vsum, v, v);
    }
    float ss = vaddvq_f32(vsum);
    for (; i < dim; i++) ss += x[i] * x[i];
    float scale = 1.f / sqrtf(ss / dim + eps);
    float32x4_t vs = vdupq_n_f32(scale);
    for (i = 0; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, vs), w));
    }
    for (; i < dim; i++) out[i] = x[i] * scale * weight[i];
}

/* ═══ RoPE ═══ */

static void apply_rope_inplace(float *x, int n_heads, int head_dim,
                                const float *cos_table, const float *sin_table,
                                int pos) {
    int half = head_dim / 2;
    const float *c = cos_table + pos * half;
    const float *s = sin_table + pos * half;
    for (int h = 0; h < n_heads; h++) {
        float *xh = x + h * head_dim;
        for (int i = 0; i < half; i += 4) {
            float32x4_t x1 = vld1q_f32(xh + i);
            float32x4_t x2 = vld1q_f32(xh + half + i);
            float32x4_t cv = vld1q_f32(c + i);
            float32x4_t sv = vld1q_f32(s + i);
            vst1q_f32(xh + i,        vmlsq_f32(vmulq_f32(x1, cv), x2, sv));
            vst1q_f32(xh + half + i, vfmaq_f32(vmulq_f32(x2, cv), x1, sv));
        }
    }
}

/* ═══ GQA Attention Decode Step ═══ */

void tg_attention_decode(
    const float *wq, const float *wk, const float *wv, const float *wo,
    const float *q_norm_w, const float *k_norm_w,
    float *kv_k, float *kv_v, int kv_len, int kv_max,
    const float *rope_cos, const float *rope_sin,
    const float *input, float *output,
    int embed_dim, int n_heads, int n_kv_heads, int head_dim, int pos)
{
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;

    float q[q_dim], k[kv_dim], v[kv_dim];

    /* Q/K/V projections */
    tg_f32_matvec(wq, input, q, q_dim, embed_dim);
    tg_f32_matvec(wk, input, k, kv_dim, embed_dim);
    tg_f32_matvec(wv, input, v, kv_dim, embed_dim);

    /* QK norms */
    float q_normed[q_dim], k_normed[kv_dim];
    for (int h = 0; h < n_heads; h++)
        tg_rms_norm(q + h * head_dim, q_norm_w, q_normed + h * head_dim, head_dim, 1e-6f);
    for (int h = 0; h < n_kv_heads; h++)
        tg_rms_norm(k + h * head_dim, k_norm_w, k_normed + h * head_dim, head_dim, 1e-6f);

    /* RoPE */
    apply_rope_inplace(q_normed, n_heads, head_dim, rope_cos, rope_sin, pos);
    apply_rope_inplace(k_normed, n_kv_heads, head_dim, rope_cos, rope_sin, pos);

    /* Append to KV cache: kv_k/kv_v are [n_kv_heads, kv_max, head_dim] */
    for (int h = 0; h < n_kv_heads; h++) {
        memcpy(kv_k + ((size_t)h * kv_max + kv_len) * head_dim,
               k_normed + h * head_dim, head_dim * sizeof(float));
        memcpy(kv_v + ((size_t)h * kv_max + kv_len) * head_dim,
               v + h * head_dim, head_dim * sizeof(float));
    }

    /* GQA attention */
    float attn_out[q_dim];
    int seq_len = kv_len + 1;
    float inv_sqrt = 1.f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;
        const float *q_h = q_normed + h * head_dim;
        const float *k_cache = kv_k + (size_t)kv_h * kv_max * head_dim;
        const float *v_cache = kv_v + (size_t)kv_h * kv_max * head_dim;

        /* Compute attention scores */
        float scores[seq_len];
        float max_score = -1e30f;
        for (int t = 0; t < seq_len; t++) {
            float32x4_t acc = vdupq_n_f32(0);
            const float *kt = k_cache + t * head_dim;
            int d = 0;
            for (; d + 3 < head_dim; d += 4)
                acc = vfmaq_f32(acc, vld1q_f32(q_h + d), vld1q_f32(kt + d));
            float dot = vaddvq_f32(acc);
            for (; d < head_dim; d++) dot += q_h[d] * kt[d];
            scores[t] = dot * inv_sqrt;
            if (scores[t] > max_score) max_score = scores[t];
        }

        /* Softmax */
        float sum_exp = 0;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = 1.f / sum_exp;

        /* Weighted value sum */
        float *out_h = attn_out + h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            float w = scores[t] * inv_sum;
            const float *vt = v_cache + t * head_dim;
            float32x4_t vw = vdupq_n_f32(w);
            int d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float32x4_t cur = vld1q_f32(out_h + d);
                cur = vfmaq_f32(cur, vw, vld1q_f32(vt + d));
                vst1q_f32(out_h + d, cur);
            }
            for (; d < head_dim; d++) out_h[d] += w * vt[d];
        }
    }

    /* Output projection */
    tg_f32_matvec(wo, attn_out, output, embed_dim, q_dim);
}

/* ═══ GQA Attention Decode Step (Q4_K weights) ═══ */

void tg_attention_decode_q4(
    const uint8_t *wq_q4, const uint8_t *wk_q4,
    const uint16_t *wv_f16, const uint8_t *wo_q4,
    const float *q_norm_w, const float *k_norm_w,
    float *kv_k, float *kv_v, int kv_len, int kv_max,
    const float *rope_cos, const float *rope_sin,
    const float *input, float *output,
    int embed_dim, int n_heads, int n_kv_heads, int head_dim, int pos)
{
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;

    float q[q_dim], k[kv_dim], v[kv_dim];

    /* Q/K projections via fused Q4×Q8 kernel */
    matmul_q4k(wq_q4, input, q, q_dim, embed_dim);
    matmul_q4k(wk_q4, input, k, kv_dim, embed_dim);

    /* V projection via f16 matvec */
    matvec_f16(wv_f16, input, v, kv_dim, embed_dim);

    /* QK norms */
    float q_normed[q_dim], k_normed[kv_dim];
    for (int h = 0; h < n_heads; h++)
        tg_rms_norm(q + h * head_dim, q_norm_w, q_normed + h * head_dim, head_dim, 1e-6f);
    for (int h = 0; h < n_kv_heads; h++)
        tg_rms_norm(k + h * head_dim, k_norm_w, k_normed + h * head_dim, head_dim, 1e-6f);

    /* RoPE */
    apply_rope_inplace(q_normed, n_heads, head_dim, rope_cos, rope_sin, pos);
    apply_rope_inplace(k_normed, n_kv_heads, head_dim, rope_cos, rope_sin, pos);

    /* Append to KV cache */
    for (int h = 0; h < n_kv_heads; h++) {
        memcpy(kv_k + ((size_t)h * kv_max + kv_len) * head_dim,
               k_normed + h * head_dim, head_dim * sizeof(float));
        memcpy(kv_v + ((size_t)h * kv_max + kv_len) * head_dim,
               v + h * head_dim, head_dim * sizeof(float));
    }

    /* GQA attention */
    float attn_out[q_dim];
    int seq_len = kv_len + 1;
    float inv_sqrt = 1.f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;
        const float *q_h = q_normed + h * head_dim;
        const float *k_cache = kv_k + (size_t)kv_h * kv_max * head_dim;
        const float *v_cache = kv_v + (size_t)kv_h * kv_max * head_dim;

        float scores[seq_len];
        float max_score = -1e30f;
        for (int t = 0; t < seq_len; t++) {
            float32x4_t acc = vdupq_n_f32(0);
            const float *kt = k_cache + t * head_dim;
            int d = 0;
            for (; d + 3 < head_dim; d += 4)
                acc = vfmaq_f32(acc, vld1q_f32(q_h + d), vld1q_f32(kt + d));
            float dot = vaddvq_f32(acc);
            for (; d < head_dim; d++) dot += q_h[d] * kt[d];
            scores[t] = dot * inv_sqrt;
            if (scores[t] > max_score) max_score = scores[t];
        }

        float sum_exp = 0;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = 1.f / sum_exp;

        float *out_h = attn_out + h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            float w = scores[t] * inv_sum;
            const float *vt = v_cache + t * head_dim;
            float32x4_t vw = vdupq_n_f32(w);
            int d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float32x4_t cur = vld1q_f32(out_h + d);
                cur = vfmaq_f32(cur, vw, vld1q_f32(vt + d));
                vst1q_f32(out_h + d, cur);
            }
            for (; d < head_dim; d++) out_h[d] += w * vt[d];
        }
    }

    /* Output projection via fused Q4×Q8 */
    matmul_q4k(wo_q4, attn_out, output, embed_dim, q_dim);
}

/* ═══ Expert forward pass (original, all-Q4_K) ═══ */

void tg_expert_forward(const uint8_t *q4_data, const float *input,
                       float *output, float weight,
                       int embed_dim, int inter_dim) {
    int gate_bpr = embed_dim / QK_K;
    size_t gate_bytes = (size_t)inter_dim * gate_bpr * Q4K_BSIZE;
    int down_bpr = inter_dim / QK_K;

    const uint8_t *gate = q4_data;
    const uint8_t *up   = q4_data + gate_bytes;
    const uint8_t *down = q4_data + 2 * gate_bytes;

    q8_block xq_embed[gate_bpr];
    quantize_q8(input, xq_embed, gate_bpr);

    float gate_out[inter_dim];
    for (int r = 0; r < inter_dim; r++)
        gate_out[r] = dot_q4k_q8(gate + (size_t)r * gate_bpr * Q4K_BSIZE,
                                  xq_embed, gate_bpr);

    float up_out[inter_dim];
    for (int r = 0; r < inter_dim; r++)
        up_out[r] = dot_q4k_q8(up + (size_t)r * gate_bpr * Q4K_BSIZE,
                                xq_embed, gate_bpr);

    float hidden[inter_dim];
    for (int i = 0; i < inter_dim; i++) {
        float silu = gate_out[i] / (1.f + expf(-gate_out[i]));
        hidden[i] = silu * up_out[i];
    }

    q8_block xq_inter[down_bpr];
    quantize_q8(hidden, xq_inter, down_bpr);

    float down_out[embed_dim];
    for (int r = 0; r < embed_dim; r++)
        down_out[r] = dot_q4k_q8(down + (size_t)r * down_bpr * Q4K_BSIZE,
                                  xq_inter, down_bpr);

    for (int i = 0; i < embed_dim; i++)
        output[i] += weight * down_out[i];
}

/* ═══ Expert forward pass (mixed Q4_K gate/up + f16 or Q4_K down) ═══ */

void tg_expert_forward_mixed(
    const uint8_t *gate_q4, const uint8_t *up_q4,
    const void *down_data, int down_format,
    const float *input, float *output, float weight,
    int embed_dim, int inter_dim)
{
    int gate_bpr = embed_dim / QK_K;
    int down_bpr = inter_dim / QK_K;

    q8_block xq_embed[gate_bpr];
    quantize_q8(input, xq_embed, gate_bpr);

    float gate_out[inter_dim];
    for (int r = 0; r < inter_dim; r++)
        gate_out[r] = dot_q4k_q8(gate_q4 + (size_t)r * gate_bpr * Q4K_BSIZE,
                                  xq_embed, gate_bpr);

    float up_out[inter_dim];
    for (int r = 0; r < inter_dim; r++)
        up_out[r] = dot_q4k_q8(up_q4 + (size_t)r * gate_bpr * Q4K_BSIZE,
                                xq_embed, gate_bpr);

    float hidden[inter_dim];
    for (int i = 0; i < inter_dim; i++) {
        float silu = gate_out[i] / (1.f + expf(-gate_out[i]));
        hidden[i] = silu * up_out[i];
    }

    float down_out[embed_dim];
    if (down_format == 1) {
        matvec_f16((const uint16_t *)down_data, hidden, down_out, embed_dim, inter_dim);
    } else if (down_format == 2) {
        matmul_q6k((const uint8_t *)down_data, hidden, down_out, embed_dim, inter_dim);
    } else {
        q8_block xq_inter[down_bpr];
        quantize_q8(hidden, xq_inter, down_bpr);
        for (int r = 0; r < embed_dim; r++)
            down_out[r] = dot_q4k_q8((const uint8_t *)down_data +
                                      (size_t)r * down_bpr * Q4K_BSIZE,
                                      xq_inter, down_bpr);
    }

    for (int i = 0; i < embed_dim; i++)
        output[i] += weight * down_out[i];
}

/* ═══ Batch expert forward ═══ */

void tg_moe_forward(const uint8_t **expert_ptrs, const float *weights,
                     int n_experts, const float *input, float *output,
                     int embed_dim, int inter_dim) {
    memset(output, 0, embed_dim * sizeof(float));
    for (int e = 0; e < n_experts; e++) {
        tg_expert_forward(expert_ptrs[e], input, output, weights[e],
                          embed_dim, inter_dim);
    }
}

/* ═══ Full transformer layer ═══
 *
 * Routing is done in Python (to manage expert cache lookups).
 * This function takes pre-loaded expert pointers and weights.
 */
void tg_transformer_layer(
    /* Attention weights (f32) */
    const float *wq, const float *wk, const float *wv, const float *wo,
    const float *attn_norm_w, const float *q_norm_w, const float *k_norm_w,
    const float *ffn_norm_w,
    /* KV cache */
    float *kv_k, float *kv_v, int kv_len, int kv_max,
    const float *rope_cos, const float *rope_sin,
    /* Pre-routed expert data */
    const uint8_t **gate_ptrs, const uint8_t **up_ptrs,
    const void **down_ptrs, const int *down_is_f16,
    const float *expert_weights, int n_active,
    /* Input/output (x modified in place) */
    float *x,
    int embed_dim, int inter_dim,
    int n_heads, int n_kv_heads, int head_dim, int pos)
{
    float normed[embed_dim];

    /* Attention block */
    tg_rms_norm(x, attn_norm_w, normed, embed_dim, 1e-6f);
    float attn_out[embed_dim];
    tg_attention_decode(wq, wk, wv, wo, q_norm_w, k_norm_w,
                        kv_k, kv_v, kv_len, kv_max,
                        rope_cos, rope_sin,
                        normed, attn_out,
                        embed_dim, n_heads, n_kv_heads, head_dim, pos);
    for (int i = 0; i < embed_dim; i++) x[i] += attn_out[i];

    /* MoE block */
    tg_rms_norm(x, ffn_norm_w, normed, embed_dim, 1e-6f);
    float moe_out[embed_dim];
    memset(moe_out, 0, embed_dim * sizeof(float));
    for (int e = 0; e < n_active; e++) {
        tg_expert_forward_mixed(gate_ptrs[e], up_ptrs[e],
                                down_ptrs[e], down_is_f16[e],
                                normed, moe_out, expert_weights[e],
                                embed_dim, inter_dim);
    }
    for (int i = 0; i < embed_dim; i++) x[i] += moe_out[i];
}

/* ═══ Verification: scalar Q4_K dot product ═══ */

static float dot_q4k_scalar(const uint8_t *q4, const float *x, int bpr) {
    float total = 0;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q4 + b * Q4K_BSIZE;
        uint16_t dr, dmr;
        memcpy(&dr, bl, 2);
        memcpy(&dmr, bl + 2, 2);
        float d = f16_to_f32(dr), dmin = f16_to_f32(dmr);
        const uint8_t *scales = bl + 4;
        const uint8_t *qs = bl + 16;
        const float *xb = x + b * QK_K;

        for (int g = 0; g < 4; g++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);
            const uint8_t *qs_g = qs + g * 32;
            for (int j = 0; j < 32; j++) {
                uint8_t v = qs_g[j];
                float val_lo = d * sc_lo * (float)(v & 0xF) - dmin * m_lo;
                float val_hi = d * sc_hi * (float)(v >> 4)  - dmin * m_hi;
                total += val_lo * xb[g * 64 + j]
                       + val_hi * xb[g * 64 + 32 + j];
            }
        }
    }
    return total;
}

void tg_matmul_q4k_scalar(const uint8_t *q4_matrix, const float *input,
                           float *output, int nrows, int ncols) {
    int bpr = ncols / QK_K;
    for (int r = 0; r < nrows; r++)
        output[r] = dot_q4k_scalar(q4_matrix + (size_t)r * bpr * Q4K_BSIZE,
                                    input, bpr);
}

/* ═══ Prefetch (madvise WILLNEED) ═══ */

void tg_prefetch(const void *addr, size_t len) {
    uintptr_t page_mask = ~((uintptr_t)16384 - 1);
    void *aligned = (void *)(((uintptr_t)addr) & page_mask);
    size_t aligned_len = len + ((uintptr_t)addr - (uintptr_t)aligned);
    posix_madvise(aligned, aligned_len, POSIX_MADV_WILLNEED);
}

void tg_prefetch_batch(const void **addrs, const size_t *lens, int count) {
    uintptr_t page_mask = ~((uintptr_t)16384 - 1);
    for (int i = 0; i < count; i++) {
        void *aligned = (void *)(((uintptr_t)addrs[i]) & page_mask);
        size_t aligned_len = lens[i] + ((uintptr_t)addrs[i] - (uintptr_t)aligned);
        posix_madvise(aligned, aligned_len, POSIX_MADV_WILLNEED);
    }
}

int tg_mlock(const void *addr, size_t len) {
    uintptr_t page_mask = ~((uintptr_t)16384 - 1);
    void *aligned = (void *)(((uintptr_t)addr) & page_mask);
    size_t aligned_len = len + ((uintptr_t)addr - (uintptr_t)aligned);
    return mlock(aligned, aligned_len);
}

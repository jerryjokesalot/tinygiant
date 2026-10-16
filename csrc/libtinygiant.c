/*
 * libtinygiant.c — Fused Q4_K × Q8 inference library
 *
 * Correct Q4_K_M handling with sub-block scales (8 sub-blocks per 256-value block).
 * ARM NEON + vdotq_s32 for the hot path.
 *
 * Build (macOS):
 *   clang -dynamiclib -O3 -mcpu=apple-m1 -o libtinygiant.dylib csrc/libtinygiant.c
 *
 * Build (Linux aarch64):
 *   gcc -shared -fPIC -O3 -march=armv8.2-a+dotprod -o libtinygiant.so csrc/libtinygiant.c
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>

#if defined(__ARM_NEON)
#  include <arm_neon.h>
#elif defined(__AVX2__)
#  include <immintrin.h>
#endif

#define QK_K      256
#define Q4K_BSIZE 144   /* 2(d) + 2(dmin) + 12(scales) + 128(qs) */

/* ═══ Portable vector helpers for attention ═══ */

static inline float tg_dot_f32(const float *a, const float *b, int n) {
    int i = 0;
    float result = 0.0f;

#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    result = vaddvq_f32(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8)
        acc = _mm256_add_ps(acc,
                            _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                          _mm256_loadu_ps(b + i)));
    __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(acc),
                             _mm256_extractf128_ps(acc, 1));
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    result = _mm_cvtss_f32(sum4);
#endif

    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

static inline void tg_accumulate_scaled_f32(float *dst, const float *src,
                                             float scale, int n) {
    int i = 0;

#if defined(__ARM_NEON)
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; i + 3 < n; i += 4) {
        float32x4_t value = vld1q_f32(dst + i);
        value = vfmaq_f32(value, vscale, vld1q_f32(src + i));
        vst1q_f32(dst + i, value);
    }
#elif defined(__AVX2__)
    __m256 vscale = _mm256_set1_ps(scale);
    for (; i + 7 < n; i += 8) {
        __m256 value = _mm256_loadu_ps(dst + i);
        value = _mm256_add_ps(value,
                              _mm256_mul_ps(vscale, _mm256_loadu_ps(src + i)));
        _mm256_storeu_ps(dst + i, value);
    }
#endif

    for (; i < n; i++) dst[i] += scale * src[i];
}

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

#if defined(__ARM_NEON)
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
#elif defined(__AVX2__)
        __m256 vmax = _mm256_setzero_ps();
        __m256 sign_mask = _mm256_set1_ps(-0.0f);
        for (int i = 0; i < QK_K; i += 8) {
            __m256 v = _mm256_loadu_ps(xb + i);
            __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
            vmax = _mm256_max_ps(vmax, abs_v);
        }
        /* Horizontal max over 8 floats */
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(vmax, 0), _mm256_extractf128_ps(vmax, 1));
        max4 = _mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(1, 0, 3, 2)));
        max4 = _mm_max_ps(max4, _mm_shuffle_ps(max4, max4, _MM_SHUFFLE(2, 3, 0, 1)));
        float amax = _mm_cvtss_f32(max4);

        float sc = amax / 127.f;
        float inv = amax > 0 ? 127.f / amax : 0;
        out[b].scale = sc;

        int32_t sum = 0;
        __m256 vinv = _mm256_set1_ps(inv);
        for (int i = 0; i < QK_K; i += 8) {
            __m256 v = _mm256_loadu_ps(xb + i);
            __m256 scaled = _mm256_mul_ps(v, vinv);
            __m256i rounded = _mm256_cvtps_epi32(scaled);

            /* Pack int32 -> int16 -> int8 */
            __m128i r_lo = _mm256_castsi256_si128(rounded);
            __m128i r_hi = _mm256_extracti128_si256(rounded, 1);
            __m128i pack16 = _mm_packs_epi32(r_lo, r_hi);
            __m128i pack8 = _mm_packs_epi16(pack16, pack16);
            _mm_storel_epi64((__m128i*)(out[b].qs + i), pack8);

            /* Accumulate sum */
            sum += _mm_extract_epi32(r_lo, 0) + _mm_extract_epi32(r_lo, 1) +
                   _mm_extract_epi32(r_lo, 2) + _mm_extract_epi32(r_lo, 3) +
                   _mm_extract_epi32(r_hi, 0) + _mm_extract_epi32(r_hi, 1) +
                   _mm_extract_epi32(r_hi, 2) + _mm_extract_epi32(r_hi, 3);
        }
        out[b].sum = sum;
#else
        float amax = 0.0f;
        for (int i = 0; i < QK_K; i++) {
            float v = fabsf(xb[i]);
            if (v > amax) amax = v;
        }

        float sc = amax / 127.f;
        float inv = amax > 0 ? 127.f / amax : 0;
        out[b].scale = sc;

        int32_t sum = 0;
        for (int i = 0; i < QK_K; i++) {
            int val = (int)roundf(xb[i] * inv);
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            out[b].qs[i] = (int8_t)val;
            sum += val;
        }
        out[b].sum = sum;
#endif
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

#if defined(__ARM_NEON)
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

        for (int g = 0; g < 4; g++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);

            const uint8_t *qs_g = qs + g * 32;
            const int8_t *xq_lo = xqs + g * 64;
            const int8_t *xq_hi = xqs + g * 64 + 32;

            int32x4_t idot_lo = vdupq_n_s32(0);
            int32x4_t idot_hi = vdupq_n_s32(0);
            int32x4_t isum_lo = vdupq_n_s32(0);
            int32x4_t isum_hi = vdupq_n_s32(0);

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
#elif defined(__AVX2__)
    __m128i m0f = _mm_set1_epi8(0x0F);
    __m256i ones_16 = _mm256_set1_epi16(1);

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

        for (int g = 0; g < 4; g++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);

            const uint8_t *qs_g = qs + g * 32;
            const int8_t *xq_lo = xqs + g * 64;
            const int8_t *xq_hi = xqs + g * 64 + 32;

            /* Load 32 bytes of Q4 qs_g */
            __m128i raw_lo_16 = _mm_loadu_si128((const __m128i *)qs_g);
            __m128i raw_hi_16 = _mm_loadu_si128((const __m128i *)(qs_g + 16));

            __m256i raw = _mm256_inserti128_si256(_mm256_castsi128_si256(raw_lo_16), raw_hi_16, 1);
            __m256i q_lo = _mm256_and_si256(raw, _mm256_set1_epi8(0x0F));
            __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), _mm256_set1_epi8(0x0F));

            __m256i x_lo_v = _mm256_loadu_si256((const __m256i *)xq_lo);
            __m256i x_hi_v = _mm256_loadu_si256((const __m256i *)xq_hi);

            /* Multiply unsigned q_lo (0..15) by signed x_lo_v */
            __m256i prod_lo16 = _mm256_maddubs_epi16(q_lo, x_lo_v);
            __m256i prod_hi16 = _mm256_maddubs_epi16(q_hi, x_hi_v);

            __m256i idot_lo_256 = _mm256_madd_epi16(prod_lo16, ones_16);
            __m256i idot_hi_256 = _mm256_madd_epi16(prod_hi16, ones_16);

            /* Sum elements of idot_lo_256 & idot_hi_256 */
            __m128i sum_lo_128 = _mm_add_epi32(_mm256_castsi256_si128(idot_lo_256), _mm256_extracti128_si256(idot_lo_256, 1));
            __m128i sum_hi_128 = _mm_add_epi32(_mm256_castsi256_si128(idot_hi_256), _mm256_extracti128_si256(idot_hi_256, 1));

            int32_t dot_lo = _mm_extract_epi32(sum_lo_128, 0) + _mm_extract_epi32(sum_lo_128, 1) +
                             _mm_extract_epi32(sum_lo_128, 2) + _mm_extract_epi32(sum_lo_128, 3);
            int32_t dot_hi = _mm_extract_epi32(sum_hi_128, 0) + _mm_extract_epi32(sum_hi_128, 1) +
                             _mm_extract_epi32(sum_hi_128, 2) + _mm_extract_epi32(sum_hi_128, 3);

            /* Calculate sum_lo and sum_hi (sum of xqs elements) */
            int32_t sum_lo_val = 0, sum_hi_val = 0;
            for (int j = 0; j < 32; j++) {
                sum_lo_val += xq_lo[j];
                sum_hi_val += xq_hi[j];
            }

            block_sum += d * (float)sc_lo * (float)dot_lo
                       - dmin * (float)m_lo * (float)sum_lo_val;
            block_sum += d * (float)sc_hi * (float)dot_hi
                       - dmin * (float)m_hi * (float)sum_hi_val;
        }

        total += sx * block_sum;
    }
#else
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

        for (int g = 0; g < 4; g++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4(2 * g,     scales, &sc_lo, &m_lo);
            get_scale_min_k4(2 * g + 1, scales, &sc_hi, &m_hi);

            const uint8_t *qs_g = qs + g * 32;
            const int8_t *xq_lo = xqs + g * 64;
            const int8_t *xq_hi = xqs + g * 64 + 32;

            int32_t dot_lo = 0, dot_hi = 0;
            int32_t sum_lo = 0, sum_hi = 0;

            for (int j = 0; j < 32; j++) {
                uint8_t v = qs_g[j];
                int8_t nib_lo = v & 0xF;
                int8_t nib_hi = v >> 4;
                dot_lo += (int32_t)nib_lo * (int32_t)xq_lo[j];
                dot_hi += (int32_t)nib_hi * (int32_t)xq_hi[j];
                sum_lo += xq_lo[j];
                sum_hi += xq_hi[j];
            }

            block_sum += d * (float)sc_lo * (float)dot_lo
                       - dmin * (float)m_lo * (float)sum_lo;
            block_sum += d * (float)sc_hi * (float)dot_hi
                       - dmin * (float)m_hi * (float)sum_hi;
        }

        total += sx * block_sum;
    }
#endif
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

#if defined(__ARM_NEON)
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
#else
                int32_t s1 = 0, s2 = 0, s3 = 0, s4 = 0;
                for (int k = 0; k < 16; k++) {
                    s1 += (int32_t)v1[k] * (int32_t)xq_h[l + k];
                    s2 += (int32_t)v2[k] * (int32_t)xq_h[l + 32 + k];
                    s3 += (int32_t)v3[k] * (int32_t)xq_h[l + 64 + k];
                    s4 += (int32_t)v4[k] * (int32_t)xq_h[l + 96 + k];
                }

                block_sum += (float)sc_h[is_base + 0] * (float)s1;
                block_sum += (float)sc_h[is_base + 2] * (float)s2;
                block_sum += (float)sc_h[is_base + 4] * (float)s3;
                block_sum += (float)sc_h[is_base + 6] * (float)s4;
#endif
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
        int i = 0;
        float sum = 0;
#if defined(__ARM_NEON)
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        for (; i + 7 < ncols; i += 8) {
            uint16x4_t h0 = vld1_u16(row + i);
            uint16x4_t h1 = vld1_u16(row + i + 4);
            float32x4_t a0 = vcvt_f32_f16(vreinterpret_f16_u16(h0));
            float32x4_t a1 = vcvt_f32_f16(vreinterpret_f16_u16(h1));
            acc0 = vfmaq_f32(acc0, a0, vld1q_f32(x + i));
            acc1 = vfmaq_f32(acc1, a1, vld1q_f32(x + i + 4));
        }
        sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#elif defined(__AVX2__) && defined(__F16C__)
        __m256 acc = _mm256_setzero_ps();
        for (; i + 7 < ncols; i += 8) {
            __m128i h = _mm_loadu_si128((const __m128i *)(row + i));
            __m256 a = _mm256_cvtph_ps(h);
            __m256 xv = _mm256_loadu_ps(x + i);
            acc = _mm256_fmadd_ps(a, xv, acc);
        }
        __m128 acc4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        acc4 = _mm_add_ps(acc4, _mm_shuffle_ps(acc4, acc4, _MM_SHUFFLE(1, 0, 3, 2)));
        acc4 = _mm_add_ps(acc4, _mm_shuffle_ps(acc4, acc4, _MM_SHUFFLE(2, 3, 0, 1)));
        sum = _mm_cvtss_f32(acc4);
#endif
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
        int i = 0;
        float sum = 0;
#if defined(__ARM_NEON)
        float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
        for (; i + 7 < ncols; i += 8) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(row + i),     vld1q_f32(x + i));
            acc1 = vfmaq_f32(acc1, vld1q_f32(row + i + 4), vld1q_f32(x + i + 4));
        }
        sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#elif defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; i + 7 < ncols; i += 8) {
            __m256 a = _mm256_loadu_ps(row + i);
            __m256 xv = _mm256_loadu_ps(x + i);
            acc = _mm256_fmadd_ps(a, xv, acc);
        }
        __m128 acc4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        acc4 = _mm_add_ps(acc4, _mm_shuffle_ps(acc4, acc4, _MM_SHUFFLE(1, 0, 3, 2)));
        acc4 = _mm_add_ps(acc4, _mm_shuffle_ps(acc4, acc4, _MM_SHUFFLE(2, 3, 0, 1)));
        sum = _mm_cvtss_f32(acc4);
#endif
        for (; i < ncols; i++) sum += row[i] * x[i];
        out[r] = sum;
    }
}

/* ═══ RMS Norm ═══ */

void tg_rms_norm(const float *x, const float *weight, float *out,
                  int dim, float eps) {
    int i = 0;
    float ss = 0;
#if defined(__ARM_NEON)
    float32x4_t vsum = vdupq_n_f32(0);
    for (; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vsum = vfmaq_f32(vsum, v, v);
    }
    ss = vaddvq_f32(vsum);
#elif defined(__AVX2__)
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vsum = _mm256_fmadd_ps(v, v, vsum);
    }
    __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(vsum), _mm256_extractf128_ps(vsum, 1));
    sum4 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(1, 0, 3, 2)));
    sum4 = _mm_add_ps(sum4, _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(2, 3, 0, 1)));
    ss = _mm_cvtss_f32(sum4);
#endif
    for (; i < dim; i++) ss += x[i] * x[i];
    float scale = 1.f / sqrtf(ss / dim + eps);

    i = 0;
#if defined(__ARM_NEON)
    float32x4_t vs = vdupq_n_f32(scale);
    for (; i + 3 < dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, vs), w));
    }
#elif defined(__AVX2__)
    __m256 vs = _mm256_set1_ps(scale);
    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(v, vs), w));
    }
#endif
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
        int i = 0;
#if defined(__ARM_NEON)
        for (; i + 3 < half; i += 4) {
            float32x4_t x1 = vld1q_f32(xh + i);
            float32x4_t x2 = vld1q_f32(xh + half + i);
            float32x4_t cv = vld1q_f32(c + i);
            float32x4_t sv = vld1q_f32(s + i);
            vst1q_f32(xh + i,        vmlsq_f32(vmulq_f32(x1, cv), x2, sv));
            vst1q_f32(xh + half + i, vfmaq_f32(vmulq_f32(x2, cv), x1, sv));
        }
#elif defined(__AVX2__)
        for (; i + 7 < half; i += 8) {
            __m256 x1 = _mm256_loadu_ps(xh + i);
            __m256 x2 = _mm256_loadu_ps(xh + half + i);
            __m256 cv = _mm256_loadu_ps(c + i);
            __m256 sv = _mm256_loadu_ps(s + i);
            __m256 out1 = _mm256_sub_ps(_mm256_mul_ps(x1, cv), _mm256_mul_ps(x2, sv));
            __m256 out2 = _mm256_add_ps(_mm256_mul_ps(x2, cv), _mm256_mul_ps(x1, sv));
            _mm256_storeu_ps(xh + i, out1);
            _mm256_storeu_ps(xh + half + i, out2);
        }
#endif
        for (; i < half; i++) {
            float x1 = xh[i];
            float x2 = xh[half + i];
            xh[i]        = x1 * c[i] - x2 * s[i];
            xh[half + i] = x2 * c[i] + x1 * s[i];
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
            const float *kt = k_cache + t * head_dim;
            float dot = tg_dot_f32(q_h, kt, head_dim);
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
            tg_accumulate_scaled_f32(out_h, vt, w, head_dim);
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
            const float *kt = k_cache + t * head_dim;
            float dot = tg_dot_f32(q_h, kt, head_dim);
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
            tg_accumulate_scaled_f32(out_h, vt, w, head_dim);
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

static size_t get_page_size(void) {
#if defined(_SC_PAGESIZE)
    long sz = sysconf(_SC_PAGESIZE);
    if (sz > 0) return (size_t)sz;
#endif
    return 4096;
}

void tg_prefetch(const void *addr, size_t len) {
    size_t page_sz = get_page_size();
    uintptr_t page_mask = ~(page_sz - 1);
    void *aligned = (void *)(((uintptr_t)addr) & page_mask);
    size_t aligned_len = len + ((uintptr_t)addr - (uintptr_t)aligned);
    posix_madvise(aligned, aligned_len, POSIX_MADV_WILLNEED);
}

void tg_prefetch_batch(const void **addrs, const size_t *lens, int count) {
    size_t page_sz = get_page_size();
    uintptr_t page_mask = ~(page_sz - 1);
    for (int i = 0; i < count; i++) {
        void *aligned = (void *)(((uintptr_t)addrs[i]) & page_mask);
        size_t aligned_len = lens[i] + ((uintptr_t)addrs[i] - (uintptr_t)aligned);
        posix_madvise(aligned, aligned_len, POSIX_MADV_WILLNEED);
    }
}

int tg_mlock(const void *addr, size_t len) {
    size_t page_sz = get_page_size();
    uintptr_t page_mask = ~(page_sz - 1);
    void *aligned = (void *)(((uintptr_t)addr) & page_mask);
    size_t aligned_len = len + ((uintptr_t)addr - (uintptr_t)aligned);
    return mlock(aligned, aligned_len);
}

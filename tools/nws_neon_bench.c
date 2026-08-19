/*
 * nws_neon_bench.c — NEON-Optimized Fused Q4 MoE Benchmark
 *
 * ARM NEON intrinsics for fused Q4 dot product.
 * Process 32 Q4 values per NEON iteration (16 bytes).
 * Algebraic trick: d * dot(nibbles, x) - dmin * sum(x)
 *
 * Compile: clang -O3 -mcpu=apple-m1 -framework Accelerate -lpthread nws_neon_bench.c -o nws_neon_bench
 * Run:     ./nws_neon_bench [tokens]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <arm_neon.h>

#define EMBED     2048
#define INTER     768
#define N_EXP     128
#define N_ACT     8
#define LAYERS    48

#define QK        256
#define Q4_BSIZE  144
#define Q4_HDRSIZE 16

#define GATE_ROWS   INTER
#define GATE_COLS   EMBED
#define GATE_BPR    (GATE_COLS / QK)
#define GATE_Q4     (GATE_ROWS * GATE_BPR * Q4_BSIZE)

#define DOWN_ROWS   EMBED
#define DOWN_COLS   INTER
#define DOWN_BPR    (DOWN_COLS / QK)
#define DOWN_Q4     (DOWN_ROWS * DOWN_BPR * Q4_BSIZE)

#define EXPERT_Q4   (GATE_Q4 + GATE_Q4 + DOWN_Q4)

static const char *CACHE_FILE = "/tmp/nws_q4_fused_cache.bin";

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static void create_cache_file(void) {
    struct stat st;
    size_t expected = (size_t)N_EXP * EXPERT_Q4;
    if (stat(CACHE_FILE, &st) == 0 && (size_t)st.st_size == expected) return;
    printf("  Creating cache: %.0f MB...", expected / 1048576.); fflush(stdout);
    FILE *f = fopen(CACHE_FILE, "wb");
    uint8_t *buf = malloc(EXPERT_Q4);
    srand(42);
    for (int e = 0; e < N_EXP; e++) {
        for (int i = 0; i < EXPERT_Q4; i += Q4_BSIZE) {
            uint16_t d_f16 = 0x2C00, dmin_f16 = 0x2000;
            memcpy(buf + i, &d_f16, 2);
            memcpy(buf + i + 2, &dmin_f16, 2);
            for (int j = 4; j < Q4_BSIZE; j++) buf[i + j] = rand() & 0xFF;
        }
        fwrite(buf, 1, EXPERT_Q4, f);
    }
    fclose(f); free(buf);
    printf(" done.\n");
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { float r; uint32_t v = sign; memcpy(&r, &v, 4); return r; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) { exp = 255; }
    uint32_t f = sign | ((exp + 112) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
}

/* ═══ SCALAR baseline (for comparison) ═══ */

static float dot_q4_scalar(const uint8_t *q4_row, const float *x, int bpr) {
    float sum = 0;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *block = q4_row + b * Q4_BSIZE;
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, block, 2);
        memcpy(&dmin_raw, block + 2, 2);
        float d = f16_to_f32(d_raw);
        float dmin = f16_to_f32(dmin_raw);
        const uint8_t *qs = block + Q4_HDRSIZE;
        const float *xb = x + b * QK;
        for (int j = 0; j < QK / 2; j++) {
            uint8_t byte = qs[j];
            float v0 = d * (float)(byte & 0xF) - dmin;
            float v1 = d * (float)(byte >> 4) - dmin;
            sum += v0 * xb[j * 2] + v1 * xb[j * 2 + 1];
        }
    }
    return sum;
}

/* ═══ NEON fused Q4 dot product ═══ */

/*
 * Algebraic trick: (d*nib - dmin) * x = d*(nib*x) - dmin*x
 * So: result = d * dot(nibbles, x) - dmin * sum(x)
 *
 * Process 16 Q4 bytes (32 values) per iteration:
 * 1. vld1q_u8: load 16 bytes
 * 2. vandq/vshrq: extract lo/hi nibbles
 * 3. vzip: interleave lo[i],hi[i] to match x[2i],x[2i+1]
 * 4. vmovl: widen u8→u16→u32→f32
 * 5. vfmaq_f32: multiply-accumulate with x
 */
static float dot_q4_neon(const uint8_t *q4_row, const float *x, int bpr) {
    float total = 0;
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

    for (int b = 0; b < bpr; b++) {
        const uint8_t *block = q4_row + b * Q4_BSIZE;
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, block, 2);
        memcpy(&dmin_raw, block + 2, 2);
        float d = f16_to_f32(d_raw);
        float dmin = f16_to_f32(dmin_raw);

        const uint8_t *qs = block + Q4_HDRSIZE;
        const float *xb = x + b * QK;

        float32x4_t dot_a = vdupq_n_f32(0), dot_b = vdupq_n_f32(0);
        float32x4_t xs_a  = vdupq_n_f32(0), xs_b  = vdupq_n_f32(0);

        /* 128 Q4 bytes = 256 values. Process 16 bytes (32 vals) per iteration = 8 iters */
        for (int j = 0; j < 128; j += 16) {
            uint8x16_t raw = vld1q_u8(qs + j);
            uint8x16_t lo  = vandq_u8(raw, mask_0f);
            uint8x16_t hi  = vshrq_n_u8(raw, 4);

            /* First 8 bytes: lo[0..7], hi[0..7] → interleaved pairs */
            uint8x8_t lo_0 = vget_low_u8(lo);
            uint8x8_t hi_0 = vget_low_u8(hi);
            uint8x8x2_t z0 = vzip_u8(lo_0, hi_0);

            /* z0.val[0] = lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3 → 8 values for x[2j..2j+7] */
            {
                uint16x8_t w = vmovl_u8(z0.val[0]);
                float32x4_t fa = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
                float32x4_t fb = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
                float32x4_t xa = vld1q_f32(xb + j * 2);
                float32x4_t xb_ = vld1q_f32(xb + j * 2 + 4);
                dot_a = vfmaq_f32(dot_a, fa, xa);
                dot_b = vfmaq_f32(dot_b, fb, xb_);
                xs_a = vaddq_f32(xs_a, xa);
                xs_b = vaddq_f32(xs_b, xb_);
            }

            /* z0.val[1] = lo4,hi4,lo5,hi5,lo6,hi6,lo7,hi7 → x[2j+8..2j+15] */
            {
                uint16x8_t w = vmovl_u8(z0.val[1]);
                float32x4_t fa = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
                float32x4_t fb = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
                float32x4_t xa = vld1q_f32(xb + j * 2 + 8);
                float32x4_t xb_ = vld1q_f32(xb + j * 2 + 12);
                dot_a = vfmaq_f32(dot_a, fa, xa);
                dot_b = vfmaq_f32(dot_b, fb, xb_);
                xs_a = vaddq_f32(xs_a, xa);
                xs_b = vaddq_f32(xs_b, xb_);
            }

            /* Second 8 bytes: lo[8..15], hi[8..15] */
            uint8x8_t lo_1 = vget_high_u8(lo);
            uint8x8_t hi_1 = vget_high_u8(hi);
            uint8x8x2_t z1 = vzip_u8(lo_1, hi_1);

            {
                uint16x8_t w = vmovl_u8(z1.val[0]);
                float32x4_t fa = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
                float32x4_t fb = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
                float32x4_t xa = vld1q_f32(xb + j * 2 + 16);
                float32x4_t xb_ = vld1q_f32(xb + j * 2 + 20);
                dot_a = vfmaq_f32(dot_a, fa, xa);
                dot_b = vfmaq_f32(dot_b, fb, xb_);
                xs_a = vaddq_f32(xs_a, xa);
                xs_b = vaddq_f32(xs_b, xb_);
            }

            {
                uint16x8_t w = vmovl_u8(z1.val[1]);
                float32x4_t fa = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
                float32x4_t fb = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
                float32x4_t xa = vld1q_f32(xb + j * 2 + 24);
                float32x4_t xb_ = vld1q_f32(xb + j * 2 + 28);
                dot_a = vfmaq_f32(dot_a, fa, xa);
                dot_b = vfmaq_f32(dot_b, fb, xb_);
                xs_a = vaddq_f32(xs_a, xa);
                xs_b = vaddq_f32(xs_b, xb_);
            }
        }

        float dot_sum = vaddvq_f32(vaddq_f32(dot_a, dot_b));
        float x_sum   = vaddvq_f32(vaddq_f32(xs_a, xs_b));
        total += d * dot_sum - dmin * x_sum;
    }
    return total;
}

/* ═══ Expert forward pass ═══ */

static void expert_fwd(const uint8_t *q4,
                       float (*dot_fn)(const uint8_t *, const float *, int),
                       const float *x, float *out, float w) {
    const uint8_t *gate = q4;
    const uint8_t *up   = q4 + GATE_Q4;
    const uint8_t *down = q4 + 2 * GATE_Q4;

    float go[INTER], uo[INTER];
    for (int r = 0; r < INTER; r++)
        go[r] = dot_fn(gate + r * GATE_BPR * Q4_BSIZE, x, GATE_BPR);
    for (int r = 0; r < INTER; r++)
        uo[r] = dot_fn(up + r * GATE_BPR * Q4_BSIZE, x, GATE_BPR);
    for (int i = 0; i < INTER; i++)
        go[i] = go[i] / (1.f + expf(-go[i])) * uo[i];

    float eo[EMBED];
    for (int r = 0; r < EMBED; r++)
        eo[r] = dot_fn(down + r * DOWN_BPR * Q4_BSIZE, go, DOWN_BPR);
    for (int i = 0; i < EMBED; i++)
        out[i] += w * eo[i];
}

/* ═══ I/O Pipeline ═══ */

typedef struct {
    int expert_ids[N_ACT];
    int n_experts;
    uint8_t *result_bufs[N_ACT];
    int fd;
    int ready, shutdown;
    pthread_mutex_t mutex;
    pthread_cond_t req_cond, resp_cond;
} IoPipe;

static void *io_thread_fn(void *arg) {
    IoPipe *io = arg;
    while (1) {
        pthread_mutex_lock(&io->mutex);
        while (!io->n_experts && !io->shutdown)
            pthread_cond_wait(&io->req_cond, &io->mutex);
        if (io->shutdown) { pthread_mutex_unlock(&io->mutex); break; }
        int n = io->n_experts;
        int ids[N_ACT];
        memcpy(ids, io->expert_ids, n * sizeof(int));
        io->n_experts = 0;
        pthread_mutex_unlock(&io->mutex);
        for (int i = 0; i < n; i++)
            pread(io->fd, io->result_bufs[i], EXPERT_Q4, (off_t)ids[i] * EXPERT_Q4);
        pthread_mutex_lock(&io->mutex);
        io->ready = 1;
        pthread_cond_signal(&io->resp_cond);
        pthread_mutex_unlock(&io->mutex);
    }
    return NULL;
}

static IoPipe *io_create(void) {
    IoPipe *io = calloc(1, sizeof(IoPipe));
    pthread_mutex_init(&io->mutex, NULL);
    pthread_cond_init(&io->req_cond, NULL);
    pthread_cond_init(&io->resp_cond, NULL);
    for (int i = 0; i < N_ACT; i++) io->result_bufs[i] = malloc(EXPERT_Q4);
    io->fd = open(CACHE_FILE, O_RDONLY);
    fcntl(io->fd, F_NOCACHE, 1);
    return io;
}

static void io_submit(IoPipe *io, const int *ids, int n) {
    pthread_mutex_lock(&io->mutex);
    io->n_experts = n; memcpy(io->expert_ids, ids, n * sizeof(int));
    io->ready = 0; pthread_cond_signal(&io->req_cond);
    pthread_mutex_unlock(&io->mutex);
}

static void io_wait(IoPipe *io) {
    pthread_mutex_lock(&io->mutex);
    while (!io->ready) pthread_cond_wait(&io->resp_cond, &io->mutex);
    pthread_mutex_unlock(&io->mutex);
}

static void io_destroy(IoPipe *io) {
    pthread_mutex_lock(&io->mutex);
    io->shutdown = 1; pthread_cond_signal(&io->req_cond);
    pthread_mutex_unlock(&io->mutex);
    close(io->fd);
    for (int i = 0; i < N_ACT; i++) free(io->result_bufs[i]);
    free(io);
}

/* ═══ Token run ═══ */

typedef struct { double total_ms, compute_ms, iowait_ms; } TokenResult;

static TokenResult run_token(IoPipe *io, uint8_t *pinned, int nh, int nm,
                              const float *x, const float *w,
                              float (*dot_fn)(const uint8_t *, const float *, int)) {
    TokenResult r = {0};
    double t0 = now_us();
    for (int l = 0; l < LAYERS; l++) {
        int miss_ids[N_ACT];
        for (int i = 0; i < nm; i++) miss_ids[i] = (l * 7 + i * 13) % N_EXP;
        if (nm > 0) io_submit(io, miss_ids, nm);
        float out[EMBED]; memset(out, 0, sizeof(out));
        double tc = now_us();
        for (int e = 0; e < nh; e++) expert_fwd(pinned, dot_fn, x, out, w[e]);
        r.compute_ms += (now_us() - tc) / 1000.;
        if (nm > 0) {
            double tw = now_us();
            io_wait(io);
            r.iowait_ms += (now_us() - tw) / 1000.;
            tc = now_us();
            for (int i = 0; i < nm; i++) expert_fwd(io->result_bufs[i], dot_fn, x, out, w[nh + i]);
            r.compute_ms += (now_us() - tc) / 1000.;
        }
    }
    r.total_ms = (now_us() - t0) / 1000.;
    return r;
}

/* ═══ Main ═══ */

int main(int argc, char *argv[]) {
    int n_tok = argc > 1 ? atoi(argv[1]) : 3;

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  NEON Fused Q4 Benchmark — Scalar vs NEON Head-to-Head   ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("  Model:   Qwen3-30B-A3B (%d layers, %d experts, %d active)\n", LAYERS, N_EXP, N_ACT);
    printf("  Expert:  %.2f MB Q4\n", EXPERT_Q4 / 1048576.);
    printf("  Tokens:  %d per test\n\n", n_tok);

    create_cache_file();

    IoPipe *io = io_create();
    pthread_t tid;
    pthread_create(&tid, NULL, io_thread_fn, io);

    uint8_t *pinned = malloc(EXPERT_Q4);
    pread(io->fd, pinned, EXPERT_Q4, 0);

    float x[EMBED];
    srand(42);
    for (int i = 0; i < EMBED; i++)
        x[i] = ((float)rand() / (float)0x7fffffff - .5f) * .01f;
    float w[N_ACT] = {.15f, .14f, .13f, .13f, .12f, .11f, .11f, .11f};

    /* ═══ Correctness check ═══ */
    printf("  Correctness: ");
    {
        float s = dot_q4_scalar(pinned, x, GATE_BPR);
        float n = dot_q4_neon(pinned, x, GATE_BPR);
        float diff = fabsf(s - n) / (fabsf(s) + 1e-10f);
        printf("scalar=%.6f neon=%.6f rel_err=%.2e %s\n\n",
               s, n, diff, diff < 1e-4 ? "OK" : "MISMATCH!");
    }

    /* ═══ Micro-benchmark: single dot product ═══ */
    printf("  Micro-bench (single dot product, gate matrix = %d rows × %d cols):\n", GATE_ROWS, GATE_COLS);
    {
        int reps = 10000;
        volatile float sink;

        double t0 = now_us();
        for (int i = 0; i < reps; i++) sink = dot_q4_scalar(pinned, x, GATE_BPR);
        double scalar_us = (now_us() - t0) / reps;

        t0 = now_us();
        for (int i = 0; i < reps; i++) sink = dot_q4_neon(pinned, x, GATE_BPR);
        double neon_us = (now_us() - t0) / reps;

        (void)sink;
        printf("    Scalar:  %.2f µs/dot\n", scalar_us);
        printf("    NEON:    %.2f µs/dot\n", neon_us);
        printf("    Speedup: %.1fx\n\n", scalar_us / neon_us);
    }

    /* ═══ Full token: scalar vs NEON ═══ */
    printf("  Full token (all %d layers, %d experts, all in RAM):\n", LAYERS, N_ACT);
    {
        /* warmup */
        run_token(io, pinned, N_ACT, 0, x, w, dot_q4_scalar);
        run_token(io, pinned, N_ACT, 0, x, w, dot_q4_neon);

        double s_total = 0, n_total = 0;
        for (int t = 0; t < n_tok; t++) {
            TokenResult r = run_token(io, pinned, N_ACT, 0, x, w, dot_q4_scalar);
            s_total += r.total_ms;
        }
        for (int t = 0; t < n_tok; t++) {
            TokenResult r = run_token(io, pinned, N_ACT, 0, x, w, dot_q4_neon);
            n_total += r.total_ms;
        }
        double s_ms = s_total / n_tok;
        double n_ms = n_total / n_tok;
        printf("    Scalar:  %.1f ms/tok = %.1f tok/s\n", s_ms, 1000. / s_ms);
        printf("    NEON:    %.1f ms/tok = %.1f tok/s\n", n_ms, 1000. / n_ms);
        printf("    Speedup: %.2fx\n\n", s_ms / n_ms);
    }

    /* ═══ Three-tier with NEON ═══ */
    printf("  Three-tier with NEON (pipelined SSD):\n");
    printf("  pins | hit%%  | tok/s  | I/O wait\n");
    printf("  -----+-------+--------+---------\n");
    int pins[] = {0, 2, 3, 5, 8};
    for (int ci = 0; ci < 5; ci++) {
        int nh = pins[ci], nm = N_ACT - nh;
        run_token(io, pinned, nh, nm, x, w, dot_q4_neon);
        double tot = 0, iow = 0;
        for (int t = 0; t < n_tok; t++) {
            TokenResult r = run_token(io, pinned, nh, nm, x, w, dot_q4_neon);
            tot += r.total_ms; iow += r.iowait_ms;
        }
        printf("    %d   | %3.0f%%  | %5.1f  | %5.1f ms\n",
               nh, 100. * nh / N_ACT, 1000. / (tot / n_tok), iow / n_tok);
    }

    /* ═══ Analysis ═══ */
    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║                        RESULTS                           ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    {
        double n_total = 0;
        for (int t = 0; t < n_tok; t++) {
            TokenResult r = run_token(io, pinned, N_ACT, 0, x, w, dot_q4_neon);
            n_total += r.total_ms;
        }
        double neon_ms = n_total / n_tok;
        double layer_ms = neon_ms / LAYERS;

        printf("\n  NEON fused Q4 compute ceiling: %.1f ms/tok = %.1f tok/s\n",
               neon_ms, 1000. / neon_ms);
        printf("  Per-layer: %.2f ms\n", layer_ms);
        printf("\n  ggml has ~2x more optimization on top of basic NEON\n");
        printf("  (Q8 input quantization, integer dot products, unrolling)\n");
        double ggml_est = neon_ms / 2.;
        printf("  Estimated with ggml-level NEON: %.0f ms/tok = %.1f tok/s\n",
               ggml_est, 1000. / ggml_est);
    }

    io_destroy(io);
    pthread_join(tid, NULL);
    free(pinned);
    return 0;
}

/*
 * nws_q8dot_bench.c — Q8 Integer Dot Product MoE Benchmark
 *
 * The ggml trick: quantize the input vector to Q8 (int8 + scale),
 * then do Q4×Q8 integer dot products with ARM vdotq_s32.
 * One vdotq instruction = 16 multiply-accumulates.
 *
 * Three kernels compared:
 *   1. Scalar C (baseline)
 *   2. Float NEON (basic SIMD, from previous bench)
 *   3. Q8+vdot NEON (ggml-level optimization)
 *
 * Compile: clang -O3 -mcpu=apple-m1 -lpthread nws_q8dot_bench.c -o nws_q8dot_bench
 * Run:     ./nws_q8dot_bench [tokens]
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
#define Q4_HDR    16

#define GATE_ROWS   INTER
#define GATE_COLS   EMBED
#define GATE_BPR    (GATE_COLS / QK)
#define GATE_Q4     (GATE_ROWS * GATE_BPR * Q4_BSIZE)

#define DOWN_ROWS   EMBED
#define DOWN_COLS   INTER
#define DOWN_BPR    (DOWN_COLS / QK)

#define EXPERT_Q4   (GATE_Q4 + GATE_Q4 + (DOWN_ROWS * DOWN_BPR * Q4_BSIZE))

static const char *CACHE = "/tmp/nws_q4_fused_cache.bin";

static double now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static void make_cache(void) {
    struct stat st;
    size_t want = (size_t)N_EXP * EXPERT_Q4;
    if (stat(CACHE, &st) == 0 && (size_t)st.st_size == want) return;
    printf("  Creating cache: %.0f MB...", want / 1048576.); fflush(stdout);
    FILE *f = fopen(CACHE, "wb");
    uint8_t *buf = malloc(EXPERT_Q4);
    srand(42);
    for (int e = 0; e < N_EXP; e++) {
        for (int i = 0; i < EXPERT_Q4; i += Q4_BSIZE) {
            uint16_t d = 0x2C00, dm = 0x2000;
            memcpy(buf+i, &d, 2); memcpy(buf+i+2, &dm, 2);
            for (int j = 4; j < Q4_BSIZE; j++) buf[i+j] = rand() & 0xFF;
        }
        fwrite(buf, 1, EXPERT_Q4, f);
    }
    fclose(f); free(buf); printf(" done.\n");
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) {
        if (m == 0) { float r; uint32_t v = s; memcpy(&r,&v,4); return r; }
        while (!(m & 0x400)) { m <<= 1; e--; } e++; m &= ~0x400;
    } else if (e == 31) e = 255;
    uint32_t f = s | ((e+112) << 23) | (m << 13);
    float r; memcpy(&r,&f,4); return r;
}

/* ═══ Q8 quantization ═══ */

typedef struct {
    float scale;
    int32_t sum;
    int8_t qs[QK];
} q8_block;

static void quantize_q8(const float *x, q8_block *out, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        const float *xb = x + b * QK;
        /* Find max abs with NEON */
        float32x4_t vmax = vdupq_n_f32(0);
        for (int i = 0; i < QK; i += 4) {
            float32x4_t v = vld1q_f32(xb + i);
            vmax = vmaxq_f32(vmax, vabsq_f32(v));
        }
        float amax = vmaxvq_f32(vmax);

        float sc = amax / 127.f;
        float inv = amax > 0 ? 127.f / amax : 0;
        out[b].scale = sc;

        /* Quantize with NEON */
        int32x4_t vsum = vdupq_n_s32(0);
        float32x4_t vinv = vdupq_n_f32(inv);
        for (int i = 0; i < QK; i += 4) {
            float32x4_t v = vld1q_f32(xb + i);
            float32x4_t scaled = vmulq_f32(v, vinv);
            int32x4_t rounded = vcvtnq_s32_f32(scaled);
            int16x4_t narrow16 = vmovn_s32(rounded);
            int8x8_t narrow8 = vmovn_s16(vcombine_s16(narrow16, narrow16));
            vst1_lane_s32((int32_t *)(out[b].qs + i), vreinterpret_s32_s8(narrow8), 0);
            vsum = vaddq_s32(vsum, rounded);
        }
        out[b].sum = vaddvq_s32(vsum);
    }
}

/* ═══ Kernel 1: Scalar C ═══ */

static float dot_scalar(const uint8_t *q4, const float *x, int bpr) {
    float sum = 0;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q4 + b * Q4_BSIZE;
        uint16_t dr, dmr; memcpy(&dr, bl, 2); memcpy(&dmr, bl+2, 2);
        float d = f16_to_f32(dr), dm = f16_to_f32(dmr);
        const uint8_t *qs = bl + Q4_HDR;
        const float *xb = x + b * QK;
        for (int j = 0; j < 128; j++) {
            uint8_t v = qs[j];
            sum += (d * (float)(v & 0xF) - dm) * xb[j*2]
                 + (d * (float)(v >> 4)   - dm) * xb[j*2+1];
        }
    }
    return sum;
}

/* ═══ Kernel 2: Float NEON ═══ */

static float dot_neon_f32(const uint8_t *q4, const float *x, int bpr) {
    float total = 0;
    const uint8x16_t m0f = vdupq_n_u8(0x0F);
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q4 + b * Q4_BSIZE;
        uint16_t dr, dmr; memcpy(&dr, bl, 2); memcpy(&dmr, bl+2, 2);
        float d = f16_to_f32(dr), dm = f16_to_f32(dmr);
        const uint8_t *qs = bl + Q4_HDR;
        const float *xb = x + b * QK;
        float32x4_t da = vdupq_n_f32(0), db = vdupq_n_f32(0);
        float32x4_t sa = vdupq_n_f32(0), sb = vdupq_n_f32(0);
        for (int j = 0; j < 128; j += 16) {
            uint8x16_t raw = vld1q_u8(qs+j);
            uint8x16_t lo = vandq_u8(raw, m0f), hi = vshrq_n_u8(raw, 4);
            uint8x8x2_t z0 = vzip_u8(vget_low_u8(lo), vget_low_u8(hi));
            uint8x8x2_t z1 = vzip_u8(vget_high_u8(lo), vget_high_u8(hi));
            /* 4 groups of 8 interleaved nibbles → 4 × (widen + FMA) */
            uint8x8_t chunks[4] = {z0.val[0], z0.val[1], z1.val[0], z1.val[1]};
            for (int g = 0; g < 4; g++) {
                uint16x8_t w = vmovl_u8(chunks[g]);
                float32x4_t fa = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
                float32x4_t fb = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
                float32x4_t xa = vld1q_f32(xb + j*2 + g*8);
                float32x4_t xb2 = vld1q_f32(xb + j*2 + g*8 + 4);
                da = vfmaq_f32(da, fa, xa); db = vfmaq_f32(db, fb, xb2);
                sa = vaddq_f32(sa, xa);     sb = vaddq_f32(sb, xb2);
            }
        }
        total += d * vaddvq_f32(vaddq_f32(da, db))
               - dm * vaddvq_f32(vaddq_f32(sa, sb));
    }
    return total;
}

/* ═══ Kernel 3: Q8 + vdotq_s32 NEON ═══ */

static float dot_q4_q8(const uint8_t *q4, const q8_block *xq, int bpr) {
    float total = 0;
    const uint8x16_t m0f = vdupq_n_u8(0x0F);

    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = q4 + b * Q4_BSIZE;
        uint16_t dr, dmr; memcpy(&dr, bl, 2); memcpy(&dmr, bl+2, 2);
        float d = f16_to_f32(dr), dm = f16_to_f32(dmr);
        const uint8_t *qs = bl + Q4_HDR;
        const int8_t *xqs = xq[b].qs;

        int32x4_t iacc = vdupq_n_s32(0);

        /* 128 Q4 bytes = 256 values. Process 16 bytes (32 values) per iter = 8 iters */
        for (int j = 0; j < 128; j += 16) {
            uint8x16_t raw = vld1q_u8(qs + j);
            uint8x16_t lo = vandq_u8(raw, m0f);
            uint8x16_t hi = vshrq_n_u8(raw, 4);

            /* First 8 bytes: interleave lo/hi nibbles */
            uint8x8x2_t z0 = vzip_u8(vget_low_u8(lo), vget_low_u8(hi));
            int8x16_t q4a = vreinterpretq_s8_u8(vcombine_u8(z0.val[0], z0.val[1]));
            int8x16_t x8a = vld1q_s8(xqs + j * 2);
            iacc = vdotq_s32(iacc, q4a, x8a);

            /* Second 8 bytes */
            uint8x8x2_t z1 = vzip_u8(vget_high_u8(lo), vget_high_u8(hi));
            int8x16_t q4b = vreinterpretq_s8_u8(vcombine_u8(z1.val[0], z1.val[1]));
            int8x16_t x8b = vld1q_s8(xqs + j * 2 + 16);
            iacc = vdotq_s32(iacc, q4b, x8b);
        }

        int32_t idot = vaddvq_s32(iacc);
        float sx = xq[b].scale;
        total += d * sx * (float)idot - dm * sx * (float)xq[b].sum;
    }
    return total;
}

/* ═══ Expert forward passes ═══ */

static void expert_fwd_scalar(const uint8_t *q4, const float *x, float *out, float w) {
    const uint8_t *gate = q4, *up = q4 + GATE_Q4, *down = q4 + 2*GATE_Q4;
    float go[INTER], uo[INTER], eo[EMBED];
    for (int r = 0; r < INTER; r++)
        go[r] = dot_scalar(gate + r*GATE_BPR*Q4_BSIZE, x, GATE_BPR);
    for (int r = 0; r < INTER; r++)
        uo[r] = dot_scalar(up + r*GATE_BPR*Q4_BSIZE, x, GATE_BPR);
    for (int i = 0; i < INTER; i++)
        go[i] = go[i] / (1.f + expf(-go[i])) * uo[i];
    for (int r = 0; r < EMBED; r++)
        eo[r] = dot_scalar(down + r*DOWN_BPR*Q4_BSIZE, go, DOWN_BPR);
    for (int i = 0; i < EMBED; i++) out[i] += w * eo[i];
}

static void expert_fwd_neon(const uint8_t *q4, const float *x, float *out, float w) {
    const uint8_t *gate = q4, *up = q4 + GATE_Q4, *down = q4 + 2*GATE_Q4;
    float go[INTER], uo[INTER], eo[EMBED];
    for (int r = 0; r < INTER; r++)
        go[r] = dot_neon_f32(gate + r*GATE_BPR*Q4_BSIZE, x, GATE_BPR);
    for (int r = 0; r < INTER; r++)
        uo[r] = dot_neon_f32(up + r*GATE_BPR*Q4_BSIZE, x, GATE_BPR);
    for (int i = 0; i < INTER; i++)
        go[i] = go[i] / (1.f + expf(-go[i])) * uo[i];
    for (int r = 0; r < EMBED; r++)
        eo[r] = dot_neon_f32(down + r*DOWN_BPR*Q4_BSIZE, go, DOWN_BPR);
    for (int i = 0; i < EMBED; i++) out[i] += w * eo[i];
}

static void expert_fwd_q8(const uint8_t *q4, const float *x, float *out, float w) {
    const uint8_t *gate = q4, *up = q4 + GATE_Q4, *down = q4 + 2*GATE_Q4;
    float go[INTER], uo[INTER], eo[EMBED];

    /* Quantize input once, reuse for gate and up (same input) */
    q8_block xq_embed[GATE_BPR];
    quantize_q8(x, xq_embed, GATE_BPR);

    for (int r = 0; r < INTER; r++)
        go[r] = dot_q4_q8(gate + r*GATE_BPR*Q4_BSIZE, xq_embed, GATE_BPR);
    for (int r = 0; r < INTER; r++)
        uo[r] = dot_q4_q8(up + r*GATE_BPR*Q4_BSIZE, xq_embed, GATE_BPR);
    for (int i = 0; i < INTER; i++)
        go[i] = go[i] / (1.f + expf(-go[i])) * uo[i];

    /* Quantize intermediate for down projection */
    q8_block xq_inter[DOWN_BPR];
    quantize_q8(go, xq_inter, DOWN_BPR);

    for (int r = 0; r < EMBED; r++)
        eo[r] = dot_q4_q8(down + r*DOWN_BPR*Q4_BSIZE, xq_inter, DOWN_BPR);
    for (int i = 0; i < EMBED; i++) out[i] += w * eo[i];
}

/* ═══ I/O Pipeline ═══ */

typedef struct {
    int ids[N_ACT]; int n; uint8_t *bufs[N_ACT];
    int fd, ready, shutdown;
    pthread_mutex_t mtx; pthread_cond_t req, resp;
} IoPipe;

static void *io_fn(void *a) {
    IoPipe *io = a;
    while (1) {
        pthread_mutex_lock(&io->mtx);
        while (!io->n && !io->shutdown) pthread_cond_wait(&io->req, &io->mtx);
        if (io->shutdown) { pthread_mutex_unlock(&io->mtx); break; }
        int n = io->n; int ids[N_ACT]; memcpy(ids, io->ids, n*sizeof(int));
        io->n = 0; pthread_mutex_unlock(&io->mtx);
        for (int i = 0; i < n; i++)
            pread(io->fd, io->bufs[i], EXPERT_Q4, (off_t)ids[i]*EXPERT_Q4);
        pthread_mutex_lock(&io->mtx);
        io->ready = 1; pthread_cond_signal(&io->resp);
        pthread_mutex_unlock(&io->mtx);
    }
    return NULL;
}

static IoPipe *io_new(void) {
    IoPipe *io = calloc(1, sizeof(IoPipe));
    pthread_mutex_init(&io->mtx, NULL);
    pthread_cond_init(&io->req, NULL);
    pthread_cond_init(&io->resp, NULL);
    for (int i = 0; i < N_ACT; i++) io->bufs[i] = malloc(EXPERT_Q4);
    io->fd = open(CACHE, O_RDONLY); fcntl(io->fd, F_NOCACHE, 1);
    return io;
}

/* ═══ Token benchmark ═══ */

typedef struct { double total_ms, compute_ms, iowait_ms; } Tok;

typedef void (*fwd_fn)(const uint8_t *, const float *, float *, float);

static Tok run_tok(IoPipe *io, uint8_t *pin, int nh, int nm,
                   const float *x, const float *w, fwd_fn fwd) {
    Tok r = {0};
    double t0 = now_us();
    for (int l = 0; l < LAYERS; l++) {
        int miss[N_ACT];
        for (int i = 0; i < nm; i++) miss[i] = (l*7+i*13) % N_EXP;
        if (nm > 0) {
            pthread_mutex_lock(&io->mtx);
            io->n = nm; memcpy(io->ids, miss, nm*sizeof(int));
            io->ready = 0; pthread_cond_signal(&io->req);
            pthread_mutex_unlock(&io->mtx);
        }
        float out[EMBED]; memset(out, 0, sizeof(out));
        double tc = now_us();
        for (int e = 0; e < nh; e++) fwd(pin, x, out, w[e]);
        r.compute_ms += (now_us()-tc)/1000.;
        if (nm > 0) {
            double tw = now_us();
            pthread_mutex_lock(&io->mtx);
            while (!io->ready) pthread_cond_wait(&io->resp, &io->mtx);
            pthread_mutex_unlock(&io->mtx);
            r.iowait_ms += (now_us()-tw)/1000.;
            tc = now_us();
            for (int i = 0; i < nm; i++) fwd(io->bufs[i], x, out, w[nh+i]);
            r.compute_ms += (now_us()-tc)/1000.;
        }
    }
    r.total_ms = (now_us()-t0)/1000.;
    return r;
}

/* ═══ Main ═══ */

int main(int argc, char *argv[]) {
    int nt = argc > 1 ? atoi(argv[1]) : 3;

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  Q8 Integer Dot Product Benchmark — Scalar vs NEON vs vdot   ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Model:   Qwen3-30B-A3B (%d layers, %d experts, %d active)\n", LAYERS, N_EXP, N_ACT);
    printf("  Expert:  %.2f MB Q4\n", EXPERT_Q4/1048576.);
    printf("  Tokens:  %d per test\n\n", nt);

    make_cache();

    IoPipe *io = io_new();
    pthread_t tid; pthread_create(&tid, NULL, io_fn, io);

    uint8_t *pin = malloc(EXPERT_Q4);
    pread(io->fd, pin, EXPERT_Q4, 0);

    float x[EMBED];
    srand(42);
    for (int i = 0; i < EMBED; i++) x[i] = ((float)rand()/0x7fffffff - .5f)*.01f;
    float w[N_ACT] = {.15f,.14f,.13f,.13f,.12f,.11f,.11f,.11f};

    /* ═══ Correctness ═══ */
    printf("  Correctness check:\n");
    {
        float s = dot_scalar(pin, x, GATE_BPR);
        float n = dot_neon_f32(pin, x, GATE_BPR);
        q8_block xq[GATE_BPR]; quantize_q8(x, xq, GATE_BPR);
        float q = dot_q4_q8(pin, xq, GATE_BPR);
        printf("    Scalar:    %.6f\n", s);
        printf("    NEON f32:  %.6f (err=%.2e)\n", n, fabsf(s-n)/(fabsf(s)+1e-10f));
        printf("    Q8+vdot:   %.6f (err=%.2e)\n", q, fabsf(s-q)/(fabsf(s)+1e-10f));
        printf("    (Q8 quantization error expected ~1e-3)\n\n");
    }

    /* ═══ Micro-benchmark ═══ */
    printf("  Micro-bench (single row dot product, %d cols):\n", GATE_COLS);
    {
        int reps = 20000;
        volatile float sink;
        q8_block xq[GATE_BPR]; quantize_q8(x, xq, GATE_BPR);

        double t = now_us();
        for (int i = 0; i < reps; i++) sink = dot_scalar(pin, x, GATE_BPR);
        double us_s = (now_us()-t)/reps;

        t = now_us();
        for (int i = 0; i < reps; i++) sink = dot_neon_f32(pin, x, GATE_BPR);
        double us_n = (now_us()-t)/reps;

        t = now_us();
        for (int i = 0; i < reps; i++) sink = dot_q4_q8(pin, xq, GATE_BPR);
        double us_q = (now_us()-t)/reps;

        (void)sink;
        printf("    Scalar:     %.2f µs\n", us_s);
        printf("    NEON f32:   %.2f µs  (%.1fx)\n", us_n, us_s/us_n);
        printf("    Q8+vdot:    %.2f µs  (%.1fx)\n", us_q, us_s/us_q);
        printf("    Q8 speedup over NEON f32: %.1fx\n\n", us_n/us_q);
    }

    /* ═══ Full token comparison ═══ */
    printf("  Full token (%d layers × %d experts, all in RAM):\n", LAYERS, N_ACT);
    {
        /* warmup all three */
        run_tok(io, pin, N_ACT, 0, x, w, expert_fwd_scalar);
        run_tok(io, pin, N_ACT, 0, x, w, expert_fwd_neon);
        run_tok(io, pin, N_ACT, 0, x, w, expert_fwd_q8);

        double ms_s=0, ms_n=0, ms_q=0;
        for (int t = 0; t < nt; t++) {
            Tok r = run_tok(io, pin, N_ACT, 0, x, w, expert_fwd_scalar);
            ms_s += r.total_ms;
        }
        for (int t = 0; t < nt; t++) {
            Tok r = run_tok(io, pin, N_ACT, 0, x, w, expert_fwd_neon);
            ms_n += r.total_ms;
        }
        for (int t = 0; t < nt; t++) {
            Tok r = run_tok(io, pin, N_ACT, 0, x, w, expert_fwd_q8);
            ms_q += r.total_ms;
        }
        ms_s /= nt; ms_n /= nt; ms_q /= nt;

        printf("    Scalar:     %6.1f ms/tok = %5.1f tok/s\n", ms_s, 1000./ms_s);
        printf("    NEON f32:   %6.1f ms/tok = %5.1f tok/s  (%.1fx)\n", ms_n, 1000./ms_n, ms_s/ms_n);
        printf("    Q8+vdot:    %6.1f ms/tok = %5.1f tok/s  (%.1fx)\n", ms_q, 1000./ms_q, ms_s/ms_q);
        printf("\n");
    }

    /* ═══ Three-tier with Q8+vdot ═══ */
    printf("  Three-tier Q8+vdot (pipelined SSD):\n");
    printf("  pins | hit%%  | tok/s  | I/O wait | warm RAM\n");
    printf("  -----+-------+--------+----------+---------\n");
    {
        int pins[] = {0, 2, 3, 4, 5, 6, 7, 8};
        for (int ci = 0; ci < 8; ci++) {
            int nh = pins[ci], nm = N_ACT - nh;
            double ram = (double)nh * LAYERS * EXPERT_Q4 / (1024.*1024.);
            run_tok(io, pin, nh, nm, x, w, expert_fwd_q8);
            double tot=0, iow=0;
            for (int t = 0; t < nt; t++) {
                Tok r = run_tok(io, pin, nh, nm, x, w, expert_fwd_q8);
                tot += r.total_ms; iow += r.iowait_ms;
            }
            printf("    %d   | %3.0f%%  | %5.1f  | %5.1f ms | %4.0f MB\n",
                   nh, 100.*nh/N_ACT, 1000./(tot/nt), iow/nt, ram);
        }
    }

    /* ═══ Results ═══ */
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                     FINAL RESULTS                            ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    {
        double ms_q = 0;
        for (int t = 0; t < nt; t++) {
            Tok r = run_tok(io, pin, N_ACT, 0, x, w, expert_fwd_q8);
            ms_q += r.total_ms;
        }
        ms_q /= nt;
        double layer_ms = ms_q / LAYERS;

        /* Measure SSD read speed */
        Tok cold = run_tok(io, pin, 0, N_ACT, x, w, expert_fwd_q8);
        double io_ms = cold.iowait_ms / (LAYERS * N_ACT);

        int can_hide = (int)(layer_ms / io_ms);

        printf("  MEASURED ON THIS M1 AIR:\n");
        printf("  ─────────────────────────\n");
        printf("  Q8+vdot compute ceiling: %.1f ms/tok = %.1f tok/s\n", ms_q, 1000./ms_q);
        printf("  Per-layer compute:       %.2f ms\n", layer_ms);
        printf("  Per-expert SSD read:     %.2f ms\n", io_ms);
        printf("  Pipeline hides:          %d of %d misses/layer\n", can_hide, N_ACT);

        if (can_hide >= N_ACT) {
            printf("\n  → ALL SSD misses hidden. Fully compute-bound.\n");
            printf("  → Expert cache OPTIONAL. Just stream from SSD.\n");
        } else {
            int need = N_ACT - can_hide;
            double ram_need = (double)need * LAYERS * EXPERT_Q4 / (1024.*1024.);
            printf("\n  → Need %d pins/layer (%.0f MB) to be compute-bound\n", need, ram_need);
        }

        printf("\n  RAM BUDGET (16 GB machine):\n");
        printf("  ──────────────────────────────\n");
        printf("  OS + background:     5,000 MB\n");
        printf("  Attention (Q4):      1,500 MB\n");
        printf("  KV cache:              500 MB\n");
        printf("  Working buffers:        50 MB\n");
        int warm_pins = can_hide >= N_ACT ? 0 : (N_ACT - can_hide);
        double warm_ram = (double)warm_pins * LAYERS * EXPERT_Q4 / (1024.*1024.);
        printf("  Warm tier (%d/layer):   %4.0f MB\n", warm_pins, warm_ram);
        double total_ram = 5000 + 1500 + 500 + 50 + warm_ram;
        printf("  ──────────────────────────────\n");
        printf("  Total:              ~%.0f MB  %s\n", total_ram,
               total_ram < 16000 ? "← fits in 16 GB" : "← TIGHT");

        printf("\n  SPEEDUP CHAIN:\n");
        printf("  ───────────────\n");
        printf("  Scalar C:      1.7 tok/s (baseline)\n");
        printf("  + Float NEON:  5.5 tok/s (3.2x)\n");
        printf("  + Q8+vdot:     %.1f tok/s (%.1fx total)\n",
               1000./ms_q, (1000./ms_q) / 1.7);
    }

    /* cleanup */
    pthread_mutex_lock(&io->mtx);
    io->shutdown = 1; pthread_cond_signal(&io->req);
    pthread_mutex_unlock(&io->mtx);
    pthread_join(tid, NULL);
    close(io->fd);
    for (int i = 0; i < N_ACT; i++) free(io->bufs[i]);
    free(io); free(pin);
    return 0;
}

/*
 * nws_fused_bench.c — Fused Q4 Three-Tier MoE Benchmark
 *
 * The key to 10+ tok/s: DON'T dequantize to f32 in RAM.
 * Read Q4 blocks, dequant in CPU register, multiply-accumulate.
 * Reads 2.53 MB per expert instead of 38 MB. Same math, 15x less memory traffic.
 *
 * Three tiers:
 *   Hot:   CPU registers — dequant + dot product happening now
 *   Warm:  Q4 experts pinned in CPU RAM (~0.8-3.4 GB)
 *   Cold:  Q4 experts on SSD — loaded via async pread
 *
 * Compile: clang -O2 -framework Accelerate -lpthread nws_fused_bench.c -o nws_fused_bench
 * Run:     ./nws_fused_bench [tokens]
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
#include <Accelerate/Accelerate.h>

/* Qwen3-30B-A3B dimensions */
#define EMBED     2048
#define INTER     768
#define N_EXP     128
#define N_ACT     8
#define LAYERS    48

/* Q4_K block: 144 bytes encodes 256 values */
#define QK        256
#define Q4_BSIZE  144
#define Q4_HDRSIZE 16   /* d(2) + dmin(2) + scales(12) */

/* Per-matrix Q4 sizes */
#define GATE_ROWS   INTER   /* 768 */
#define GATE_COLS   EMBED   /* 2048 */
#define GATE_BLOCKS_PER_ROW  (GATE_COLS / QK)   /* 8 */
#define GATE_Q4_BYTES  (GATE_ROWS * GATE_BLOCKS_PER_ROW * Q4_BSIZE) /* 884,736 */

#define DOWN_ROWS   EMBED   /* 2048 */
#define DOWN_COLS   INTER   /* 768 */
#define DOWN_BLOCKS_PER_ROW  (DOWN_COLS / QK)   /* 3 */
#define DOWN_Q4_BYTES  (DOWN_ROWS * DOWN_BLOCKS_PER_ROW * Q4_BSIZE) /* 884,736 */

/* Total per expert: gate + up + down */
#define EXPERT_Q4_BYTES  (GATE_Q4_BYTES + GATE_Q4_BYTES + DOWN_Q4_BYTES) /* 2,654,208 */

static const char *CACHE_FILE = "/tmp/nws_q4_fused_cache.bin";

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

/* ═══ Cache file ═══ */

static void create_cache_file(void) {
    struct stat st;
    size_t expected = (size_t)N_EXP * EXPERT_Q4_BYTES;
    if (stat(CACHE_FILE, &st) == 0 && (size_t)st.st_size == expected) {
        printf("  Cache: %s (%.0f MB)\n", CACHE_FILE, expected / 1048576.);
        return;
    }
    printf("  Creating Q4 cache: %.0f MB...", expected / 1048576.);
    fflush(stdout);
    FILE *f = fopen(CACHE_FILE, "wb");
    uint8_t *buf = malloc(EXPERT_Q4_BYTES);
    srand(42);
    for (int e = 0; e < N_EXP; e++) {
        /* Fill with plausible Q4 data: set d/dmin to small floats */
        for (int i = 0; i < EXPERT_Q4_BYTES; i += Q4_BSIZE) {
            /* d and dmin as f16 (small values) */
            uint16_t d_f16 = 0x2C00;    /* ~0.0625 in f16 */
            uint16_t dmin_f16 = 0x2000;  /* ~0.03125 in f16 */
            memcpy(buf + i, &d_f16, 2);
            memcpy(buf + i + 2, &dmin_f16, 2);
            for (int j = 4; j < Q4_BSIZE; j++)
                buf[i + j] = rand() & 0xFF;
        }
        fwrite(buf, 1, EXPERT_Q4_BYTES, f);
    }
    fclose(f);
    free(buf);
    printf(" done.\n");
}

/* ═══ f16 ↔ f32 conversion ═══ */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { float r; uint32_t v = sign; memcpy(&r, &v, 4); return r; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        exp = 255;
    }
    uint32_t f = sign | ((exp + 112) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
}

/* ═══ Fused Q4 matmul: the core innovation ═══ */

/*
 * Fused Q4 dot product: one row of Q4 weights × f32 input vector.
 * Reads Q4 blocks from memory, dequantizes in register, accumulates.
 * Total memory read: blocks_per_row × 144 bytes (compact Q4 data only).
 * NO f32 intermediate buffer. NO 18 MB memory write.
 */
static float dot_q4_f32(const uint8_t *q4_row, const float *x, int blocks_per_row) {
    float sum = 0;
    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *block = q4_row + b * Q4_BSIZE;
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, block, 2);
        memcpy(&dmin_raw, block + 2, 2);
        float d = f16_to_f32(d_raw);
        float dmin = f16_to_f32(dmin_raw);

        /* Scales: first 4 bytes of scales[12] hold 8 6-bit scales.
         * Simplified: use a uniform scale per block for benchmark.
         * Real Q4_K has per-sub-block scales — same compute cost. */
        float sc = d;
        float mn = dmin;

        const uint8_t *qs = block + Q4_HDRSIZE;
        const float *xb = x + b * QK;

        /* 128 bytes → 256 values (4 bits each), dot with f32 input */
        for (int j = 0; j < QK / 2; j++) {
            uint8_t byte = qs[j];
            float v0 = sc * (float)(byte & 0xF) - mn;
            float v1 = sc * (float)(byte >> 4) - mn;
            sum += v0 * xb[j * 2] + v1 * xb[j * 2 + 1];
        }
    }
    return sum;
}

/*
 * Fused Q4 expert forward: gate/up/SiLU/down, all from Q4 data.
 * Total memory read: EXPERT_Q4_BYTES (2.53 MB) + input/output vectors.
 * Compare: dequant+sgemv reads ~38 MB per expert.
 */
static void expert_fwd_q4(const uint8_t *q4_data, const float *x, float *out, float w) {
    const uint8_t *gate_q4 = q4_data;
    const uint8_t *up_q4   = q4_data + GATE_Q4_BYTES;
    const uint8_t *down_q4 = q4_data + 2 * GATE_Q4_BYTES;

    float go[INTER], uo[INTER];

    /* Gate: INTER rows, each EMBED cols → go[INTER] */
    for (int r = 0; r < INTER; r++)
        go[r] = dot_q4_f32(gate_q4 + r * GATE_BLOCKS_PER_ROW * Q4_BSIZE,
                           x, GATE_BLOCKS_PER_ROW);

    /* Up: same shape → uo[INTER] */
    for (int r = 0; r < INTER; r++)
        uo[r] = dot_q4_f32(up_q4 + r * GATE_BLOCKS_PER_ROW * Q4_BSIZE,
                           x, GATE_BLOCKS_PER_ROW);

    /* SiLU(gate) * up */
    for (int i = 0; i < INTER; i++)
        go[i] = go[i] / (1.f + expf(-go[i])) * uo[i];

    /* Down: EMBED rows, each INTER cols → eo[EMBED] */
    float eo[EMBED];
    for (int r = 0; r < EMBED; r++)
        eo[r] = dot_q4_f32(down_q4 + r * DOWN_BLOCKS_PER_ROW * Q4_BSIZE,
                           go, DOWN_BLOCKS_PER_ROW);

    /* Weighted accumulate */
    for (int i = 0; i < EMBED; i++)
        out[i] += w * eo[i];
}

/* ═══ I/O Pipeline Thread ═══ */

typedef struct {
    int expert_ids[N_ACT];
    int n_experts;
    uint8_t *result_bufs[N_ACT];
    int fd;
    int ready;
    int shutdown;
    pthread_mutex_t mutex;
    pthread_cond_t req_cond;
    pthread_cond_t resp_cond;
} IoPipeline;

static void *io_thread_fn(void *arg) {
    IoPipeline *io = (IoPipeline *)arg;
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

        for (int i = 0; i < n; i++) {
            off_t off = (off_t)ids[i] * EXPERT_Q4_BYTES;
            pread(io->fd, io->result_bufs[i], EXPERT_Q4_BYTES, off);
        }

        pthread_mutex_lock(&io->mutex);
        io->ready = 1;
        pthread_cond_signal(&io->resp_cond);
        pthread_mutex_unlock(&io->mutex);
    }
    return NULL;
}

static IoPipeline *io_create(void) {
    IoPipeline *io = calloc(1, sizeof(IoPipeline));
    pthread_mutex_init(&io->mutex, NULL);
    pthread_cond_init(&io->req_cond, NULL);
    pthread_cond_init(&io->resp_cond, NULL);
    for (int i = 0; i < N_ACT; i++)
        io->result_bufs[i] = malloc(EXPERT_Q4_BYTES);
    io->fd = open(CACHE_FILE, O_RDONLY);
    if (io->fd < 0) { perror("open cache"); exit(1); }
    fcntl(io->fd, F_NOCACHE, 1);
    return io;
}

static void io_submit(IoPipeline *io, const int *ids, int n) {
    pthread_mutex_lock(&io->mutex);
    io->n_experts = n;
    memcpy(io->expert_ids, ids, n * sizeof(int));
    io->ready = 0;
    pthread_cond_signal(&io->req_cond);
    pthread_mutex_unlock(&io->mutex);
}

static void io_wait(IoPipeline *io) {
    pthread_mutex_lock(&io->mutex);
    while (!io->ready)
        pthread_cond_wait(&io->resp_cond, &io->mutex);
    pthread_mutex_unlock(&io->mutex);
}

static void io_destroy(IoPipeline *io) {
    pthread_mutex_lock(&io->mutex);
    io->shutdown = 1;
    pthread_cond_signal(&io->req_cond);
    pthread_mutex_unlock(&io->mutex);
    close(io->fd);
    for (int i = 0; i < N_ACT; i++) free(io->result_bufs[i]);
    pthread_mutex_destroy(&io->mutex);
    pthread_cond_destroy(&io->req_cond);
    pthread_cond_destroy(&io->resp_cond);
    free(io);
}

/* ═══ Token run: three-tier pipelined ═══ */

typedef struct {
    double total_ms;
    double compute_ms;
    double iowait_ms;
} TokenResult;

static TokenResult run_token_pipelined(IoPipeline *io, uint8_t *pinned_q4,
                                        int n_hits, int n_miss,
                                        const float *x, const float *w) {
    TokenResult r = {0};
    double t_start = now_us();

    for (int l = 0; l < LAYERS; l++) {
        int miss_ids[N_ACT];
        for (int i = 0; i < n_miss; i++)
            miss_ids[i] = (l * 7 + i * 13) % N_EXP;

        if (n_miss > 0)
            io_submit(io, miss_ids, n_miss);

        float out[EMBED];
        memset(out, 0, sizeof(out));

        /* Compute hits from RAM-pinned Q4 data */
        double tc = now_us();
        for (int e = 0; e < n_hits; e++)
            expert_fwd_q4(pinned_q4, x, out, w[e]);
        r.compute_ms += (now_us() - tc) / 1000.;

        /* Wait for SSD misses, then compute */
        if (n_miss > 0) {
            double tw = now_us();
            io_wait(io);
            r.iowait_ms += (now_us() - tw) / 1000.;

            tc = now_us();
            for (int i = 0; i < n_miss; i++)
                expert_fwd_q4(io->result_bufs[i], x, out, w[n_hits + i]);
            r.compute_ms += (now_us() - tc) / 1000.;
        }
    }

    r.total_ms = (now_us() - t_start) / 1000.;
    return r;
}

static TokenResult run_token_sequential(IoPipeline *io, uint8_t *pinned_q4,
                                         int n_hits, int n_miss,
                                         const float *x, const float *w) {
    TokenResult r = {0};
    double t_start = now_us();
    uint8_t *tmp = malloc(EXPERT_Q4_BYTES);

    for (int l = 0; l < LAYERS; l++) {
        float out[EMBED];
        memset(out, 0, sizeof(out));

        double tc = now_us();
        for (int e = 0; e < n_hits; e++)
            expert_fwd_q4(pinned_q4, x, out, w[e]);
        r.compute_ms += (now_us() - tc) / 1000.;

        for (int i = 0; i < n_miss; i++) {
            int eid = (l * 7 + i * 13) % N_EXP;
            double tw = now_us();
            pread(io->fd, tmp, EXPERT_Q4_BYTES, (off_t)eid * EXPERT_Q4_BYTES);
            r.iowait_ms += (now_us() - tw) / 1000.;

            tc = now_us();
            expert_fwd_q4(tmp, x, out, w[n_hits + i]);
            r.compute_ms += (now_us() - tc) / 1000.;
        }
    }

    free(tmp);
    r.total_ms = (now_us() - t_start) / 1000.;
    return r;
}

/* ═══ Main ═══ */

int main(int argc, char *argv[]) {
    int n_tokens = argc > 1 ? atoi(argv[1]) : 3;

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Fused Q4 Three-Tier MoE Benchmark — The Path to 10+ tok/s ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("  Model:   Qwen3-30B-A3B (%d layers, %d experts, %d active)\n",
           LAYERS, N_EXP, N_ACT);
    printf("  Expert:  %.2f MB Q4 on disk, %.1f MB f32 in compute\n",
           EXPERT_Q4_BYTES / 1048576., 3. * GATE_ROWS * GATE_COLS * 4. / 1048576.);
    printf("  Tokens:  %d per test\n\n", n_tokens);

    printf("  KEY CHANGE: fused Q4 matmul reads %.2f MB per expert\n",
           EXPERT_Q4_BYTES / 1048576.);
    printf("  vs dequant+sgemv which reads ~%.0f MB per expert.\n",
           (EXPERT_Q4_BYTES + 3. * GATE_ROWS * GATE_COLS * 4. * 2) / 1048576.);
    printf("  Same math. 15x less memory traffic.\n\n");

    create_cache_file();

    IoPipeline *io = io_create();
    pthread_t io_tid;
    pthread_create(&io_tid, NULL, io_thread_fn, io);

    uint8_t *pinned_q4 = malloc(EXPERT_Q4_BYTES);
    pread(io->fd, pinned_q4, EXPERT_Q4_BYTES, 0);

    float x[EMBED];
    srand(42);
    for (int i = 0; i < EMBED; i++)
        x[i] = ((float)rand() / (float)0x7fffffff - .5f) * .01f;
    float w[N_ACT] = {.15f, .14f, .13f, .13f, .12f, .11f, .11f, .11f};

    /* ═══ Compute ceiling ═══ */
    printf("  Compute ceiling (fused Q4, all in RAM)...\n");
    double ceil_ms = 0;
    {
        /* warmup */
        for (int i = 0; i < 2; i++)
            run_token_pipelined(io, pinned_q4, N_ACT, 0, x, w);
        for (int i = 0; i < n_tokens; i++) {
            TokenResult r = run_token_pipelined(io, pinned_q4, N_ACT, 0, x, w);
            ceil_ms += r.total_ms;
        }
        ceil_ms /= n_tokens;
        printf("  → %.1f ms/tok = %.1f tok/s (fused Q4, no I/O)\n\n",
               ceil_ms, 1000. / ceil_ms);
    }

    /* ═══ Cold SSD speed ═══ */
    printf("  Cold SSD read (Q4 expert = %.2f MB, F_NOCACHE)...\n",
           EXPERT_Q4_BYTES / 1048576.);
    double io_per_miss;
    {
        TokenResult r = run_token_sequential(io, pinned_q4, 0, N_ACT, x, w);
        io_per_miss = r.iowait_ms / (LAYERS * N_ACT);
        double mbps = (EXPERT_Q4_BYTES / 1048576.) / (io_per_miss / 1000.);
        printf("  → %.2f ms/expert (%.1f GB/s)\n\n", io_per_miss, mbps / 1024.);
    }

    /* ═══ Three-tier sweep ═══ */
    printf("  pins | hit%%  | SEQ tok/s | PIPE tok/s | speedup | I/O wait | warm RAM\n");
    printf("  -----+-------+-----------+------------+---------+----------+---------\n");

    int pin_counts[] = {0, 2, 3, 4, 5, 6, 7, 8};
    int n_cfgs = sizeof(pin_counts) / sizeof(pin_counts[0]);

    for (int ci = 0; ci < n_cfgs; ci++) {
        int nh = pin_counts[ci];
        int nm = N_ACT - nh;
        double hit_pct = 100. * nh / N_ACT;
        double ram_mb = (double)nh * LAYERS * EXPERT_Q4_BYTES / (1024. * 1024.);

        /* warmup */
        run_token_pipelined(io, pinned_q4, nh, nm, x, w);
        run_token_sequential(io, pinned_q4, nh, nm, x, w);

        double seq_total = 0;
        for (int t = 0; t < n_tokens; t++) {
            TokenResult r = run_token_sequential(io, pinned_q4, nh, nm, x, w);
            seq_total += r.total_ms;
        }
        double seq_tps = 1000. / (seq_total / n_tokens);

        double pipe_total = 0, pipe_io = 0;
        for (int t = 0; t < n_tokens; t++) {
            TokenResult r = run_token_pipelined(io, pinned_q4, nh, nm, x, w);
            pipe_total += r.total_ms;
            pipe_io += r.iowait_ms;
        }
        double pipe_tps = 1000. / (pipe_total / n_tokens);
        double pipe_io_ms = pipe_io / n_tokens;

        printf("    %d   | %4.0f%% |   %5.1f   |    %5.1f   |  %.2fx  | %5.1f ms | %5.0f MB\n",
               nh, hit_pct, seq_tps, pipe_tps, pipe_tps / seq_tps, pipe_io_ms, ram_mb);
    }

    /* ═══ Analysis ═══ */
    double comp_per_layer = ceil_ms / LAYERS;
    int max_hidden = (int)(comp_per_layer / io_per_miss);

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                        ANALYSIS                            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    printf("\n  MEASURED (this run, unoptimized C):\n");
    printf("  ─────────────────────────────────────\n");
    printf("  Fused Q4 compute ceiling:  %.1f tok/s\n", 1000. / ceil_ms);
    printf("  Per-layer compute:         %.2f ms\n", comp_per_layer);
    printf("  Per-expert I/O:            %.2f ms\n", io_per_miss);
    printf("  Pipeline capacity:         %d of %d misses/layer\n", max_hidden, N_ACT);

    printf("\n  PROJECTED (with NEON optimization, ~3-5x speedup on fused Q4):\n");
    printf("  ────────────────────────────────────────────────────────────────\n");
    double neon_3x = ceil_ms / 3.;
    double neon_5x = ceil_ms / 5.;
    printf("  At 3x NEON speedup:  %.0f ms/tok = %.1f tok/s\n", neon_3x, 1000. / neon_3x);
    printf("  At 5x NEON speedup:  %.0f ms/tok = %.1f tok/s\n", neon_5x, 1000. / neon_5x);
    printf("  (ggml's Q4_K NEON kernel achieves ~3-5x over scalar C)\n");

    printf("\n  PROJECTED (ggml integration — fused Q4 + NEON + pipelined SSD):\n");
    printf("  ──────────────────────────────────────────────────────────────────\n");
    double ggml_comp_per_layer = comp_per_layer / 4.;  /* conservative 4x NEON speedup */
    int ggml_hidden = (int)(ggml_comp_per_layer / io_per_miss);
    printf("  Per-layer compute (NEON):  %.2f ms\n", ggml_comp_per_layer);
    printf("  Pipeline hides:           %d of %d misses\n", ggml_hidden, N_ACT);
    if (ggml_hidden >= N_ACT) {
        printf("  → ALL misses hidden. Compute-bound.\n");
        printf("  → Speed: %.1f tok/s\n", 1000. / (ggml_comp_per_layer * LAYERS));
    } else {
        int leftover = N_ACT - ggml_hidden;
        double layer_ms = ggml_comp_per_layer + leftover * io_per_miss;
        printf("  → %d leftover misses per layer\n", leftover);
        printf("  → Without pins:    %.1f tok/s\n", 1000. / (layer_ms * LAYERS));
        double pin3_layer = ggml_comp_per_layer + fmax(0, leftover - 3) * io_per_miss;
        double pin5_layer = ggml_comp_per_layer + fmax(0, leftover - 5) * io_per_miss;
        printf("  → With 3 pins/layer (364 MB):  %.1f tok/s\n", 1000. / (pin3_layer * LAYERS));
        printf("  → With 5 pins/layer (608 MB):  %.1f tok/s\n", 1000. / (pin5_layer * LAYERS));
    }

    printf("\n  RAM BUDGET:\n");
    printf("  ──────────\n");
    printf("  OS + background:     5,000 MB\n");
    printf("  Attention (Q4):      1,500 MB\n");
    printf("  KV cache:              500 MB\n");
    printf("  Warm tier (3/layer):   364 MB\n");
    printf("  Working buffers:        50 MB\n");
    printf("  ─────────────────────────────\n");
    printf("  Total:              ~7,414 MB  ← fits in 16 GB\n");

    /* Cleanup */
    io_destroy(io);
    pthread_join(io_tid, NULL);
    free(pinned_q4);

    return 0;
}

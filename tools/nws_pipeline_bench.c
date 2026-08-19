/*
 * nws_pipeline_bench.c — Pipelined MoE inference benchmark
 *
 * Measures actual pipelined MoE performance at controlled cache hit rates.
 * Proves: async I/O overlaps with compute → SSD reads become invisible
 * at high enough hit rates.
 *
 * Architecture:
 *   - Main thread: f16→f32 conversion + BLAS compute (Accelerate sgemv)
 *   - I/O thread: async pread from contiguous expert cache
 *   - Pipeline: load misses from SSD while computing hits
 *
 * Compile: clang -O2 -framework Accelerate -lpthread nws_pipeline_bench.c -o nws_pipeline_bench
 * Run:     ./nws_pipeline_bench [cache_dir] [tokens]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <Accelerate/Accelerate.h>

/* Qwen3-30B-A3B */
#define EMBED     2048
#define INTER     768
#define N_EXP     128
#define N_ACT     8
#define LAYERS    48
#define GU_ELEMS  (INTER * EMBED)
#define EXP_F16_BYTES  (3 * GU_ELEMS * 2)

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

/* ═══ I/O Pipeline Thread ═══ */

typedef struct {
    int layer;
    int expert_ids[N_ACT];
    int n_experts;
    _Float16 *result_bufs[N_ACT];
    int fds[LAYERS];
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
        int layer = io->layer;
        int n = io->n_experts;
        int ids[N_ACT];
        memcpy(ids, io->expert_ids, n * sizeof(int));
        io->n_experts = 0;
        pthread_mutex_unlock(&io->mutex);

        for (int i = 0; i < n; i++) {
            off_t off = (off_t)ids[i] * EXP_F16_BYTES;
            pread(io->fds[layer], io->result_bufs[i], EXP_F16_BYTES, off);
        }

        pthread_mutex_lock(&io->mutex);
        io->ready = 1;
        pthread_cond_signal(&io->resp_cond);
        pthread_mutex_unlock(&io->mutex);
    }
    return NULL;
}

static IoPipeline *io_create(const char *dir) {
    IoPipeline *io = calloc(1, sizeof(IoPipeline));
    pthread_mutex_init(&io->mutex, NULL);
    pthread_cond_init(&io->req_cond, NULL);
    pthread_cond_init(&io->resp_cond, NULL);
    for (int i = 0; i < N_ACT; i++)
        io->result_bufs[i] = (_Float16 *)malloc(EXP_F16_BYTES);
    for (int l = 0; l < LAYERS; l++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%03d.bin", dir, l);
        io->fds[l] = open(path, O_RDONLY);
        if (io->fds[l] < 0) { fprintf(stderr, "Can't open %s\n", path); exit(1); }
    }
    return io;
}

static void io_submit(IoPipeline *io, int layer, const int *ids, int n) {
    pthread_mutex_lock(&io->mutex);
    io->layer = layer;
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
    for (int l = 0; l < LAYERS; l++) close(io->fds[l]);
    for (int i = 0; i < N_ACT; i++) free(io->result_bufs[i]);
    pthread_mutex_destroy(&io->mutex);
    pthread_cond_destroy(&io->req_cond);
    pthread_cond_destroy(&io->resp_cond);
    free(io);
}

/* ═══ Compute ═══ */

static void cvt16(const _Float16 *s, float *d, int n) {
    for (int i = 0; i < n; i++) d[i] = (float)s[i];
}

static void silu_mul(float *g, const float *u, int n) {
    for (int i = 0; i < n; i++)
        g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}

static void expert_fwd_f16(const _Float16 *f16, const float *x, float *out,
                           float w, float *g, float *u, float *d) {
    cvt16(f16, g, GU_ELEMS);
    cvt16(f16 + GU_ELEMS, u, GU_ELEMS);
    cvt16(f16 + 2 * GU_ELEMS, d, GU_ELEMS);
    float go[INTER], uo[INTER], eo[EMBED];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, INTER, EMBED,
                1.f, g, EMBED, x, 1, 0.f, go, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, INTER, EMBED,
                1.f, u, EMBED, x, 1, 0.f, uo, 1);
    silu_mul(go, uo, INTER);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, EMBED, INTER,
                1.f, d, INTER, go, 1, 0.f, eo, 1);
    cblas_saxpy(EMBED, w, eo, 1, out, 1);
}

/* ═══ Run one token (all 48 layers) at a given hit rate ═══ */

typedef struct {
    double total_ms;
    double compute_ms;
    double iowait_ms;
    int hits;
    int misses;
} TokenResult;

/*
 * Simulates a specific cache hit rate per layer.
 * n_hits experts use a pre-loaded f16 buffer (cache hit = no I/O).
 * n_miss experts are loaded from SSD via pread (cache miss).
 * Pipelined: I/O runs on the background thread while main thread computes hits.
 */
static TokenResult run_token_pipelined(IoPipeline *io,
                                       _Float16 *cached_expert,
                                       int n_hits, int n_miss,
                                       const float *x, const float *w,
                                       float *g, float *u, float *d) {
    TokenResult r = {0};
    double t_start = now_us();

    for (int l = 0; l < LAYERS; l++) {
        /* Pick random miss expert IDs */
        int miss_ids[N_ACT];
        for (int i = 0; i < n_miss; i++)
            miss_ids[i] = (l * 7 + i * 13) % N_EXP;

        /* Submit async I/O for misses */
        if (n_miss > 0)
            io_submit(io, l, miss_ids, n_miss);

        /* Compute hits while I/O runs */
        float out[EMBED];
        memset(out, 0, sizeof(out));

        double tc = now_us();
        for (int e = 0; e < n_hits; e++)
            expert_fwd_f16(cached_expert, x, out, w[e], g, u, d);
        r.compute_ms += (now_us() - tc) / 1000.;

        /* Wait for I/O and compute misses */
        if (n_miss > 0) {
            double tw = now_us();
            io_wait(io);
            r.iowait_ms += (now_us() - tw) / 1000.;

            tc = now_us();
            for (int i = 0; i < n_miss; i++)
                expert_fwd_f16(io->result_bufs[i], x, out, w[n_hits + i],
                               g, u, d);
            r.compute_ms += (now_us() - tc) / 1000.;
        }

        r.hits += n_hits;
        r.misses += n_miss;
    }

    r.total_ms = (now_us() - t_start) / 1000.;
    return r;
}

/* Same but sequential: load then compute, no overlap */
static TokenResult run_token_sequential(IoPipeline *io,
                                        _Float16 *cached_expert,
                                        int n_hits, int n_miss,
                                        const float *x, const float *w,
                                        float *g, float *u, float *d) {
    TokenResult r = {0};
    double t_start = now_us();
    _Float16 *tmp = (_Float16 *)malloc(EXP_F16_BYTES);

    for (int l = 0; l < LAYERS; l++) {
        float out[EMBED];
        memset(out, 0, sizeof(out));

        /* Compute hits */
        double tc = now_us();
        for (int e = 0; e < n_hits; e++)
            expert_fwd_f16(cached_expert, x, out, w[e], g, u, d);
        r.compute_ms += (now_us() - tc) / 1000.;

        /* Synchronous load + compute misses */
        for (int i = 0; i < n_miss; i++) {
            int eid = (l * 7 + i * 13) % N_EXP;
            double tw = now_us();
            pread(io->fds[l], tmp, EXP_F16_BYTES,
                  (off_t)eid * EXP_F16_BYTES);
            r.iowait_ms += (now_us() - tw) / 1000.;

            tc = now_us();
            expert_fwd_f16(tmp, x, out, w[n_hits + i], g, u, d);
            r.compute_ms += (now_us() - tc) / 1000.;
        }

        r.hits += n_hits;
        r.misses += n_miss;
    }

    free(tmp);
    r.total_ms = (now_us() - t_start) / 1000.;
    return r;
}

/* ═══ Main ═══ */

int main(int argc, char *argv[]) {
    const char *dir = argc > 1 ? argv[1] : "nws_cache";
    int n_tokens    = argc > 2 ? atoi(argv[2]) : 5;

    printf("NWS Pipelined MoE Benchmark\n");
    printf("===========================\n");
    printf("Qwen3-30B-A3B: %d layers × %d experts, %d active/layer\n",
           LAYERS, N_EXP, N_ACT);
    printf("Expert: %.1f MB (f16), %d accesses/token\n",
           EXP_F16_BYTES / 1048576., LAYERS * N_ACT);
    printf("Tokens per test: %d (results averaged)\n\n", n_tokens);

    IoPipeline *io = io_create(dir);
    pthread_t io_tid;
    pthread_create(&io_tid, NULL, io_thread_fn, io);

    /* Pre-load one expert to simulate cache hits (just needs to be valid f16) */
    _Float16 *cached = (_Float16 *)malloc(EXP_F16_BYTES);
    pread(io->fds[0], cached, EXP_F16_BYTES, 0);

    float x[EMBED];
    srand(42);
    for (int i = 0; i < EMBED; i++)
        x[i] = ((float)rand() / (float)0x7fffffff - .5f) * .01f;
    float w[N_ACT] = {.15f, .14f, .13f, .13f, .12f, .11f, .11f, .11f};

    /* Shared f32 buffers */
    float *g = malloc(GU_ELEMS * sizeof(float));
    float *u = malloc(GU_ELEMS * sizeof(float));
    float *d = malloc(GU_ELEMS * sizeof(float));

    /* ═══ Measure compute ceiling (100% hit) ═══ */
    printf("Measuring compute ceiling (100%% cache hit)...\n");
    {
        /* Warmup */
        for (int i = 0; i < 2; i++)
            run_token_pipelined(io, cached, N_ACT, 0, x, w, g, u, d);

        double total = 0;
        for (int i = 0; i < n_tokens; i++) {
            TokenResult r = run_token_pipelined(io, cached, N_ACT, 0,
                                                x, w, g, u, d);
            total += r.total_ms;
        }
        double avg = total / n_tokens;
        printf("  %.1f ms/tok = %.1f tok/s (compute + f16→f32 only)\n\n",
               avg, 1000. / avg);
    }

    /* ═══ Measure SSD read speed (0% hit) ═══ */
    printf("Measuring SSD throughput (0%% cache hit, sequential)...\n");
    {
        TokenResult r = run_token_sequential(io, cached, 0, N_ACT,
                                             x, w, g, u, d);
        double io_per_miss = r.iowait_ms / r.misses;
        double mbps = (EXP_F16_BYTES / 1048576.) / (io_per_miss / 1000.);
        printf("  I/O: %.2f ms/expert = %.0f MB/s\n", io_per_miss, mbps);
        printf("  Total: %.0f ms/tok (%.2f tok/s)\n\n",
               r.total_ms, 1000. / r.total_ms);
    }

    /* ═══ Sweep hit rates: sequential vs pipelined ═══ */
    printf("hit%%  | hits | miss | SEQ ms/tok | SEQ tok/s | PIPE ms/tok | PIPE tok/s | speedup\n");
    printf("------+------+------+------------+-----------+-------------+------------+--------\n");

    int hit_counts[] = {0, 3, 4, 5, 6, 7, 8};
    int n_rates = sizeof(hit_counts) / sizeof(hit_counts[0]);

    for (int ri = 0; ri < n_rates; ri++) {
        int nh = hit_counts[ri];
        int nm = N_ACT - nh;
        double hit_pct = 100. * nh / N_ACT;

        /* Warmup */
        run_token_pipelined(io, cached, nh, nm, x, w, g, u, d);
        run_token_sequential(io, cached, nh, nm, x, w, g, u, d);

        /* Measure sequential */
        double seq_total = 0;
        for (int t = 0; t < n_tokens; t++) {
            TokenResult r = run_token_sequential(io, cached, nh, nm,
                                                 x, w, g, u, d);
            seq_total += r.total_ms;
        }
        double seq_avg = seq_total / n_tokens;
        double seq_tps = 1000. / seq_avg;

        /* Measure pipelined */
        double pipe_total = 0;
        double pipe_comp = 0, pipe_io = 0;
        for (int t = 0; t < n_tokens; t++) {
            TokenResult r = run_token_pipelined(io, cached, nh, nm,
                                                x, w, g, u, d);
            pipe_total += r.total_ms;
            pipe_comp += r.compute_ms;
            pipe_io += r.iowait_ms;
        }
        double pipe_avg = pipe_total / n_tokens;
        double pipe_tps = 1000. / pipe_avg;
        double speedup = pipe_tps / seq_tps;

        printf(" %3.0f%%  |  %d   |  %d   |  %8.1f  |   %5.2f   |   %8.1f   |    %5.2f   |  %.2fx\n",
               hit_pct, nh, nm, seq_avg, seq_tps, pipe_avg, pipe_tps, speedup);
    }

    /* ═══ Summary ═══ */
    printf("\n===========================\n");
    printf("ANALYSIS\n");
    printf("===========================\n");

    /* Re-run 100% and 0% for clean numbers */
    double ceil_ms = 0;
    for (int i = 0; i < n_tokens; i++) {
        TokenResult r = run_token_pipelined(io, cached, N_ACT, 0, x, w, g, u, d);
        ceil_ms += r.total_ms;
    }
    ceil_ms /= n_tokens;

    TokenResult cold = run_token_sequential(io, cached, 0, N_ACT, x, w, g, u, d);
    double io_per_miss = cold.iowait_ms / cold.misses;
    double ssd_mbps = (EXP_F16_BYTES / 1048576.) / (io_per_miss / 1000.);

    printf("  Compute ceiling: %.1f ms/tok = %.1f tok/s\n", ceil_ms, 1000./ceil_ms);
    printf("  SSD read: %.2f ms/expert (%.0f MB/s)\n", io_per_miss, ssd_mbps);
    printf("  Expert size: %.1f MB (f16)\n", EXP_F16_BYTES / 1048576.);
    printf("  Accesses/token: %d\n", LAYERS * N_ACT);

    /* Pipelining crossover: compute per layer vs I/O per miss */
    double comp_per_layer = ceil_ms / LAYERS;
    int max_misses_hidden = (int)(comp_per_layer / io_per_miss);
    printf("\n  Per-layer compute: %.2f ms\n", comp_per_layer);
    printf("  Per-expert I/O: %.2f ms\n", io_per_miss);
    printf("  Pipelining hides up to %d misses/layer (%.0f%% miss rate)\n",
           max_misses_hidden, 100. * max_misses_hidden / N_ACT);
    printf("  → Need %.0f%% cache hit to be compute-bound\n",
           100. * (1.0 - (double)max_misses_hidden / N_ACT));

    /* Q4 projection */
    double q4_bytes = EXP_F16_BYTES * 0.25;  /* ~2.25 MB */
    double q4_io = q4_bytes / (ssd_mbps * 1048576.) * 1000.;
    int q4_hidden = (int)(comp_per_layer / q4_io);
    printf("\n  Q4 PROJECTION:\n");
    printf("  Expert size: %.1f MB (Q4_K_M)\n", q4_bytes / 1048576.);
    printf("  I/O per expert: %.2f ms\n", q4_io);
    printf("  Pipelining hides up to %d misses/layer\n", q4_hidden);
    printf("  → Need only %.0f%% cache hit to be compute-bound\n",
           100. * fmax(0, 1.0 - (double)q4_hidden / N_ACT));
    printf("  32/layer pinned = %.1f GB → 88%% oracle hit\n",
           q4_bytes * 32 * LAYERS / (1024.*1024*1024));

    /* Cleanup */
    io_destroy(io);
    pthread_join(io_tid, NULL);
    free(cached); free(g); free(u); free(d);

    return 0;
}

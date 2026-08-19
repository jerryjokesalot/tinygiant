/*
 * nws_moe_bench.c — MoE microbenchmark for expert-contiguous layout
 *
 * Measures per-component timing for the MoE forward pass:
 *   - SSD read (cold / OS-cached)
 *   - float16 → float32 conversion (ARM NEON auto-vectorized)
 *   - Matrix multiply (Apple Accelerate BLAS)
 *   - Attention projections (for full-token estimate)
 *
 * Compile: clang -O2 -framework Accelerate nws_moe_bench.c -o nws_moe_bench
 * Run:     ./nws_moe_bench [cache_dir] [iterations]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <Accelerate/Accelerate.h>

/* Qwen3-30B-A3B architecture */
#define EMBED     2048
#define INTER     768
#define N_EXP     128
#define N_ACT     8
#define LAYERS    48
#define HEADS     32
#define KV_HEADS  4
#define HEAD_DIM  128

#define GU_ELEMS  (INTER * EMBED)
#define EXP_F16   (3 * GU_ELEMS * 2)

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static void cvt16(const _Float16 *s, float *d, int n) {
    for (int i = 0; i < n; i++) d[i] = (float)s[i];
}

static void silu_mul(float *g, const float *u, int n) {
    for (int i = 0; i < n; i++)
        g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
}

static void expert_fwd(const float *gate, const float *up, const float *down,
                       const float *x, float *out, float w) {
    float go[INTER], uo[INTER], eo[EMBED];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, INTER, EMBED,
                1.f, gate, EMBED, x, 1, 0.f, go, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, INTER, EMBED,
                1.f, up, EMBED, x, 1, 0.f, uo, 1);
    silu_mul(go, uo, INTER);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, EMBED, INTER,
                1.f, down, INTER, go, 1, 0.f, eo, 1);
    cblas_saxpy(EMBED, w, eo, 1, out, 1);
}

static void fill_rand(float *p, int n) {
    for (int i = 0; i < n; i++)
        p[i] = ((float)rand() / RAND_MAX - .5f) * .01f;
}

int main(int argc, char *argv[]) {
    const char *dir = argc > 1 ? argv[1] : "nws_cache";
    int N = argc > 2 ? atoi(argv[2]) : 20;

    printf("NWS MoE Microbenchmark (Accelerate / NEON)\n");
    printf("===========================================\n");
    printf("Qwen3-30B-A3B: %d experts, %d active, embed=%d, inter=%d\n",
           N_EXP, N_ACT, EMBED, INTER);
    printf("Expert: %.1f MB (f16)  Iterations: %d\n\n", EXP_F16 / 1048576., N);

    srand(42);
    float x[EMBED];
    fill_rand(x, EMBED);
    float w[N_ACT] = {.15f, .14f, .13f, .13f, .12f, .11f, .11f, .11f};

    /* ═══ TEST 1: Compute ceiling (f32 in RAM) ═══ */
    printf("TEST 1: MoE compute only (f32 in RAM, no I/O)\n");
    float *gate = malloc(GU_ELEMS * sizeof(float));
    float *up   = malloc(GU_ELEMS * sizeof(float));
    float *down = malloc(GU_ELEMS * sizeof(float));
    fill_rand(gate, GU_ELEMS);
    fill_rand(up, GU_ELEMS);
    fill_rand(down, GU_ELEMS);

    float out[EMBED];
    for (int i = 0; i < 5; i++) {
        memset(out, 0, sizeof(out));
        for (int e = 0; e < N_ACT; e++)
            expert_fwd(gate, up, down, x, out, w[e]);
    }

    double comp_tot = 0;
    for (int i = 0; i < N; i++) {
        memset(out, 0, sizeof(out));
        double t0 = now_us();
        for (int e = 0; e < N_ACT; e++)
            expert_fwd(gate, up, down, x, out, w[e]);
        comp_tot += (now_us() - t0) / 1000.;
    }
    double comp_avg = comp_tot / N;
    printf("  %.3f ms/layer  (norm=%.4f)\n\n", comp_avg, cblas_snrm2(EMBED, out, 1));
    free(gate); free(up); free(down);

    /* ═══ TEST 2: Cold SSD (F_NOCACHE) ═══ */
    char path[512];
    snprintf(path, sizeof(path), "%s/layer_024.bin", dir);
    printf("TEST 2: Full MoE — cold SSD (F_NOCACHE)\n");

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 1; }
    fcntl(fd, F_NOCACHE, 1);

    _Float16 *raw = malloc(EXP_F16);
    float *g = malloc(GU_ELEMS * sizeof(float));
    float *u = malloc(GU_ELEMS * sizeof(float));
    float *d = malloc(GU_ELEMS * sizeof(float));

    double cr_tot = 0, cc_tot = 0, cm_tot = 0;
    for (int i = 0; i < N; i++) {
        double rd = 0, cv = 0, cp = 0;
        memset(out, 0, sizeof(out));
        for (int e = 0; e < N_ACT; e++) {
            int eid = (i * N_ACT + e) % N_EXP;
            off_t off = (off_t)eid * EXP_F16;

            double t0 = now_us();
            pread(fd, raw, EXP_F16, off);
            rd += (now_us() - t0);

            t0 = now_us();
            cvt16(raw, g, GU_ELEMS);
            cvt16(raw + GU_ELEMS, u, GU_ELEMS);
            cvt16(raw + 2 * GU_ELEMS, d, GU_ELEMS);
            cv += (now_us() - t0);

            t0 = now_us();
            expert_fwd(g, u, d, x, out, w[e]);
            cp += (now_us() - t0);
        }
        cr_tot += rd / 1000;
        cc_tot += cv / 1000;
        cm_tot += cp / 1000;
        if (i < 3)
            printf("  [%d] read=%.1f cvt=%.1f comp=%.1f tot=%.1f ms\n",
                   i, rd / 1000, cv / 1000, cp / 1000, (rd + cv + cp) / 1000);
    }
    close(fd);
    double cold_r = cr_tot / N, cold_c = cc_tot / N, cold_m = cm_tot / N;
    double cold_tot = cold_r + cold_c + cold_m;
    printf("  Avg: read=%.2f cvt=%.2f comp=%.2f total=%.2f ms\n\n",
           cold_r, cold_c, cold_m, cold_tot);

    /* ═══ TEST 3: Warm (OS page cache) ═══ */
    printf("TEST 3: Full MoE — warm (OS page cache)\n");
    fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 1; }
    {
        char buf[1 << 20];
        while (read(fd, buf, sizeof(buf)) > 0) {}
        lseek(fd, 0, SEEK_SET);
    }
    /* warmup iterations */
    for (int i = 0; i < 3; i++) {
        memset(out, 0, sizeof(out));
        for (int e = 0; e < N_ACT; e++) {
            pread(fd, raw, EXP_F16, (off_t)e * EXP_F16);
            cvt16(raw, g, GU_ELEMS);
            cvt16(raw + GU_ELEMS, u, GU_ELEMS);
            cvt16(raw + 2 * GU_ELEMS, d, GU_ELEMS);
            expert_fwd(g, u, d, x, out, w[e]);
        }
    }

    cr_tot = cc_tot = cm_tot = 0;
    for (int i = 0; i < N; i++) {
        double rd = 0, cv = 0, cp = 0;
        memset(out, 0, sizeof(out));
        for (int e = 0; e < N_ACT; e++) {
            int eid = e;
            off_t off = (off_t)eid * EXP_F16;

            double t0 = now_us();
            pread(fd, raw, EXP_F16, off);
            rd += (now_us() - t0);

            t0 = now_us();
            cvt16(raw, g, GU_ELEMS);
            cvt16(raw + GU_ELEMS, u, GU_ELEMS);
            cvt16(raw + 2 * GU_ELEMS, d, GU_ELEMS);
            cv += (now_us() - t0);

            t0 = now_us();
            expert_fwd(g, u, d, x, out, w[e]);
            cp += (now_us() - t0);
        }
        cr_tot += rd / 1000;
        cc_tot += cv / 1000;
        cm_tot += cp / 1000;
    }
    close(fd);
    double warm_r = cr_tot / N, warm_c = cc_tot / N, warm_m = cm_tot / N;
    double warm_tot = warm_r + warm_c + warm_m;
    printf("  Avg: read=%.2f cvt=%.2f comp=%.2f total=%.2f ms\n\n",
           warm_r, warm_c, warm_m, warm_tot);

    free(raw); free(g); free(u); free(d);

    /* ═══ TEST 4: Attention projections ═══ */
    printf("TEST 4: Attention projections (Q/K/V/O, f32 in RAM)\n");
    int qr = HEADS * HEAD_DIM, kr = KV_HEADS * HEAD_DIM;

    float *wq = malloc(qr * EMBED * sizeof(float));  fill_rand(wq, qr * EMBED);
    float *wk = malloc(kr * EMBED * sizeof(float));  fill_rand(wk, kr * EMBED);
    float *wv = malloc(kr * EMBED * sizeof(float));  fill_rand(wv, kr * EMBED);
    float *wo = malloc(EMBED * qr * sizeof(float));  fill_rand(wo, EMBED * qr);
    float *qo = malloc(qr * sizeof(float));
    float *ko = malloc(kr * sizeof(float));
    float *vo = malloc(kr * sizeof(float));
    float *ao = calloc(qr, sizeof(float));

    for (int i = 0; i < 5; i++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, qr, EMBED, 1, wq, EMBED, x, 1, 0, qo, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, kr, EMBED, 1, wk, EMBED, x, 1, 0, ko, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, kr, EMBED, 1, wv, EMBED, x, 1, 0, vo, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, EMBED, qr, 1, wo, qr, ao, 1, 0, out, 1);
    }

    double attn_tot = 0;
    for (int i = 0; i < N; i++) {
        double t0 = now_us();
        cblas_sgemv(CblasRowMajor, CblasNoTrans, qr, EMBED, 1, wq, EMBED, x, 1, 0, qo, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, kr, EMBED, 1, wk, EMBED, x, 1, 0, ko, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, kr, EMBED, 1, wv, EMBED, x, 1, 0, vo, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, EMBED, qr, 1, wo, qr, ao, 1, 0, out, 1);
        attn_tot += (now_us() - t0) / 1000.;
    }
    double attn_avg = attn_tot / N;
    printf("  %.3f ms/layer\n\n", attn_avg);

    free(wq); free(wk); free(wv); free(wo);
    free(qo); free(ko); free(vo); free(ao);

    /* ═══ PROJECTIONS ═══ */
    printf("===========================================\n");
    printf("SPEED PROJECTIONS (%d layers/token)\n", LAYERS);
    printf("===========================================\n\n");

    printf("Per-layer costs:\n");
    printf("  MoE compute:   %6.3f ms\n", comp_avg);
    printf("  MoE cold SSD:  %6.2f ms  (r=%.2f c=%.2f m=%.2f)\n",
           cold_tot, cold_r, cold_c, cold_m);
    printf("  MoE warm:      %6.2f ms  (r=%.2f c=%.2f m=%.2f)\n",
           warm_tot, warm_r, warm_c, warm_m);
    printf("  Attention:     %6.3f ms\n\n", attn_avg);

    printf("Cache hit = in-app f32 cache (compute only)\n");
    printf("Cache miss = cold SSD read + convert + compute\n\n");

    printf("hit%%  | MoE/lyr |  +attn  | x%d lyr | tok/s\n", LAYERS);
    printf("------+---------+---------+---------+------\n");
    double rates[] = {0, .40, .56, .70, .85, 1.0};
    for (int i = 0; i < 6; i++) {
        double h = rates[i];
        double moe = h * comp_avg + (1 - h) * cold_tot;
        double lyr = moe + attn_avg;
        double tok = lyr * LAYERS;
        double tps = 1000. / tok;
        printf(" %4.0f%% |  %5.2f  |  %5.2f  | %6.1f  | %5.1f\n",
               h * 100, moe, lyr, tok, tps);
    }

    double ceil_lyr = comp_avg + attn_avg;
    double ceil_tok = ceil_lyr * LAYERS;
    printf("\nCompute ceiling (everything in f32 RAM):\n");
    printf("  %.3f ms/layer x %d = %.1f ms/token = %.1f tok/s\n",
           ceil_lyr, LAYERS, ceil_tok, 1000. / ceil_tok);

    return 0;
}

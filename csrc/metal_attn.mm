/* metal_attn.mm — Metal attention+KV for TinyGiant Slice A
 *
 * Runtime-compiled Metal shaders (no metallib / xcrun metal required).
 * FP16 weights live in MTLResourceStorageModeShared buffers (unified memory —
 * migrate off host mlock; do not mirror). KV cache lives only on Metal.
 *
 * Semantics match tg_attention_decode (GQA, QK-RMS, RoPE, softmax).
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>

#include "metal_attn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kShaderSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void matvec_f16(
    device const half  *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *y    [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    device const half *rowW = W + ((size_t)row * (size_t)cols);
    float sum = 0.0f;
    for (uint j = 0; j < cols; ++j)
        sum += float(rowW[j]) * x[j];
    y[row] = sum;
}

kernel void rms_norm_heads(
    device const float *x     [[buffer(0)]],
    device const float *w     [[buffer(1)]],
    device float       *y     [[buffer(2)]],
    constant uint      &hd    [[buffer(3)]],
    constant float     &eps   [[buffer(4)]],
    uint head [[thread_position_in_grid]])
{
    device const float *xh = x + (size_t)head * hd;
    device float *yh = y + (size_t)head * hd;
    float ss = 0.0f;
    for (uint i = 0; i < hd; ++i) ss += xh[i] * xh[i];
    float scale = rsqrt(ss / float(hd) + eps);
    for (uint i = 0; i < hd; ++i) yh[i] = xh[i] * scale * w[i];
}

kernel void rope_inplace(
    device float       *x    [[buffer(0)]],
    device const float *cosv [[buffer(1)]],
    device const float *sinv [[buffer(2)]],
    constant uint      &hd   [[buffer(3)]],
    constant uint      &pos  [[buffer(4)]],
    uint head [[thread_position_in_grid]])
{
    uint half = hd / 2u;
    device float *xh = x + (size_t)head * hd;
    device const float *c = cosv + (size_t)pos * half;
    device const float *s = sinv + (size_t)pos * half;
    for (uint i = 0; i < half; ++i) {
        float x1 = xh[i];
        float x2 = xh[half + i];
        xh[i]        = x1 * c[i] - x2 * s[i];
        xh[half + i] = x2 * c[i] + x1 * s[i];
    }
}

kernel void kv_append(
    device const float *k_src  [[buffer(0)]],
    device const float *v_src  [[buffer(1)]],
    device float       *k_dst  [[buffer(2)]],
    device float       *v_dst  [[buffer(3)]],
    constant uint      &hd     [[buffer(4)]],
    constant uint      &kv_max [[buffer(5)]],
    constant uint      &kv_len [[buffer(6)]],
    uint h [[thread_position_in_grid]])
{
    size_t src = (size_t)h * hd;
    size_t dst = ((size_t)h * kv_max + kv_len) * hd;
    for (uint i = 0; i < hd; ++i) {
        k_dst[dst + i] = k_src[src + i];
        v_dst[dst + i] = v_src[src + i];
    }
}

kernel void gqa_attn(
    device const float *q       [[buffer(0)]],
    device const float *k_cache [[buffer(1)]],
    device const float *v_cache [[buffer(2)]],
    device float       *out     [[buffer(3)]],
    constant uint      &hd      [[buffer(4)]],
    constant uint      &kv_max  [[buffer(5)]],
    constant uint      &seq_len [[buffer(6)]],
    constant uint      &gqa     [[buffer(7)]],
    uint head [[thread_position_in_grid]])
{
    uint kv_h = head / gqa;
    device const float *qh = q + (size_t)head * hd;
    device const float *kc = k_cache + (size_t)kv_h * kv_max * hd;
    device const float *vc = v_cache + (size_t)kv_h * kv_max * hd;
    device float *oh = out + (size_t)head * hd;

    float inv_sqrt = rsqrt(float(hd));
    // Find max score (online) — seq_len is small-to-medium for decode.
    float max_score = -1e30f;
    // Cap stack-ish scores via iterative two-pass without dynamic alloc:
    // Pass 1: max; Pass 2: softmax sum; Pass 3: weighted V.
    for (uint t = 0; t < seq_len; ++t) {
        device const float *kt = kc + (size_t)t * hd;
        float dot = 0.0f;
        for (uint i = 0; i < hd; ++i) dot += qh[i] * kt[i];
        float sc = dot * inv_sqrt;
        if (sc > max_score) max_score = sc;
    }
    float sum_exp = 0.0f;
    for (uint t = 0; t < seq_len; ++t) {
        device const float *kt = kc + (size_t)t * hd;
        float dot = 0.0f;
        for (uint i = 0; i < hd; ++i) dot += qh[i] * kt[i];
        sum_exp += exp(dot * inv_sqrt - max_score);
    }
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < hd; ++i) oh[i] = 0.0f;
    for (uint t = 0; t < seq_len; ++t) {
        device const float *kt = kc + (size_t)t * hd;
        device const float *vt = vc + (size_t)t * hd;
        float dot = 0.0f;
        for (uint i = 0; i < hd; ++i) dot += qh[i] * kt[i];
        float w = exp(dot * inv_sqrt - max_score) * inv_sum;
        for (uint i = 0; i < hd; ++i) oh[i] += w * vt[i];
    }
}
)METAL";

struct LayerBufs {
    id<MTLBuffer> wq;
    id<MTLBuffer> wk;
    id<MTLBuffer> wv;
    id<MTLBuffer> wo;
    id<MTLBuffer> q_norm;
    id<MTLBuffer> k_norm;
    id<MTLBuffer> kv_k;
    id<MTLBuffer> kv_v;
    int uploaded;
};

struct MetalAttnCtx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_matvec;
    id<MTLComputePipelineState> p_rms;
    id<MTLComputePipelineState> p_rope;
    id<MTLComputePipelineState> p_kv_append;
    id<MTLComputePipelineState> p_gqa;

    id<MTLBuffer> input;
    id<MTLBuffer> q;
    id<MTLBuffer> k;
    id<MTLBuffer> v;
    id<MTLBuffer> q_normed;
    id<MTLBuffer> k_normed;
    id<MTLBuffer> attn_out;
    id<MTLBuffer> output;
    id<MTLBuffer> rope_cos;
    id<MTLBuffer> rope_sin;
    id<MTLBuffer> u_cols_q;
    id<MTLBuffer> u_cols_kv;
    id<MTLBuffer> u_cols_o;
    id<MTLBuffer> u_hd;
    id<MTLBuffer> u_eps;
    id<MTLBuffer> u_pos;
    id<MTLBuffer> u_kv_max;
    id<MTLBuffer> u_kv_len;
    id<MTLBuffer> u_seq;
    id<MTLBuffer> u_gqa;

    LayerBufs *layers;
    int n_layers;
    int embed_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int kv_max;
    int q_dim;
    int kv_dim;
    int gqa_ratio;
    int rope_max_pos;
    int kv_len; /* global decode length; matches engine pos after each step */
    char status[128];
};

static int g_available = -1; /* -1 unknown, 0 no, 1 yes */
static char g_status[160] = "Metal attn not probed";

static id retain_obj(id obj) {
    if (obj) CFRetain((__bridge CFTypeRef)obj);
    return obj;
}

static void release_obj(id obj) {
    if (obj) CFRelease((__bridge CFTypeRef)obj);
}

static id<MTLBuffer> buf_bytes(id<MTLDevice> dev, const void *data, size_t n) {
    return retain_obj([dev newBufferWithBytes:data length:n options:MTLResourceStorageModeShared]);
}

static id<MTLBuffer> buf_len(id<MTLDevice> dev, size_t n) {
    id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    if (b && [b contents]) memset([b contents], 0, n);
    return retain_obj(b);
}

static id<MTLComputePipelineState> make_pso(id<MTLDevice> dev, id<MTLLibrary> lib,
                                            NSString *name, NSError **err) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) return nil;
    return [dev newComputePipelineStateWithFunction:fn error:err];
}

static int compile_pipelines(MetalAttnCtx *ctx, NSError **err) {
    id<MTLLibrary> lib = [ctx->device newLibraryWithSource:@(kShaderSrc)
                                                   options:nil
                                                     error:err];
    if (!lib) return -1;
    ctx->p_matvec = retain_obj(make_pso(ctx->device, lib, @"matvec_f16", err));
    ctx->p_rms = retain_obj(make_pso(ctx->device, lib, @"rms_norm_heads", err));
    ctx->p_rope = retain_obj(make_pso(ctx->device, lib, @"rope_inplace", err));
    ctx->p_kv_append = retain_obj(make_pso(ctx->device, lib, @"kv_append", err));
    ctx->p_gqa = retain_obj(make_pso(ctx->device, lib, @"gqa_attn", err));
    if (!ctx->p_matvec || !ctx->p_rms || !ctx->p_rope || !ctx->p_kv_append || !ctx->p_gqa)
        return -2;
    return 0;
}

static void dispatch_1d(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso, NSUInteger n) {
    if (n == 0) return;
    NSUInteger w = pso.threadExecutionWidth;
    if (w == 0) w = 32;
    NSUInteger tg = (n < w) ? n : w;
    [enc setComputePipelineState:pso];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

static void barrier_bufs(id<MTLComputeCommandEncoder> enc) {
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

int tg_metal_available(void) {
    if (g_available >= 0) return g_available;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            g_available = 0;
            snprintf(g_status, sizeof(g_status), "No Metal device");
            return 0;
        }
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kShaderSrc)
                                               options:nil
                                                 error:&err];
        if (!lib) {
            g_available = 0;
            snprintf(g_status, sizeof(g_status), "Metal shader compile failed: %s",
                     err ? [[err localizedDescription] UTF8String] : "?");
            return 0;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"gqa_attn"];
        if (!fn) {
            g_available = 0;
            snprintf(g_status, sizeof(g_status), "Metal shader missing gqa_attn");
            return 0;
        }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            g_available = 0;
            snprintf(g_status, sizeof(g_status), "Metal PSO failed: %s",
                     err ? [[err localizedDescription] UTF8String] : "?");
            return 0;
        }
        g_available = 1;
        snprintf(g_status, sizeof(g_status), "Metal attn ready (%s)",
                 [[dev name] UTF8String]);
        return 1;
    }
}

const char *tg_metal_status(void) {
    if (g_available < 0) tg_metal_available();
    return g_status;
}

void *tg_metal_create(int n_layers, int embed_dim, int n_heads, int n_kv_heads,
                      int head_dim, int kv_max) {
    if (n_layers <= 0 || embed_dim <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || kv_max <= 0 || (n_heads % n_kv_heads) != 0) {
        return NULL;
    }
    if (!tg_metal_available()) return NULL;

    @autoreleasepool {
        MetalAttnCtx *ctx = (MetalAttnCtx *)calloc(1, sizeof(MetalAttnCtx));
        if (!ctx) return NULL;
        ctx->device = MTLCreateSystemDefaultDevice(); /* +1 */
        if (!ctx->device) { free(ctx); return NULL; }
        ctx->queue = retain_obj([ctx->device newCommandQueue]);
        NSError *err = nil;
        if (compile_pipelines(ctx, &err) != 0) {
            snprintf(g_status, sizeof(g_status), "create: shader compile failed");
            free(ctx);
            return NULL;
        }

        ctx->n_layers = n_layers;
        ctx->embed_dim = embed_dim;
        ctx->n_heads = n_heads;
        ctx->n_kv_heads = n_kv_heads;
        ctx->head_dim = head_dim;
        ctx->kv_max = kv_max;
        ctx->q_dim = n_heads * head_dim;
        ctx->kv_dim = n_kv_heads * head_dim;
        ctx->gqa_ratio = n_heads / n_kv_heads;
        ctx->kv_len = 0;
        ctx->rope_max_pos = 0;

        ctx->layers = (LayerBufs *)calloc((size_t)n_layers, sizeof(LayerBufs));
        if (!ctx->layers) { free(ctx); return NULL; }

        size_t f = sizeof(float);
        ctx->input = buf_len(ctx->device, (size_t)embed_dim * f);
        ctx->q = buf_len(ctx->device, (size_t)ctx->q_dim * f);
        ctx->k = buf_len(ctx->device, (size_t)ctx->kv_dim * f);
        ctx->v = buf_len(ctx->device, (size_t)ctx->kv_dim * f);
        ctx->q_normed = buf_len(ctx->device, (size_t)ctx->q_dim * f);
        ctx->k_normed = buf_len(ctx->device, (size_t)ctx->kv_dim * f);
        ctx->attn_out = buf_len(ctx->device, (size_t)ctx->q_dim * f);
        ctx->output = buf_len(ctx->device, (size_t)embed_dim * f);

        uint32_t cols_q = (uint32_t)embed_dim;
        uint32_t cols_kv = (uint32_t)embed_dim;
        uint32_t cols_o = (uint32_t)ctx->q_dim;
        uint32_t hd = (uint32_t)head_dim;
        float eps = 1e-6f;
        uint32_t gqa = (uint32_t)ctx->gqa_ratio;
        uint32_t kvm = (uint32_t)kv_max;

        ctx->u_cols_q = buf_bytes(ctx->device, &cols_q, 4);
        ctx->u_cols_kv = buf_bytes(ctx->device, &cols_kv, 4);
        ctx->u_cols_o = buf_bytes(ctx->device, &cols_o, 4);
        ctx->u_hd = buf_bytes(ctx->device, &hd, 4);
        ctx->u_eps = buf_bytes(ctx->device, &eps, 4);
        ctx->u_gqa = buf_bytes(ctx->device, &gqa, 4);
        ctx->u_kv_max = buf_bytes(ctx->device, &kvm, 4);
        ctx->u_pos = buf_len(ctx->device, 4);
        ctx->u_kv_len = buf_len(ctx->device, 4);
        ctx->u_seq = buf_len(ctx->device, 4);

        for (int i = 0; i < n_layers; i++) {
            size_t kv_bytes = (size_t)n_kv_heads * (size_t)kv_max * (size_t)head_dim * f;
            ctx->layers[i].kv_k = buf_len(ctx->device, kv_bytes);
            ctx->layers[i].kv_v = buf_len(ctx->device, kv_bytes);
            ctx->layers[i].uploaded = 0;
            if (!ctx->layers[i].kv_k || !ctx->layers[i].kv_v) {
                tg_metal_destroy(ctx);
                return NULL;
            }
        }

        if (!ctx->input || !ctx->q || !ctx->k || !ctx->v || !ctx->output) {
            tg_metal_destroy(ctx);
            return NULL;
        }

        snprintf(ctx->status, sizeof(ctx->status), "Metal attn ctx (%s)",
                 [[ctx->device name] UTF8String]);
        snprintf(g_status, sizeof(g_status), "%s", ctx->status);
        return ctx;
    }
}

void tg_metal_destroy(void *ctx_v) {
    if (!ctx_v) return;
    MetalAttnCtx *ctx = (MetalAttnCtx *)ctx_v;
    @autoreleasepool {
        if (ctx->layers) {
            for (int i = 0; i < ctx->n_layers; i++) {
                release_obj(ctx->layers[i].wq);
                release_obj(ctx->layers[i].wk);
                release_obj(ctx->layers[i].wv);
                release_obj(ctx->layers[i].wo);
                release_obj(ctx->layers[i].q_norm);
                release_obj(ctx->layers[i].k_norm);
                release_obj(ctx->layers[i].kv_k);
                release_obj(ctx->layers[i].kv_v);
            }
            free(ctx->layers);
            ctx->layers = NULL;
        }
        release_obj(ctx->p_matvec);
        release_obj(ctx->p_rms);
        release_obj(ctx->p_rope);
        release_obj(ctx->p_kv_append);
        release_obj(ctx->p_gqa);
        release_obj(ctx->input);
        release_obj(ctx->q);
        release_obj(ctx->k);
        release_obj(ctx->v);
        release_obj(ctx->q_normed);
        release_obj(ctx->k_normed);
        release_obj(ctx->attn_out);
        release_obj(ctx->output);
        release_obj(ctx->rope_cos);
        release_obj(ctx->rope_sin);
        release_obj(ctx->u_cols_q);
        release_obj(ctx->u_cols_kv);
        release_obj(ctx->u_cols_o);
        release_obj(ctx->u_hd);
        release_obj(ctx->u_eps);
        release_obj(ctx->u_pos);
        release_obj(ctx->u_kv_max);
        release_obj(ctx->u_kv_len);
        release_obj(ctx->u_seq);
        release_obj(ctx->u_gqa);
        release_obj(ctx->queue);
        release_obj(ctx->device);
    }
    free(ctx);
}

int tg_metal_upload_layer_f16(
    void *ctx_v, int layer,
    const uint16_t *wq, const uint16_t *wk,
    const uint16_t *wv, const uint16_t *wo,
    const float *q_norm_w, const float *k_norm_w)
{
    if (!ctx_v || !wq || !wk || !wv || !wo || !q_norm_w || !k_norm_w) return -1;
    MetalAttnCtx *ctx = (MetalAttnCtx *)ctx_v;
    if (layer < 0 || layer >= ctx->n_layers) return -2;

    @autoreleasepool {
        size_t q_bytes = (size_t)ctx->q_dim * (size_t)ctx->embed_dim * sizeof(uint16_t);
        size_t kv_bytes = (size_t)ctx->kv_dim * (size_t)ctx->embed_dim * sizeof(uint16_t);
        size_t o_bytes = (size_t)ctx->embed_dim * (size_t)ctx->q_dim * sizeof(uint16_t);
        size_t hn = (size_t)ctx->head_dim * sizeof(float);

        LayerBufs *L = &ctx->layers[layer];
        release_obj(L->wq); release_obj(L->wk); release_obj(L->wv);
        release_obj(L->wo); release_obj(L->q_norm); release_obj(L->k_norm);
        L->wq = buf_bytes(ctx->device, wq, q_bytes);
        L->wk = buf_bytes(ctx->device, wk, kv_bytes);
        L->wv = buf_bytes(ctx->device, wv, kv_bytes);
        L->wo = buf_bytes(ctx->device, wo, o_bytes);
        L->q_norm = buf_bytes(ctx->device, q_norm_w, hn);
        L->k_norm = buf_bytes(ctx->device, k_norm_w, hn);
        if (!L->wq || !L->wk || !L->wv || !L->wo || !L->q_norm || !L->k_norm)
            return -3;
        L->uploaded = 1;
        return 0;
    }
}

int tg_metal_set_rope(void *ctx_v, const float *cos_table, const float *sin_table,
                      int max_pos) {
    if (!ctx_v || !cos_table || !sin_table || max_pos <= 0) return -1;
    MetalAttnCtx *ctx = (MetalAttnCtx *)ctx_v;
    @autoreleasepool {
        size_t n = (size_t)max_pos * (size_t)(ctx->head_dim / 2) * sizeof(float);
        release_obj(ctx->rope_cos); release_obj(ctx->rope_sin);
        ctx->rope_cos = buf_bytes(ctx->device, cos_table, n);
        ctx->rope_sin = buf_bytes(ctx->device, sin_table, n);
        if (!ctx->rope_cos || !ctx->rope_sin) return -2;
        ctx->rope_max_pos = max_pos;
        return 0;
    }
}

void tg_metal_reset_kv(void *ctx_v) {
    if (!ctx_v) return;
    MetalAttnCtx *ctx = (MetalAttnCtx *)ctx_v;
    @autoreleasepool {
        size_t kv_bytes = (size_t)ctx->n_kv_heads * (size_t)ctx->kv_max *
                          (size_t)ctx->head_dim * sizeof(float);
        for (int i = 0; i < ctx->n_layers; i++) {
            if (ctx->layers[i].kv_k && [ctx->layers[i].kv_k contents])
                memset([ctx->layers[i].kv_k contents], 0, kv_bytes);
            if (ctx->layers[i].kv_v && [ctx->layers[i].kv_v contents])
                memset([ctx->layers[i].kv_v contents], 0, kv_bytes);
        }
        ctx->kv_len = 0;
    }
}

int tg_metal_attention_decode(void *ctx_v, int layer, const float *input,
                              float *output, int pos) {
    if (!ctx_v || !input || !output) return -1;
    MetalAttnCtx *ctx = (MetalAttnCtx *)ctx_v;
    if (layer < 0 || layer >= ctx->n_layers) return -2;
    LayerBufs *L = &ctx->layers[layer];
    if (!L->uploaded) return -3;
    if (!ctx->rope_cos || pos < 0 || pos >= ctx->rope_max_pos) return -4;
    if (pos >= ctx->kv_max) return -5;

    @autoreleasepool {
        memcpy([ctx->input contents], input, (size_t)ctx->embed_dim * sizeof(float));

        uint32_t pos_u = (uint32_t)pos;
        uint32_t seq = (uint32_t)(pos + 1);
        memcpy([ctx->u_pos contents], &pos_u, 4);
        memcpy([ctx->u_kv_len contents], &pos_u, 4); /* append at index == pos */
        memcpy([ctx->u_seq contents], &seq, 4);

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        /* Q = Wq @ x */
        [enc setBuffer:L->wq offset:0 atIndex:0];
        [enc setBuffer:ctx->input offset:0 atIndex:1];
        [enc setBuffer:ctx->q offset:0 atIndex:2];
        [enc setBuffer:ctx->u_cols_q offset:0 atIndex:3];
        dispatch_1d(enc, ctx->p_matvec, (NSUInteger)ctx->q_dim);

        /* K */
        [enc setBuffer:L->wk offset:0 atIndex:0];
        [enc setBuffer:ctx->input offset:0 atIndex:1];
        [enc setBuffer:ctx->k offset:0 atIndex:2];
        [enc setBuffer:ctx->u_cols_kv offset:0 atIndex:3];
        dispatch_1d(enc, ctx->p_matvec, (NSUInteger)ctx->kv_dim);

        /* V */
        [enc setBuffer:L->wv offset:0 atIndex:0];
        [enc setBuffer:ctx->input offset:0 atIndex:1];
        [enc setBuffer:ctx->v offset:0 atIndex:2];
        [enc setBuffer:ctx->u_cols_kv offset:0 atIndex:3];
        dispatch_1d(enc, ctx->p_matvec, (NSUInteger)ctx->kv_dim);
        barrier_bufs(enc);

        /* Q RMS */
        [enc setBuffer:ctx->q offset:0 atIndex:0];
        [enc setBuffer:L->q_norm offset:0 atIndex:1];
        [enc setBuffer:ctx->q_normed offset:0 atIndex:2];
        [enc setBuffer:ctx->u_hd offset:0 atIndex:3];
        [enc setBuffer:ctx->u_eps offset:0 atIndex:4];
        dispatch_1d(enc, ctx->p_rms, (NSUInteger)ctx->n_heads);

        /* K RMS */
        [enc setBuffer:ctx->k offset:0 atIndex:0];
        [enc setBuffer:L->k_norm offset:0 atIndex:1];
        [enc setBuffer:ctx->k_normed offset:0 atIndex:2];
        dispatch_1d(enc, ctx->p_rms, (NSUInteger)ctx->n_kv_heads);
        barrier_bufs(enc);

        /* RoPE Q */
        [enc setBuffer:ctx->q_normed offset:0 atIndex:0];
        [enc setBuffer:ctx->rope_cos offset:0 atIndex:1];
        [enc setBuffer:ctx->rope_sin offset:0 atIndex:2];
        [enc setBuffer:ctx->u_hd offset:0 atIndex:3];
        [enc setBuffer:ctx->u_pos offset:0 atIndex:4];
        dispatch_1d(enc, ctx->p_rope, (NSUInteger)ctx->n_heads);

        /* RoPE K */
        [enc setBuffer:ctx->k_normed offset:0 atIndex:0];
        dispatch_1d(enc, ctx->p_rope, (NSUInteger)ctx->n_kv_heads);
        barrier_bufs(enc);

        /* KV append */
        [enc setBuffer:ctx->k_normed offset:0 atIndex:0];
        [enc setBuffer:ctx->v offset:0 atIndex:1];
        [enc setBuffer:L->kv_k offset:0 atIndex:2];
        [enc setBuffer:L->kv_v offset:0 atIndex:3];
        [enc setBuffer:ctx->u_hd offset:0 atIndex:4];
        [enc setBuffer:ctx->u_kv_max offset:0 atIndex:5];
        [enc setBuffer:ctx->u_kv_len offset:0 atIndex:6];
        dispatch_1d(enc, ctx->p_kv_append, (NSUInteger)ctx->n_kv_heads);
        barrier_bufs(enc);

        /* GQA */
        [enc setBuffer:ctx->q_normed offset:0 atIndex:0];
        [enc setBuffer:L->kv_k offset:0 atIndex:1];
        [enc setBuffer:L->kv_v offset:0 atIndex:2];
        [enc setBuffer:ctx->attn_out offset:0 atIndex:3];
        [enc setBuffer:ctx->u_hd offset:0 atIndex:4];
        [enc setBuffer:ctx->u_kv_max offset:0 atIndex:5];
        [enc setBuffer:ctx->u_seq offset:0 atIndex:6];
        [enc setBuffer:ctx->u_gqa offset:0 atIndex:7];
        dispatch_1d(enc, ctx->p_gqa, (NSUInteger)ctx->n_heads);
        barrier_bufs(enc);

        /* O = Wo @ attn_out */
        [enc setBuffer:L->wo offset:0 atIndex:0];
        [enc setBuffer:ctx->attn_out offset:0 atIndex:1];
        [enc setBuffer:ctx->output offset:0 atIndex:2];
        [enc setBuffer:ctx->u_cols_o offset:0 atIndex:3];
        dispatch_1d(enc, ctx->p_matvec, (NSUInteger)ctx->embed_dim);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.error) {
            snprintf(g_status, sizeof(g_status), "Metal decode error: %s",
                     [[cb.error localizedDescription] UTF8String]);
            return -6;
        }

        memcpy(output, [ctx->output contents],
               (size_t)ctx->embed_dim * sizeof(float));

        /* Track length from last layer's pos (engine calls layers 0..N-1 at same pos) */
        if (layer == ctx->n_layers - 1)
            ctx->kv_len = pos + 1;
        return 0;
    }
}

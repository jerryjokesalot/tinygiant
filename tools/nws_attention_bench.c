/* Benchmark the attention inner loop at TinyGiant's 4096-token context. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../csrc/libtinygiant.c"

int main(void) {
    const int seq_len = 4096;
    const int head_dim = 128;
    const int repeats = 100;
    const size_t cache_values = (size_t)seq_len * head_dim;

    float *query = malloc((size_t)head_dim * sizeof(float));
    float *keys = malloc(cache_values * sizeof(float));
    float *values = malloc(cache_values * sizeof(float));
    float *scores = malloc((size_t)seq_len * sizeof(float));
    float *output = malloc((size_t)head_dim * sizeof(float));
    if (!query || !keys || !values || !scores || !output) return 1;

    for (int d = 0; d < head_dim; d++) query[d] = (float)(d % 17 - 8) / 17.0f;
    for (size_t i = 0; i < cache_values; i++) {
        keys[i] = (float)((int)(i % 31) - 15) / 31.0f;
        values[i] = (float)((int)(i % 23) - 11) / 23.0f;
    }

    volatile float checksum = 0.0f;
    clock_t start = clock();
    for (int repeat = 0; repeat < repeats; repeat++) {
        for (int t = 0; t < seq_len; t++)
            scores[t] = tg_dot_f32(query, keys + (size_t)t * head_dim, head_dim);
        memset(output, 0, (size_t)head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++)
            tg_accumulate_scaled_f32(output, values + (size_t)t * head_dim,
                                     1.0f / (float)seq_len, head_dim);
        checksum += output[repeat % head_dim] + scores[repeat % seq_len];
    }
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("Attention inner loop: %.3f ms/run (%d tokens, head_dim=%d)\n",
           elapsed * 1000.0 / repeats, seq_len, head_dim);
    printf("Checksum: %.6f\n", (double)checksum);

    free(query);
    free(keys);
    free(values);
    free(scores);
    free(output);
    return 0;
}

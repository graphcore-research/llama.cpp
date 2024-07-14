#pragma once

#include <stdint.h>

#ifdef __cplusplus
#include <vector>
std::vector<std::pair<int, float>> sparq_topk(const float *x, int size, int k, bool use_abs);
void sparq_softmax(float *x, int size);
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sparq(const float *q,
           const float *K, int K_stride,
           const float *K_t, int K_t_stride,
           const float *V, int V_stride,
           const float *V_t, int V_t_stride,
           int seq_len, int head_dim,
           int k1, int k2, float *out);

#ifdef __AVX512F__
#define SPARQ_HALF_ENABLED
#endif

typedef uint16_t sparq_half;
float sparq_half_to_float(sparq_half x);
sparq_half sparq_float_to_half(float x);

void sparq_halfp(const float *q,
                 const sparq_half *K, int K_stride,
                 const sparq_half *K_t, int K_t_stride,
                 const sparq_half *V, int V_stride,
                 const sparq_half *V_t, int V_t_stride,
                 int seq_len, int head_dim,
                 int k1, int k2, float *out);

#ifdef __cplusplus
}
#endif

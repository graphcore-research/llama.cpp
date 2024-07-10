#pragma once

#ifdef __cplusplus
#include <vector>
using P = std::pair<int, float>;
std::vector<P> topk(const float *x, int size, int k, bool use_abs);
std::vector<float> step1(const float *q, const float *K, int K_stride, int seq_len, int head_dim, int k);
std::vector<float> step1_t(const float *q, const float *K_t, int K_t_stride, int seq_len, int head_dim, int k);
#endif

#ifdef __cplusplus
extern "C" {
#endif
void softmax(float *x, int size);
void sparq(const float *q,
           const float *K, int K_stride,
           const float *K_t, int K_t_stride,
           const float *V, int V_stride,
           const float *V_t, int V_t_stride,
           int seq_len, int head_dim,
           int k1, int k2, float *out);
void dense_attention(const float *q,
                     const float *K, int K_stride,
                     const float *K_t, int K_t_stride,
                     const float *V, int V_stride,
                     const float *V_t, int V_t_stride,
                     int seq_len, int head_dim, float *out);
#ifdef __cplusplus
}
#endif

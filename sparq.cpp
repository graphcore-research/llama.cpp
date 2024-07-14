#include "sparq.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>

#ifdef SPARQ_HALF_ENABLED
#include <immintrin.h>
#endif //SPARQ_HALF_ENABLED


//////////////////////////////////////////////////////////////////////////////////////////////
// Helpers (public)

using P = std::pair<int, float>;

std::vector<P> sparq_topk(const float *x, int size, int k, bool use_abs) {
    // Create a vector of indices and absolute values
    std::vector<P> x_idxs;
    for (int i = 0; i < size; i++) {
        x_idxs.emplace_back(i, use_abs ? std::abs(x[i]) : x[i]);
    }

    if (k >= size) {
        return x_idxs;
    }

    // NOTE: Consider sorting the output to speed up sparq
    std::nth_element(x_idxs.begin(), x_idxs.begin() + k - 1, x_idxs.end(), [](P a, P b)
                     { return a.second > b.second; });
    x_idxs.resize(k);
    return x_idxs;
}

void sparq_softmax(float *x, int size) {
    // NOTE: max_val could be calculated previously?
    float max_val = *std::max_element(x, x + size);
    float tot = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        tot += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= tot;
    }
}

#ifdef SPARQ_HALF_ENABLED

float sparq_half_to_float(sparq_half x) {
    return _mm_cvtph_ps(_mm_set1_epi16(x))[0];
}

sparq_half sparq_float_to_half(float x) {
    return _mm_cvtps_ph(_mm_set1_ps(x), 0)[0];
}

#else //!SPARQ_HALF_ENABLED

float sparq_half_to_float(sparq_half) {
    assert(false && "SparQ half precision requires AVX512");
    return 0;
}

sparq_half sparq_float_to_half(float) {
    assert(false && "SparQ half precision requires AVX512");
    return 0;
}

#endif //!SPARQ_HALF_ENABLED


//////////////////////////////////////////////////////////////////////////////////////////////
// Helpers (private)

namespace {

// Dot product of single-precision vectors, both contiguous
float dot_product(const float *__restrict__ a, const float *__restrict__ b, int n) {
    auto sum = 0.0f;
    for (auto i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Scaled accumulation into an output vector
void scaled_add(float scale, const float *__restrict__ data, int n, float *__restrict__ out) {
    for (auto i = 0; i < n; ++i) {
        out[i] += scale * data[i];
    }
}

// Dot product of single-precision vectors, both indexed
float dot_product_indexed2(const float *__restrict__ a, const float *__restrict__ b, const std::vector<P>& idx) {
    auto sum = 0.0f;
    for (P p : idx) {
        sum += a[p.first] * b[p.first];
    }
    return sum;
}

// Dot product of single-precision vectors, only the second is indexed
float dot_product_indexed1(const float *__restrict__ a, const float *__restrict__ b, const std::vector<P>& idx) {
    float sum = 0.0f;
    for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
        sum += a[i] * b[idx[i].first];
    }
    return sum;
}

#ifdef SPARQ_HALF_ENABLED

// Dot product of a single-precision and half-precision vector, both contiguous
float dot_product(const float *__restrict__ a, const sparq_half *__restrict__ b, int n) {
    constexpr int Stride = sizeof(__m512) / sizeof(float);
    __m512 sums = _mm512_setzero_ps();
    auto i = 0;
    for (; i + Stride <= n; i += Stride) {
        auto qi = _mm512_loadu_ps(reinterpret_cast<const __m512*>((a + i)));
        auto ki = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>((b + i))));
        sums = _mm512_fmadd_ps(qi, ki, sums);
    }
    auto sum = _mm512_reduce_add_ps(sums);
    for (; i < n; ++i) {
        sum += a[i] * sparq_half_to_float(b[i]);
    }
    return sum;
}

// Scaled accumulation of half-precision into a single-precision output
void scaled_add(float scale, const sparq_half *__restrict__ data, int n, float *__restrict__ out) {
    constexpr int Stride = sizeof(__m512) / sizeof(float);
    auto i = 0;
    const auto scale512 = _mm512_set1_ps(scale);
    for (; i + Stride <= n; i += Stride) {
        auto p_outi = reinterpret_cast<__m512*>(out + i);
        auto datai = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>((data + i))));
        _mm512_storeu_ps(p_outi, _mm512_fmadd_ps(scale512, datai, _mm512_loadu_ps(p_outi)));
    }
    for (; i < n; ++i) {
        out[i] += scale * sparq_half_to_float(data[i]);
    }
}

// Dot product of an indexed single-precision vector and an indexed half-precision vector
float dot_product_indexed2(const float *__restrict__ a, const sparq_half *__restrict__ b, const std::vector<P>& idx) {
    // NOTE: this naive implementation could be greatly improved, but is only applicable to the slower SparQ "default layout"
    auto sum = 0.0f;
    for (P p : idx) {
        sum += a[p.first] * sparq_half_to_float(b[p.first]);
    }
    return sum;
}

// Dot product of a single-precision vector and an indexed half-precision vector
float dot_product_indexed1(const float *__restrict__ a, const sparq_half *__restrict__ b, const std::vector<P>& idx) {
    // NOTE: this naive implementation could be greatly improved, but is only applicable to the slower SparQ "default layout"
    float sum = 0.0f;
    for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
        sum += a[i] * sparq_half_to_float(b[idx[i].first]);
    }
    return sum;
}

#else //!SPARQ_HALF_ENABLED

float dot_product(const float *__restrict__, const sparq_half *__restrict__, int) {
    assert(false && "SparQ half precision requires AVX512");
    return 0;
}

void scaled_add(float, const sparq_half *__restrict__, int, float *__restrict__) {
    assert(false && "SparQ half precision requires AVX512");
}

float dot_product_indexed2(const float *__restrict__, const sparq_half *__restrict__, const std::vector<P>&) {
    assert(false && "SparQ half precision requires AVX512");
    return 0;
}

float dot_product_indexed1(const float *__restrict__, const sparq_half *__restrict__, const std::vector<P>&) {
    assert(false && "SparQ half precision requires AVX512");
    return 0;
}

#endif //!SPARQ_HALF_ENABLED

} // namespace (anonymous)


//////////////////////////////////////////////////////////////////////////////////////////////
// Core

namespace {
template<class T>
void sparq_dense_attention(const float *__restrict__ q,
                           const T *__restrict__ K, int K_stride,
                           const T *__restrict__ V, int V_stride,
                           const T *__restrict__ V_t, int V_t_stride,
                           int seq_len, int head_dim, float *__restrict__ out) {
    std::vector<float> logits(seq_len);
    for (auto n = 0; n < seq_len; ++n) {
        auto qk = dot_product(q, K + n * K_stride, head_dim);
        logits[n] = qk / std::sqrt(static_cast<float>(head_dim));
    }

    sparq_softmax(logits.data(), seq_len);

    if (V_t != nullptr) {
        for (auto i = 0; i < head_dim; ++i) {
            out[i] = dot_product(logits.data(), V_t + i * V_t_stride, seq_len);
        }
    } else if (V != nullptr) {
        std::fill(out, out + head_dim, 0);
        for (auto n = 0; n < seq_len; ++n) {
            scaled_add(logits[n], V + n * V_stride, head_dim, out);
        }
    } else {
        assert(false && "dense_attention() requires V or V_t to be specified");
    }
}

template<class T>
void sparq_impl(const float *__restrict__ q,
                const T *__restrict__ K, int K_stride,
                const T *__restrict__ K_t, int K_t_stride,
                const T *__restrict__ V, int V_stride,
                const T *__restrict__ V_t, int V_t_stride,
                int seq_len, int head_dim,
                int k1, int k2, float *__restrict__ out) {
    // Dense fall-back
    if (k1 == 0 && k2 == 0) {
        sparq_dense_attention(q, K, K_stride, V, V_stride, V_t, V_t_stride, seq_len, head_dim, out);
        return;
    }
    k1 = std::min(k1, head_dim);
    k2 = std::min(k2, seq_len);

    // Compute approximate attention scores
    std::vector<float> s_hat(seq_len, 0.0f);
    std::vector<P> idx1 = sparq_topk(q, head_dim, k1, /*use_abs=*/true);
    if (K_t != nullptr) {
        for (P p : idx1) {
            scaled_add(q[p.first], K_t + p.first * K_t_stride, seq_len, s_hat.data());
        }
    } else {
        for (int i = 0; i < seq_len; i++) {
            s_hat[i] = dot_product_indexed2(q, K + i * K_stride, idx1);
        }
    }

    // Find top-k2 approximate scores
    std::vector<P> idx2 = sparq_topk(s_hat.data(), s_hat.size(), k2, /*use_abs=*/false);

    // Calculate scores for top-k2, s -- (k2,)
    std::vector<float> s(k2, 0.0);
    for (int i = 0; i < k2; i++) {
        auto qk = dot_product(q, K + idx2[i].first * K_stride, head_dim);
        s[i] = qk / std::sqrt(static_cast<float>(head_dim));
    }
    sparq_softmax(s.data(), s.size());

    // Perform weighted sum of values
    if (V != nullptr) {  // V -- (seq_len, head_dim)
        std::fill(out, out + head_dim, 0.0f);
        for (int i = 0; i < k2; i++) {
            scaled_add(s[i], V + idx2[i].first * V_stride, head_dim, out);
        }
    } else if (V_t != nullptr) {  // V_t -- (head_dim, seq_len)
        for (int j = 0; j < head_dim; j++) {
            out[j] = dot_product_indexed1(s.data(), V_t + j * V_t_stride, idx2);
        }
    } else {
        assert(false && "sparq() requires V or V_t to be specified");
    }
}
} // namespace (anonymous)


//////////////////////////////////////////////////////////////////////////////////////////////
// API

void sparq(const float *__restrict__ q,
           const float *__restrict__ K, int K_stride,
           const float *__restrict__ K_t, int K_t_stride,
           const float *__restrict__ V, int V_stride,
           const float *__restrict__ V_t, int V_t_stride,
           int seq_len, int head_dim,
           int k1, int k2, float *__restrict__ out) {
    return sparq_impl(q, K, K_stride, K_t, K_t_stride, V, V_stride, V_t, V_t_stride,
                      seq_len, head_dim, k1, k2, out);
}

void sparq_halfp(const float *__restrict__ q,
                 const sparq_half *__restrict__ K, int K_stride,
                 const sparq_half *__restrict__ K_t, int K_t_stride,
                 const sparq_half *__restrict__ V, int V_stride,
                 const sparq_half *__restrict__ V_t, int V_t_stride,
                 int seq_len, int head_dim,
                 int k1, int k2, float *__restrict__ out) {
    return sparq_impl(q, K, K_stride, K_t, K_t_stride, V, V_stride, V_t, V_t_stride,
                      seq_len, head_dim, k1, k2, out);
}

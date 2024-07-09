#include "sparq.h"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>

std::vector<P> topk(const float *x, int size, int k, bool use_abs)
{
    // Create a vector of indices and absolute values
    std::vector<P> x_idxs;
    for (int i = 0; i < size; i++)
        x_idxs.emplace_back(i, use_abs ? std::abs(x[i]) : x[i]);

    // Sort the vector based on abs values
    std::sort(x_idxs.begin(), x_idxs.end(), [](P a, P b)
              { return a.second > b.second; });

    // Keep only the top k pairs
    if (k < size)
        x_idxs.resize(k);
    return x_idxs;
}

std::vector<P> topk_fast(const float *x, int size, int k, bool use_abs)
{
    // Create a vector of indices and absolute values
    std::vector<P> x_idxs;
    for (int i = 0; i < size; i++)
        x_idxs.emplace_back(i, use_abs ? std::abs(x[i]) : x[i]);

    if (k >= size)
    {
        return x_idxs;
    }

    std::nth_element(x_idxs.begin(), x_idxs.begin() + k - 1, x_idxs.end(), [](P a, P b)
                     { return a.second > b.second; });
    x_idxs.resize(k);
    return x_idxs;
}

// K -- (seq_len, head_dim)
std::vector<float> step1(const float *q, const float *K, int seq_len, int head_dim, int k)
{
    // Output vector
    std::vector<float> out(seq_len, 0.0);

    std::vector<P> idx = topk_fast(q, head_dim, k, true);
    for (int i = 0; i < seq_len; i++)
    {
        for (P p : idx)
        {
            int j = p.first;
            out[i] += q[j] * K[i * head_dim + j];
        }
    }

    return out;
}

// K -- (head_dim, seq_len)
std::vector<float> step1_t(const float *q, const float *K_t, int seq_len, int head_dim, int k)
{
    // Output vector
    std::vector<float> out(seq_len, 0.0);

    std::vector<P> idx = topk_fast(q, head_dim, k, true);

    for (P p : idx)
    {
        int j = p.first;
        for (int i = 0; i < seq_len; i++)
        {
            out[i] += q[j] * K_t[j * seq_len + i];
        }
    }

    return out;
}

void softmax(float *x, int size)
{
    // TODO: max_val could be calculated previously?
    float max_val = *std::max_element(x, x + size);
    float tot = 0.0;
    for (int i = 0; i < size; i++)
    {
        x[i] = std::exp(x[i] - max_val);
        tot += x[i];
    }
    for (int i = 0; i < size; i++)
        x[i] /= tot;
}

void sparq(const float *q, const float *K, const float *K_t, const float *V, const float *V_t,
           int seq_len, int head_dim, int k1, int k2, float *out)
{
    // if (k1 == INT32_MAX && k2 == INT32_MAX) {
    //     return dense_attention(q, K, K_t, V, V_t, seq_len, head_dim, out);
    // }

    // Step 1
    std::vector<float> s_hat;
    if (K_t == nullptr)
    {
        s_hat = step1(q, K, seq_len, head_dim, k1);
    }
    else
    {
        s_hat = step1_t(q, K_t, seq_len, head_dim, k1);
    }

    // Find top-k2 approximate scores
    std::vector<P> topk_out = topk_fast(s_hat.data(), s_hat.size(), k2, false);

    // Calculate scores for top-k2, s -- (k2, )
    std::vector<float> s(k2, 0.0);
    for (int i = 0; i < k2; i++)
    {
        int idx = topk_out[i].first;
        for (int j = 0; j < head_dim; j++)
        {
            s[i] += q[j] * K[idx * head_dim + j];
        }
        s[i] /= std::sqrt(head_dim);
    }
    // softmax(s);
    softmax(s.data(), s.size());

    // Initialise output tensor (maybe unnecessary?)
    for (int i = 0; i < head_dim; i++)
    {
        out[i] = 0;
    }

    // Perform weighted sum of values, V -- (seq_len, head_dim)
    if (V != nullptr)
    {
        for (int i = 0; i < k2; i++)
        {
            float &w = s[i];
            int &idx = topk_out[i].first;
            for (int j = 0; j < head_dim; j++)
            {
                out[j] += w * V[idx * head_dim + j];
            }
        }
    }

    // V_t -- (head_dim, seq_len)
    else if (V_t != nullptr)
    {
        for (int j = 0; j < head_dim; j++)
        {
            for (int i = 0; i < k2; i++)
            {
                float &w = s[i];
                int &idx = topk_out[i].first;
                out[j] += w * V_t[j * seq_len + idx];
            }
        }
    }
    else
    {
        std::cout << "Pass either V or V_t" << std::endl;
        return;
    }
}

void dense_attention(const float *q,
                     const float *K, int K_stride,
                     const float */*K_t*/, int /*K_t_stride*/,
                     const float *V, int V_stride,
                     const float *V_t, int V_t_stride,
                     int seq_len, int head_dim, float *out) {
    assert(K != nullptr && "dense_attention() requires K (K_t is unimplemented)");

    std::vector<float> logits(seq_len);
    for (auto n = 0; n < seq_len; ++n) {
        auto sum = 0.0f;
        for (auto i = 0; i < head_dim; ++i) {
            sum += q[i] * K[n * K_stride + i];
        }
        logits[n] = sum / std::sqrt(static_cast<float>(head_dim));
    }

    softmax(logits.data(), seq_len);

    if (V_t != nullptr) {
        for (auto i = 0; i < head_dim; ++i) {
            auto mix = 0.0f;
            for (auto n = 0; n < seq_len; ++n) {
                mix += logits[n] * V_t[i * V_t_stride + n];
            }
            out[i] = mix;
        }
    } else if (V != nullptr) {
        std::fill(out, out + head_dim, 0);
        for (auto n = 0; n < seq_len; ++n) {
            for (auto i = 0; i < head_dim; ++i) {
                out[i] += logits[n] * V[n * V_stride + i];
            }
        }
    } else {
        assert(false && "dense_attention() requires V or V_t to be specified");
    }
}

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <chrono>

using P = std::pair<int, float>;

std::vector<P> topk(const float *x, int size, int k)
{
    // Create a vector of indices and absolute values
    std::vector<P> x_idxs;
    for (int i = 0; i < size; i++)
        x_idxs.emplace_back(i, std::abs(x[i]));

    // Sort the vector based on abs values
    std::sort(x_idxs.begin(), x_idxs.end(), [](P a, P b)
              { return a.second > b.second; });

    // Keep only the top k pairs
    if (k < size)
        x_idxs.resize(k);
    return x_idxs;
}

// K -- (seq_len, head_dim)
std::vector<float> step1(const float *q, const float *K, int seq_len, int head_dim, int k)
{
    // Output vector
    std::vector<float> out(seq_len, 0.0);

    std::vector<P> idx = topk(q, head_dim, k);
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
std::vector<float> step1_t(const float *q, const float *K, int seq_len, int head_dim, int k)
{
    // Output vector
    std::vector<float> out(seq_len, 0.0);

    std::vector<P> idx = topk(q, head_dim, k);

    for (P p : idx)
    {
        int j = p.first;
        for (int i = 0; i < seq_len; i++)
        {
            out[i] += q[j] * K[j * seq_len + i];
        }
    }

    return out;
}

void softmax(std::vector<float> &x)
{
    // TODO: max_val could be calculated previously?
    float max_val = *std::max_element(x.begin(), x.end());
    float tot = 0.0;
    for (size_t i = 0; i < x.size(); i++)
    {
        x[i] = std::exp(x[i] - max_val);
        tot += x[i];
    }
    for (size_t i = 0; i < x.size(); i++)
        x[i] /= tot;
}

void sparq(const float *q, const float *K, const float *V,
           int seq_len, int head_dim, int k1, int k2, float *out)
{
    // Step 1 - Probably requires K^T (no softmax for now)
    std::vector<float> s_hat = step1_t(q, K, seq_len, head_dim, k1);

    // Find top-k2 approximate scores
    std::vector<P> topk_out = topk(s_hat.data(), s_hat.size(), k2);

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
    softmax(s);

    // Perform weighted sum of values
    // Comments:
    // * (!) Pointer aliasing
    // * Declare that pointer aliasing is not allowed (restrict keyword)
    // * Compiler might not know V and out are separate in memory
    for (int i = 0; i < k2; i++)
    {
        float w = s[i];
        int idx = topk_out[i].first;
        for (int j = 0; j < head_dim; j++)
        {
            out[j] += w * V[idx * head_dim + j];
        }
    }
}

void test_step_1(bool use_transposed)
{
    // Tensor sizes
    const int head_dim{128};
    const int seq_len{1000000};

    // Fill with random numbers
    std::vector<float> q(head_dim);
    std::vector<float> K(head_dim * seq_len);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 10.0);
    for (int i = 0; i < head_dim; i++)
        q[i] = dis(gen);

    for (int i = 0; i < head_dim * seq_len; i++)
        K[i] = dis(gen);

    int k1 = 128; // rank

    using clock = std::chrono::system_clock;
    // Step 1
    auto size = seq_len * std::min(k1, head_dim) * sizeof(K.front());

    auto t0 = clock::now();
    std::vector<float> out;
    if (use_transposed)
        out = step1_t(q.data(), K.data(), seq_len, head_dim, k1);
    else
        out = step1(q.data(), K.data(), seq_len, head_dim, k1);
    auto duration = clock::now() - t0;
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    std::cerr << "  Size: " << size / 1e9 << " GB"               //
              << "\n    In: " << elapsed << " s"                 //
              << "\n    At: " << size / elapsed / 1e9 << " GB/s" //
              << std::endl;
}

void test_sparq()
{
    // Tensor sizes
    const int head_dim{128};
    const int seq_len{1000000};

    // Fill with random numbers
    std::vector<float> q(head_dim);
    std::vector<float> K(head_dim * seq_len);
    std::vector<float> V(head_dim * seq_len);
    std::vector<float> out(head_dim);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 10.0);
    for (int i = 0; i < head_dim; i++)
        q[i] = dis(gen);

    for (int i = 0; i < head_dim * seq_len; i++)
    {
        K[i] = dis(gen);
        V[i] = dis(gen);
    }

    int k1 = head_dim; // rank
    int k2 = 1;

    using clock = std::chrono::system_clock;
    // Step 1
    auto size = seq_len * std::min(k1, head_dim) * sizeof(K.front());
    // Step 2
    size += 2 * k2 * head_dim * sizeof(K.front());

    auto t0 = clock::now();
    sparq(q.data(), K.data(), V.data(), seq_len, head_dim, k1, k2, out.data());
    auto duration = clock::now() - t0;
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    std::cerr << "  Size: " << size / 1e9 << " GB"               //
              << "\n    In: " << elapsed << " s"                 //
              << "\n    At: " << size / elapsed / 1e9 << " GB/s" //
              << std::endl;
}

int main()
{
    test_sparq();
    return 0;
}

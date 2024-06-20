#include "sparq.h"
#include <cassert>
#include <cmath>
#include <random>
#include <iostream>
#include <chrono>

void test_step_1()
{
    // head_dim = 4, seq_len = 3
    std::vector<float> q{-5.0, 1.0, 3.0, -2.0};
    std::vector<float> K{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> K_t{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};

    std::vector<float> out1 = step1(q.data(), K.data(), 3, 4, 2);
    std::vector<float> out2 = step1_t(q.data(), K_t.data(), 3, 4, 2);

    std::vector<float> expected{4, -4, -12};
    assert(out1 == expected);
    assert(out2 == expected);
}

void test_softmax()
{
    std::vector<float> x1{10, 10, 10, 10};
    std::vector<float> x2{1, 2, 3, 4};
    softmax(x1.data(), x1.size());
    softmax(x2.data(), x2.size());

    std::vector<float> expected1{0.25, 0.25, 0.25, 0.25};
    std::vector<float> expected2{0.0321, 0.0871, 0.2369, 0.6439};

    for (size_t i = 0; i < x1.size(); i++)
    {
        float tol = 0.001;
        assert(std::abs(x1[i] - expected1[i]) < tol);
        assert(std::abs(x2[i] - expected2[i]) < tol);
    }
}

void test_sparq()
{
    // seq_len = 4, head_size = 2
    // k1 = 1, k2 = 2
    std::vector<float> q{1.0, -0.5};
    std::vector<float> K{1, 0, 4, -10, 2, 5, -3, 10};
    std::vector<float> K_t{1, 4, 2, -3, 0, -10, 50, 10};
    std::vector<float> V{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> V_t{1, 3, 5, 7, 2, 4, 6, 8};

    // After step 1, s_hat = {1, 4, 2, -3}
    // => idxs = {1, 2} (not 3!)
    std::vector<float> s{(4 + 5) / std::sqrt(2.f), (2 - 2.5f) / std::sqrt(2.f)};
    softmax(s.data(), s.size());
    float out[2], out_t[2], out_v_t[2], expected[2];
    expected[0] = s[0] * V[2] + s[1] * V[4];
    expected[1] = s[0] * V[3] + s[1] * V[5];
    
    // Check using either K and K_t
    sparq(q.data(), K.data(), nullptr, V.data(), nullptr, 4, 2, 1, 2, out);
    sparq(q.data(), K.data(), K_t.data(), V.data(), nullptr, 4, 2, 1, 2, out_t);

    // Check using V_t
    sparq(q.data(), K.data(), nullptr, nullptr, V_t.data(), 4, 2, 1, 2, out_v_t);

    for (int i = 0; i < 2; i++)
    {
        float tol = 0.001;
        assert(std::abs(out[i] - expected[i]) < tol);
        assert(std::abs(out_t[i] - expected[i]) < tol);
        assert(std::abs(out_v_t[i] - expected[i]) < tol);
    }
}

void benchmark_step_1(bool use_transposed)
{
    // Tensor sizes
    const int head_dim{128};
    const int seq_len{1000000};

    // Fill with random numbers
    std::vector<float> q(head_dim);
    std::vector<float> K(head_dim * seq_len);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 5.0);
    for (int i = 0; i < head_dim; i++)
        q[i] = dis(gen);

    for (int i = 0; i < head_dim * seq_len; i++)
        K[i] = dis(gen);

    int k1 = 16; // rank

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

void benchmark_sparq()
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
    std::normal_distribution<float> dis(0.0, 5.0);
    for (int i = 0; i < head_dim; i++)
        q[i] = dis(gen);

    for (int i = 0; i < head_dim * seq_len; i++)
    {
        K[i] = dis(gen);
        V[i] = dis(gen);
    }

    int k1 = head_dim;
    int k2 = seq_len;

    using clock = std::chrono::system_clock;
    // Step 1
    auto size = seq_len * std::min(k1, head_dim) * sizeof(K.front());
    // Step 2
    size += 2 * k2 * head_dim * sizeof(K.front());

    auto t0 = clock::now();
    sparq(q.data(), K.data(), K.data(), V.data(), nullptr, seq_len, head_dim, k1, k2, out.data());
    auto duration = clock::now() - t0;
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    std::cerr << "  Size: " << size / 1e9 << " GB"               //
              << "\n    In: " << elapsed << " s"                 //
              << "\n    At: " << size / elapsed / 1e9 << " GB/s" //
              << std::endl;
}

int main()
{
    test_step_1();
    test_softmax();
    test_sparq();
}

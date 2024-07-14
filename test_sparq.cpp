#include "sparq.h"
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <random>
#include <iostream>
#include <chrono>
#include <algorithm>

namespace {

void test_softmax()
{
    std::cerr << "test_softmax()" << std::endl;
    std::vector<float> x1{10, 10, 10, 10};
    std::vector<float> x2{1, 2, 3, 4};
    sparq_softmax(x1.data(), x1.size());
    sparq_softmax(x2.data(), x2.size());

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
    std::cerr << "test_sparq()" << std::endl;
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
    sparq_softmax(s.data(), s.size());
    float out[2], out_t[2], out_v_t[2], expected[2];
    expected[0] = s[0] * V[2] + s[1] * V[4];
    expected[1] = s[0] * V[3] + s[1] * V[5];

    // Check using either K and K_t
    sparq(q.data(), K.data(), 2, nullptr, -1, V.data(), 2, nullptr, -1, 4, 2, 1, 2, out);
    sparq(q.data(), K.data(), 2, K_t.data(), 4, V.data(), 2, nullptr, -1, 4, 2, 1, 2, out_t);

    // Check using V_t
    sparq(q.data(), K.data(), 2, nullptr, -1, nullptr, -1, V_t.data(), 4, 4, 2, 1, 2, out_v_t);

    for (int i = 0; i < 2; i++)
    {
        float tol = 0.001;
        assert(std::abs(out[i] - expected[i]) < tol);
        assert(std::abs(out_t[i] - expected[i]) < tol);
        assert(std::abs(out_v_t[i] - expected[i]) < tol);
    }
}

void test_small()
{
    std::cerr << "test_small()" << std::endl;

    // Generated test data:
    // import torch
    // torch.manual_seed(100)
    // q, K, V = torch.randn(2), torch.randn(3, 2), torch.randn(3, 2)
    // o = torch.softmax((q @ K.T) / q.shape[-1]**.5, -1) @ V
    // for t in "qKVo":
    //     print(f"float {t}[] = {{" + ", ".join(f"{x:.3f}" for x in globals()[t].flatten()) + "};")

    constexpr int head_dim = 2;
    constexpr int seq_len = 3;
    float q[] = {0.361, -0.286};
    float K[] = {-0.394, 0.243, -1.383, -2.313, -0.317, -0.866};
    float V[] = {1.748, -0.276, -0.975, 0.479, -2.365, -0.805};
    float o[] = {-0.710, -0.190};

    float V_t[head_dim * seq_len];
    float K_t[head_dim * seq_len];
    for (auto n = 0; n < seq_len; ++n) {
        for (auto i = 0; i < head_dim; ++i) {
            V_t[i * seq_len + n] = V[n * head_dim + i];
            K_t[i * seq_len + n] = K[n * head_dim + i];
        }
    }

    // Checker
    float out[head_dim];
    auto check_out = [o, &out](const std::string& name) {
        for (auto i = 0; i < head_dim; ++i) {
            if (!(std::abs(out[i] - o[i]) < 2e-3)) {
                std::cerr << "Assert failed for " << name
                    << ", expected o[" << i << "] = " << o[i]
                    << ", actual out[" << i << "] = " << out[i] << std::endl;
                assert(false);
            }
        }
    };

    // Tests
    std::fill(out, out + head_dim, NAN);
    sparq(q, K, /*K_stride=*/head_dim, /*K_t=*/nullptr, /*K_t_stride*/0,
        V, /*V_stride*/head_dim, /*V_t*/nullptr, /*V_t_stride*/0,
        seq_len, head_dim, 0, 0, out);
    check_out("dense_attention");

    std::fill(out, out + head_dim, NAN);
    sparq(q, K, /*K_stride=*/head_dim, /*K_t=*/nullptr, /*K_t_stride*/0,
        /*V*/nullptr, /*V_stride*/0, V_t, /*V_t_stride*/seq_len,
        seq_len, head_dim, 0, 0, out);
    check_out("dense_attention(V_t)");

    std::fill(out, out + head_dim, NAN);
    sparq(q, K, /*K_stride=*/head_dim, /*K_t=*/nullptr, /*K_t_stride*/0,
        V, /*V_stride*/head_dim, /*V_t*/nullptr, /*V_t_stride*/0,
        seq_len, head_dim, head_dim, seq_len, out);
    check_out("sparq");

    std::fill(out, out + head_dim, NAN);
    sparq(q, K, /*K_stride=*/head_dim, /*K_t=*/K_t, /*K_t_stride*/seq_len,
        /*V*/nullptr, /*V_stride*/0, V_t, /*V_t_stride*/seq_len,
        seq_len, head_dim, head_dim, seq_len, out);
    check_out("sparq(K_t, V_t)");
}

#ifdef SPARQ_HALF_ENABLED
std::vector<sparq_half> to_half(const std::vector<float>& x) {
    std::vector<sparq_half> out(x.size());
    std::transform(x.begin(), x.end(), out.begin(), sparq_float_to_half);
    return out;
}
#endif // SPARQ_HALF_ENABLED

void test_small_half()
{
#ifdef SPARQ_HALF_ENABLED
    std::cerr << "test_small_half()" << std::endl;

    // Simple round-trip test of sparq_float_to_half & sparq_half_to_float
    auto h = sparq_float_to_half(12.5f);
    assert(sizeof(h) == 2);
    assert(sparq_half_to_float(h) == 12.5f);

    // ### Test as per test_small()
    constexpr int head_dim = 2;
    constexpr int seq_len = 3;
    auto q = std::vector<float>({0.361, -0.286});
    auto K = to_half({-0.394, 0.243, -1.383, -2.313, -0.317, -0.866});
    auto V = to_half({1.748, -0.276, -0.975, 0.479, -2.365, -0.805});
    auto o = std::vector<float>({-0.710, -0.190});

    std::vector<sparq_half> V_t(head_dim * seq_len);
    std::vector<sparq_half> K_t(head_dim * seq_len);
    for (auto n = 0; n < seq_len; ++n) {
        for (auto i = 0; i < head_dim; ++i) {
            V_t[i * seq_len + n] = V[n * head_dim + i];
            K_t[i * seq_len + n] = K[n * head_dim + i];
        }
    }

    // Checker
    std::vector<float> out(head_dim);
    auto check_out = [&o, &out](const std::string& name) {
        for (auto i = 0; i < head_dim; ++i) {
            if (std::abs(out[i] - o[i]) > 2e-3) {
                std::cerr << "Assert failed for " << name
                    << ", expected o[" << i << "] = " << o[i]
                    << ", actual out[" << i << "] = " << out[i] << std::endl;
                assert(false);
            }
        }
    };

    // Tests
    sparq_halfp(q.data(), K.data(), /*K_stride=*/head_dim, /*K_t=*/nullptr, /*K_t_stride*/0,
        /*V*/nullptr, /*V_stride*/0, /*V_t*/V_t.data(), /*V_t_stride*/seq_len,
        seq_len, head_dim, 0, 0, out.data());
    check_out("dense_attention");

#endif // SPARQ_HALF_ENABLED
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
    sparq(q.data(), K.data(), head_dim, K.data(), seq_len, V.data(), head_dim, nullptr, -1, seq_len, head_dim, k1, k2, out.data());
    auto duration = clock::now() - t0;
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    std::cerr << "  Size: " << size / 1e9 << " GB"               //
              << "\n    In: " << elapsed << " s"                 //
              << "\n    At: " << size / elapsed / 1e9 << " GB/s" //
              << std::endl;
}

} // namespace (anonymous)

int main()
{
    test_softmax();
    test_sparq();
    test_small();
    test_small_half();
    std::cerr << "-- ALL TESTS PASSED --" << std::endl;
}

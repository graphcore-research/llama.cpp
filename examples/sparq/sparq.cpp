#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include "ggml.h"

// Compiler doesn't like missing declaration
void print_tensor(const ggml_tensor* t);

void print_tensor(const ggml_tensor* t)
{
    for (int i3 = 0; i3 < t->ne[3]; i3++)
    {
        for (int i2 = 0; i2 < t->ne[2]; i2++)
        {
            for (int i1 = 0; i1 < t->ne[1]; i1++)
            {
                for (int i0 = 0; i0 < t->ne[0]; i0++)
                {
                    std::cout << ggml_get_f32_nd(t, i0, i1, i2, i3) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

ggml_tensor* sparq_attn(ggml_context* ctx, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V, int k1, int k2)
{
    // Note -> dimensions in ne are inverted!
    // Q -- (bs * num_heads, 1, head_dim)
    // K, V -- (bs * num_heads, seq_len, head_dim)
    
    int head_dim = Q->ne[0];

    // ===== Step 1 =====

    // // i1 -- (bs * num_heads, 1, k1)
    // ggml_tensor* i1 = ggml_top_k(ctx, ggml_abs(ctx, Q), k1);
    // i1 = ggml_reshape_2d(ctx, i1, i1->ne[0], i1->ne[2]); // is this doing unnecessary copying?
    // // Q_hat -- (bs * num_heads, 1, k1)
    // ggml_tensor* Q_hat = ggml_transpose(ctx, (ctx, ggml_transpose(ctx, Q), i1));
    // // K_hat -- (bs * num_heads, seq_len, k1)
    // ggml_tensor* K_hat = ggml_transpose(ctx, ggml_get_rows(ctx, ggml_transpose(ctx, K), i1));
    // // scores_hat -- (bs * num_heads, 1, seq_len), ignore softmax for now, use just for top-k2
    // ggml_tensor* scores_hat = ggml_mul_mat(ctx, K_hat, Q_hat);

    // For "sparse-V" use normal scores
    // scores -- (bs * num_heads, 1, seq_len)
    ggml_tensor* scores = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, K, Q), nullptr, nullptr, 1.0f/sqrtf(float(head_dim)), 0.0f);
    
    // ===== Step 2 =====

    // i2 -- (bs * num_heads, 1, k2)
    ggml_tensor* i2 = ggml_top_k(ctx, scores, k2);

    // Squeeze to (bs * num_heads, k2) for gather (get_rows) op
    i2 = ggml_reshape_2d(ctx, i2, i2->ne[0], i2->ne[2]); // is this doing unnecessary copying?

    // scores_gathered -- (bs * num_heads, k2, 1)
    ggml_tensor* scores_gathered = ggml_get_rows(ctx, ggml_transpose(ctx, scores), i2);

    // V_gathered -- (bs * num_heads, k2, head_dim)
    ggml_tensor* V_gathered = ggml_get_rows(ctx, V, i2);

    // out -- (bs * num_heads, 1, head_dim)
    return ggml_mul_mat(ctx, ggml_transpose(ctx, V_gathered), ggml_transpose(ctx, scores_gathered));
}


int main()
{
    // Define memory allocation
    // (note: not sure how to interpret the no_alloc param, breaks if true)
    ggml_init_params params {
        ggml_tensor_overhead() * 3 + ggml_graph_overhead() + 1024,
        nullptr,
        false
    };

    // Create the context (and allocate the memory)
    ggml_context* ctx = ggml_init(params);

    // Define the tensors
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4); // (4, 3) tensor
    ggml_tensor* idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2); // (2,) tensor

    // Build the computational graph
    ggml_tensor* out = ggml_get_rows(ctx, x, idx);
    // a -- (bs, num_rows, num_cols)
    // b -- (bs, k)
    // out -- (bs, k, num_cols)


    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Set tensor values
    std::srand(42);
    for (int i = 0; i < 3*4; i++)
    {
        ggml_set_f32_1d(x, i, std::rand() % 100);
    }
    int idxs[] {3, 0};
    for (int i = 0; i < 2; i++)
    {
        ggml_set_i32_1d(idx, i, idxs[i]);
    }

    // Perform the computation
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Might need to release memory if repeating
    // ...

    // Print the data
    std::cout << "Original tensor:\n";
    print_tensor(x);
    std::cout << "Gather indices:\n";
    print_tensor(idx);
    std::cout << "Gathered tensor:\n";
    print_tensor(out);

    return 0;
}

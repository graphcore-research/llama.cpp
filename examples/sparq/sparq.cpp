#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include "ggml.h"


// Declarations
void print_tensor(
    const ggml_tensor *t
);
void print_structure(
    const ggml_tensor *t, 
    std::string name
);
ggml_tensor *exact_attn(
    ggml_context *ctx, 
    ggml_tensor *Q, 
    ggml_tensor *K, 
    ggml_tensor *V,
    ggml_tensor *temp
);
ggml_tensor *sparq_attn(
    ggml_context *ctx, 
    ggml_tensor *Q, 
    ggml_tensor *K, 
    ggml_tensor *V,
    ggml_tensor *temp,
    int k1, 
    int k2
);


void print_tensor(const ggml_tensor *t) {
    for (int i3 = 0; i3 < t->ne[3]; i3++) {
        for (int i2 = 0; i2 < t->ne[2]; i2++) {
            for (int i1 = 0; i1 < t->ne[1]; i1++) {
                for (int i0 = 0; i0 < t->ne[0]; i0++) {
                    std::cout << ggml_get_f32_nd(t, i0, i1, i2, i3) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}


void print_structure(const ggml_tensor *t, std::string name) {
    printf("\n%s:\n", name.c_str());
    printf("    elements (ne): %lu x %lu x %lu\n", t->ne[0], t->ne[1], t->ne[2]);
    printf("       bytes (nb): %lu x %lu x %lu\n\n", t->nb[0], t->nb[1], t->nb[2]);
} 


ggml_tensor *exact_attn(
    ggml_context *ctx, 
    ggml_tensor *Q, 
    ggml_tensor *K, 
    ggml_tensor *V,
    ggml_tensor *temp
) {
    ggml_tensor *scores;
    ggml_tensor *V_trans;
    ggml_tensor *output;

    // scores -- (batch_size * num_heads, 1, sequence_length)
    scores = ggml_mul_mat(ctx, K, Q);
    scores = ggml_div(ctx, scores, temp);
    scores = ggml_soft_max(ctx, scores);
    print_structure(scores, "scores");

    // V_trans -- (batch_size * num_heads, head_dim, sequence_length)
    V_trans = ggml_transpose(ctx, V);
    V_trans = ggml_cont(ctx, V_trans);
    print_structure(V_trans, "V_trans");

    // output -- (batch_size * num_heads, 1, head_dim)
    output = ggml_mul_mat(ctx, V_trans, scores);
    print_structure(output, "output");

    return output;
}


ggml_tensor *sparq_attn(
    ggml_context *ctx, 
    ggml_tensor *Q, 
    ggml_tensor *K, 
    ggml_tensor *V,
    ggml_tensor *temp,
    int k1, 
    int k2
) {
    ggml_tensor *i1;
    ggml_tensor *i2;
    ggml_tensor *Q_hat;
    ggml_tensor *K_hat;
    ggml_tensor *s_hat;
    ggml_tensor *Q_sum;
    ggml_tensor *Q_hat_sum;
    ggml_tensor *scaled_temp;
    ggml_tensor *K_top_k;
    ggml_tensor *s_top_k;
    ggml_tensor *V_top_k;
    ggml_tensor *alpha;
    ggml_tensor *output;

    // ========================================================================
    // STEP 1 =================================================================
    // ========================================================================

    // i1 -- (batch_size * num_heads, 1, k1)
    i1 = ggml_top_k(ctx, ggml_abs(ctx, Q), k1);
    i1 = ggml_cont(ctx, i1);
    i1 = ggml_reshape_2d(ctx, i1, i1->ne[0], i1->ne[2]); 
    print_structure(i1, "i1");

    // Q_hat -- (batch_size * num_heads, 1, k1)
    Q_hat = ggml_transpose(ctx, Q);
    Q_hat = ggml_get_rows(ctx, Q_hat, i1);
    Q_hat = ggml_transpose(ctx, Q_hat);
    print_structure(Q_hat, "Q_hat");

    // Q_sum -- (batch_size * num_heads)
    Q_sum = ggml_abs(ctx, Q);
    Q_sum = ggml_sum_rows(ctx, Q_sum);
    Q_sum = ggml_reshape_1d(ctx, Q_sum, Q_sum->ne[2]);
    print_structure(Q_sum, "Q_sum");

    // Q_hat_sum -- (batch_size * num_heads)
    Q_hat_sum = ggml_abs(ctx, Q_hat);
    Q_hat_sum = ggml_sum_rows(ctx, Q_hat_sum);
    Q_hat_sum = ggml_reshape_1d(ctx, Q_hat_sum, Q_hat_sum->ne[2]);
    print_structure(Q_hat_sum, "Q_hat_sum");

    // scaled_temp -- (batch_size * num_heads)
    scaled_temp = ggml_div(ctx, Q_hat_sum, Q_sum);
    scaled_temp = ggml_sqrt(ctx, scaled_temp);
    scaled_temp = ggml_mul(ctx, scaled_temp, temp);
    print_structure(scaled_temp, "temperature");

    // K_hat -- (batch_size * num_heads, sequence_length, k1)
    K_hat = ggml_transpose(ctx, K);
    K_hat = ggml_cont(ctx, K_hat);
    K_hat = ggml_get_rows(ctx, K_hat, i1);
    K_hat = ggml_transpose(ctx, K_hat);
    K_hat = ggml_cont(ctx, K_hat);
    print_structure(K_hat, "K_hat");

    // s_hat -- (batch_size * num_heads, 1, sequence_length)
    s_hat = ggml_mul_mat(ctx, K_hat, Q_hat);
    s_hat = ggml_div(ctx, s_hat, scaled_temp);
    s_hat = ggml_soft_max(ctx, s_hat);
    print_structure(s_hat, "s_hat");

    // ========================================================================
    // STEP 2 =================================================================
    // ========================================================================

    // i2 -- (batch_size * num_heads, 1, k2)
    i2 = ggml_top_k(ctx, s_hat, k2);
    i2 = ggml_cont(ctx, i2);
    i2 = ggml_reshape_2d(ctx, i2, i2->ne[0], i2->ne[2]);
    print_structure(i2, "i2");

    // K_top_k -- (batch_size * num_heads, k2, head_dim)
    K_top_k = ggml_get_rows(ctx, K, i2);
    print_structure(K_top_k, "K_top_k");

    // s_top_k -- (batch_size * num_heads, 1, k2)
    s_top_k = ggml_mul_mat(ctx, K_top_k, Q);
    s_top_k = ggml_div(ctx, s_top_k, temp);
    s_top_k = ggml_soft_max(ctx, s_top_k);
    print_structure(s_top_k, "s_top_k");

    // V_top_k -- (batch_size * num_heads, head_dim, k2)
    V_top_k = ggml_get_rows(ctx, V, i2);
    V_top_k = ggml_transpose(ctx, V_top_k);
    V_top_k = ggml_cont(ctx, V_top_k);
    print_structure(V_top_k, "V_top_k");

    // output -- (batch_size * num_heads, 1, head_dim)
    output = ggml_mul_mat(ctx, V_top_k, s_top_k);
    print_structure(output, "output");

    // ========================================================================
    // STEP 3 (reallocation) ==================================================
    // ========================================================================

    // alpha -- (batch_size * num_heads)
    alpha = ggml_transpose(ctx, s_hat);
    alpha = ggml_get_rows(ctx, alpha, i2);
    alpha = ggml_transpose(ctx, alpha);
    alpha = ggml_sum_rows(ctx, alpha);
    print_structure(alpha, "alpha");


    return output;
}


int main() {

    // Constants
    const bool RUN_SPARQ = true;
    const bool CHECK_IF_SPARQ_MATCHES_EXACT = false;
    const bool PRINT_OUTPUT = true;
    const int SEED = 42;

    const ggml_type DATA_TYPE = GGML_TYPE_F32;
    const long PARAMS_MEMORY = pow(2, 32);

    const int head_dim = 128;
    const int batch_size = 1;
    const int num_heads = 8;
    const int sequence_length = 128;
    const int k1 = CHECK_IF_SPARQ_MATCHES_EXACT ? head_dim : 64;
    const int k2 = CHECK_IF_SPARQ_MATCHES_EXACT ? sequence_length : 64;

    // Create GGML context
    ggml_init_params params {PARAMS_MEMORY, nullptr, false};
    ggml_context *ctx = ggml_init(params);

    // Create attention tensors
    ggml_tensor *temp = ggml_new_tensor_1d(ctx, DATA_TYPE, 1);
    ggml_tensor *Q = ggml_new_tensor_3d(ctx, DATA_TYPE, head_dim, 1, batch_size * num_heads);
    ggml_tensor *K = ggml_new_tensor_3d(ctx, DATA_TYPE, head_dim, sequence_length, batch_size * num_heads);
    ggml_tensor *V = ggml_new_tensor_3d(ctx, DATA_TYPE, head_dim, sequence_length, batch_size * num_heads);
    
    // Define and construct the computational graph
    ggml_tensor *output = RUN_SPARQ ? sparq_attn(ctx, Q, K, V, temp, k1, k2) : exact_attn(ctx, Q, K, V, temp);
    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    // Set values for tensors
    std::srand(SEED);
    ggml_set_f32(temp, sqrtf(float(Q->ne[0])));
    std::vector<ggml_tensor*> tensors = {Q, K, V};
    for (ggml_tensor *tensor : tensors) {
        for (int i = 0; i < ggml_nelements(tensor); i++) {
            ggml_set_f32_1d(tensor, i, ((std::rand() % 100 + 1) * 1.0 - 50.0) / 50.0);
        }
    }

    // Run the computational graph and potentially print the output tensor
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    if (PRINT_OUTPUT) {
        print_tensor(output);
        print_structure(output, "output");
    }    

    // Exit
    return 0;
}

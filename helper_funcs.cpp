// Copyright (c) 2024 Graphcore Ltd. All rights reserved.

#include "helper_funcs.h"
#include <cstdio>

void print_tensor_structure(const ggml_tensor *t, std::string name) {
    printf("\n%s:\n", name.c_str());
    printf(
        "    elements (ne): %lu x %lu x %lu x %lu\n",
        t->ne[0],
        t->ne[1],
        t->ne[2],
        t->ne[3]
    );
    printf(
        "       bytes (nb): %lu x %lu x %lu x %lu\n\n",
        t->nb[0],
        t->nb[1],
        t->nb[2],
        t->nb[3]
    );
}

void print_tensor_values(const ggml_tensor *t) {
    for (int i3 = 0; i3 < t->ne[3]; i3++) {
        for (int i2 = 0; i2 < t->ne[2]; i2++) {
            for (int i1 = 0; i1 < t->ne[1]; i1++) {
                for (int i0 = 0; i0 < 100; i0++) {
                    printf("%f ", ggml_get_f32_nd(t, i0, i1, i2, i3));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_contiguity(const ggml_tensor *t, std::string name) {
    printf("%s is ", name.c_str());
    printf(ggml_is_contiguous(t) ? "contiguous\n" : "not contiguous\n");
}

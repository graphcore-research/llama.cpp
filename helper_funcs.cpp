// Copyright (c) 2024 Graphcore Ltd. All rights reserved.

#include "helper_funcs.h"
#include <cstdio>

void print_tensor_structure(const ggml_tensor *t, std::string name) {
    printf("\n%s:\n", name.c_str());
    printf(
        "    elements (ne): %lld x %lld x %lld x %lld\n",
        (long long)t->ne[0],
        (long long)t->ne[1],
        (long long)t->ne[2],
        (long long)t->ne[3]
    );
    printf(
        "       bytes (nb): %lu x %lu x %lu x %lu\n\n",
        t->nb[0],
        t->nb[1],
        t->nb[2],
        t->nb[3]
    );
}

void print_contiguity(const ggml_tensor *t, std::string name) {
    printf("%s is ", name.c_str());
    printf(ggml_is_contiguous(t) ? "contiguous\n" : "not contiguous\n");
}

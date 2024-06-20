#pragma once

#include <string>
#include "ggml.h"

void print_tensor_structure(const ggml_tensor *t, std::string name);
void print_tensor_values(const ggml_tensor *t);

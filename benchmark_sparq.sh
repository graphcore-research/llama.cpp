# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

set -e
set -o xtrace

NTHREAD="32"
# PRECISION="-ctk f32 -ctv f32"

FLAGS_BASE="--model models/llama-2-7b.Q8_0.gguf -b 1 -ub 1 -n 0 -p 8192 -t ${NTHREAD}"
FLAGS_SPARQ="--sparq -k1 16 -k2 1024"
FLAGS_BENCHMARK="${FLAGS_BASE} ${PRECISION} -r 10"
FLAGS_PROFILE="${FLAGS_BASE} ${PRECISION} -r 30"

mkdir -p timing-benchmarks

if [[ $1 == "--profile" ]] ; then
    echo "# Profiling..."

    make clean
    make LLAMA_FAST=1 LLAMA_GPROF=1 -j llama-bench

    rm gmon.out || true
    ./llama-bench ${FLAGS_PROFILE}
    gprof llama-bench > timing-benchmarks/profile_dense.txt

    rm gmon.out || true
    ./llama-bench ${FLAGS_PROFILE} ${FLAGS_SPARQ}
    gprof llama-bench > timing-benchmarks/profile_sparq.txt

else
    echo "# Benchmarking..."

    make clean
    make LLAMA_FAST=1 -j llama-bench

    ./llama-bench ${FLAGS_BENCHMARK}
    ./llama-bench ${FLAGS_BENCHMARK} --sparq -k1 0 -k2 0
    ./llama-bench ${FLAGS_BENCHMARK} --sparq -k1 0 -k2 0 --sparq-default-layout
    ./llama-bench ${FLAGS_BENCHMARK} ${FLAGS_SPARQ}
    ./llama-bench ${FLAGS_BENCHMARK} ${FLAGS_SPARQ} --sparq-default-layout
fi

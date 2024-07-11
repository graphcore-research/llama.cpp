set -e
set -o xtrace

FLAGS_BASE="--model models/llama-2-7b.Q8_0.gguf -b 1 -ub 1 -n 0 -ctk f32 -ctv f32 -p 8192 -t 16"
FLAGS_BENCHMARK="${FLAGS_BASE} -r 10"
FLAGS_PROFILE="${FLAGS_BASE} -r 20"
FLAGS_SPARQ="--sparq -k1 16 -k2 1024"

mkdir -p timing-benchmarks

if [[ $1 == "--profile" ]] ; then
    echo "# Profiling..."

    make clean
    rm gmon.out || true
    make LLAMA_GPROF=1 -j llama-bench

    ./llama-bench ${FLAGS_PROFILE} ${FLAGS_SPARQ}
    gprof llama-bench > timing-benchmarks/profile.txt
else
    echo "# Benchmarking..."

    make clean
    make -j llama-bench

    ./llama-bench ${FLAGS_BENCHMARK}
    ./llama-bench ${FLAGS_BENCHMARK} --sparq -k1 0 -k2 0
    ./llama-bench ${FLAGS_BENCHMARK} ${FLAGS_SPARQ}
    ./llama-bench ${FLAGS_BENCHMARK} ${FLAGS_SPARQ} --sparq-default-layout
fi

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

// Suggested first steps:
//  - Check strides
//  - Check flags (-ffast-math)
//  - How to half precision? (Look at what existing GGML kernels do.)

// void sparq(const float* q, const float* k, const float* v,
//     unsigned batch, unsigned sequence_length, unsigned head_dim,
//     unsigned k1, unsigned k2,
//     float* result) {
// }

using dtype = float;

// __attribute__((noinline))
dtype run_TEST(const dtype* begin, const dtype* end) {
    return std::accumulate(begin, end, dtype(0));
}

// Compile & run:
//    g++ -O3 -march=native --fast-math -Wall -Wextra -Werror test.cpp -o test && ./test
//
//   Dell PowerEdge R6525? "Up to 3200MT/s"
//     4800 MT/s = 38.4 GB/s  (DDR5)
//     3200 MT/s = 25.6 GB/s  (this?)
//     2666 MT/s = 21.3 GB/s
//
// Disassemble:
//    g++ -S -O3 -march=native --fast-math -Wall -Wextra -Werror test.cpp -o test.s
//
int main() {
    using clock = std::chrono::system_clock;

    std::vector<dtype> data(1ull << 31);
    std::iota(data.begin(), data.end(), 0);
    auto t0 = clock::now();
    auto result = run_TEST(data.data(), data.data() + data.size());
    auto duration = clock::now() - t0;
    auto size = data.size() * sizeof(data.front());

    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    std::cerr << "  Size: " << size / 1e9 << " GB"                //
              << "\nResult: " << result                           //
              << "\n    In: " << elapsed << " s"                  //
              << "\n    At: " << size / elapsed / 1e9 << " GB/s"  //
              << std::endl;
    return 0;
}

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {
// See MLIR LLVM IR Target C API reference
template<typename T, size_t N>
struct MemRefDescriptor {
    T *allocated;
    T *aligned;
    intptr_t offset;
    intptr_t sizes[N];
    intptr_t strides[N];
};

// tensor<1x28x28x1xf32>
using InputMemref = MemRefDescriptor<float, 4>;
// tensor<1x10xf32>
using OutputMemref = MemRefDescriptor<float, 2>;

} // namespace

extern "C" void _mlir_ciface_main(InputMemref *input, OutputMemref *output);

// ensure nothing is optimized out
volatile float out_buf[1][10];

int main(int argc, char **argv) {
    float in_buf[1][28][28][1]; // using unint garbage as testing input :)

    auto const mix_input = [&in_buf] {
        for (size_t r = 1; r < 28; ++r)
            for (size_t c = 1; c < 28; ++c) {
                in_buf[0][r-1][c-1][0] += in_buf[0][r][c][0];
            }
    };

    InputMemref in{
        (float*)in_buf,
        (float*)in_buf,
        0,
        {1, 28, 28, 1},
        {28 * 28, 28, 1, 1}
    };

    OutputMemref out{
        (float*)out_buf,
        (float*)out_buf,
        0,
        {1, 10},
        {10, 1}
    };

    // Run 10 inferences to have something measurable
    for (size_t i = 0; i < 10; ++i) {
        mix_input();
        _mlir_ciface_main(&in, &out);
    }
}

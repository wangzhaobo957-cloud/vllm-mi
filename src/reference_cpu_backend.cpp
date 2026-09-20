// 本文件实现便于逐项检查数值的标量 CPU 基础算子。
#include "mini_infer/reference_cpu_backend.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace mini_infer {

// 使用直接三重循环计算二维矩阵乘法。
Tensor ReferenceCpuBackend::MatMul(
    const Tensor& left,
    const Tensor& right) const {
    if (left.Rank() != 2 || right.Rank() != 2) {
        throw std::invalid_argument("MatMul expects two rank-2 tensors");
    }

    const std::size_t rows = left.Shape()[0];
    const std::size_t inner = left.Shape()[1];
    const std::size_t right_inner = right.Shape()[0];
    const std::size_t columns = right.Shape()[1];
    if (inner != right_inner) {
        throw std::invalid_argument("MatMul inner dimensions do not match");
    }

    Tensor output({rows, columns});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            float sum = 0.0F;
            for (std::size_t index = 0; index < inner; ++index) {
                sum += left[row * inner + index] *
                       right[index * columns + column];
            }
            output[row * columns + column] = sum;
        }
    }
    return output;
}

// 使用直接标量循环计算最后一维的 RMSNorm。
Tensor ReferenceCpuBackend::RmsNorm(
    const Tensor& input,
    const Tensor& weight,
    float epsilon) const {
    if (input.Rank() == 0 || weight.Rank() != 1) {
        throw std::invalid_argument(
            "RmsNorm expects a non-scalar input and rank-1 weight");
    }
    if (epsilon <= 0.0F) {
        throw std::invalid_argument("RmsNorm epsilon must be greater than zero");
    }

    const std::size_t width = input.Shape().back();
    if (weight.Size() != width) {
        throw std::invalid_argument(
            "RmsNorm weight must match the input's final dimension");
    }

    Tensor output(input.Shape());
    const std::size_t rows = input.Size() / width;
    for (std::size_t row = 0; row < rows; ++row) {
        double square_sum = 0.0;
        for (std::size_t column = 0; column < width; ++column) {
            const float value = input[row * width + column];
            square_sum += static_cast<double>(value) * value;
        }

        const float inverse_rms = 1.0F / std::sqrt(
            static_cast<float>(square_sum / width) + epsilon);
        for (std::size_t column = 0; column < width; ++column) {
            const std::size_t index = row * width + column;
            output[index] = input[index] * inverse_rms * weight[column];
        }
    }
    return output;
}

}  // namespace mini_infer

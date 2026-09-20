// 本文件声明用于数值校验和学习的标量 CPU 算子集合。
#pragma once

#include "mini_infer/tensor.h"

namespace mini_infer {

// 提供便于逐项检查数值的标量 CPU 基础算子。
class ReferenceCpuBackend
{
public:
    // 使用直接三重循环计算二维矩阵乘法。
    [[nodiscard]] Tensor MatMul(const Tensor& left, const Tensor& right) const;

    // 使用直接标量循环计算最后一维的 RMSNorm。
    [[nodiscard]] Tensor RmsNorm(const Tensor& input, const Tensor& weight,
                                 float epsilon) const;
};

}  // namespace mini_infer

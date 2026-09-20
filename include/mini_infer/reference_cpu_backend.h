// 本文件声明用于数值校验和学习的标量 CPU 算子后端。
#pragma once

#include "mini_infer/backend.h"

namespace mini_infer {

class ReferenceCpuBackend final : public Backend {
public:
    // 使用直接三重循环计算二维矩阵乘法。
    [[nodiscard]] Tensor MatMul(const Tensor& left,
                                const Tensor& right) const override;

    // 使用直接标量循环计算最后一维的 RMSNorm。
    [[nodiscard]] Tensor RmsNorm(const Tensor& input,
                                 const Tensor& weight,
                                 float epsilon) const override;
};

}  // namespace mini_infer

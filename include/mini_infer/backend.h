// 本文件定义推理算子后端必须提供的最小计算接口。
#pragma once

#include "mini_infer/tensor.h"

namespace mini_infer {

class Backend {
public:
    // 通过虚析构函数支持从接口安全释放具体后端。
    virtual ~Backend() = default;

    // 将两个二维张量执行矩阵乘法并返回结果。
    [[nodiscard]] virtual Tensor MatMul(const Tensor& left,
                                        const Tensor& right) const = 0;

    // 沿最后一个维度执行 RMSNorm。
    [[nodiscard]] virtual Tensor RmsNorm(const Tensor& input,
                                         const Tensor& weight,
                                         float epsilon) const = 0;
};

}  // namespace mini_infer

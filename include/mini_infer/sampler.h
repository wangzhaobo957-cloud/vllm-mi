// 本文件定义从 logits 中选择下一个 token 的采样接口。
#pragma once

#include "mini_infer/tensor.h"

namespace mini_infer {

class Sampler {
public:
    // 通过虚析构函数支持从接口安全释放具体采样器。
    virtual ~Sampler() = default;

    // 根据一维 logits 选择下一个 token ID。
    [[nodiscard]] virtual int Sample(const Tensor& logits) const = 0;
};

class GreedySampler final : public Sampler {
public:
    // 返回 logits 中数值最大位置对应的 token ID。
    [[nodiscard]] int Sample(const Tensor& logits) const override;
};

}  // namespace mini_infer

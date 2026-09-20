// 本文件定义从 logits 中选择下一个 token 的贪心采样器。
#pragma once

#include "mini_infer/tensor.h"

namespace mini_infer {

// 从一维 logits 中选择数值最大位置的确定性采样器。
class GreedySampler
{
public:
    // 返回 logits 中数值最大位置对应的 token ID。
    [[nodiscard]] int Sample(const Tensor& logits) const;
};

}  // namespace mini_infer

// 本文件定义从 logits 中选择下一个 token 的贪心采样器（声明与实现合并）。
#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>

#include "mini_infer/tensor.h"

namespace mini_infer {

// 从一维 logits 中选择数值最大位置的确定性采样器。
class GreedySampler
{
public:
    // 返回 logits 中数值最大位置对应的 token ID。
    [[nodiscard]] int Sample(const Tensor& logits) const
    {
        if (logits.Rank() != 1)
        {
            throw std::invalid_argument("GreedySampler expects rank-1 logits");
        }
        if (logits.Size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("vocabulary size exceeds token ID range");
        }

        const auto values = logits.Values();
        const auto maximum = std::max_element(values.begin(), values.end());
        return static_cast<int>(std::distance(values.begin(), maximum));
    }
};

}  // namespace mini_infer

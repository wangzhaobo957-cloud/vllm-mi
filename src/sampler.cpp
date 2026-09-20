// 本文件实现确定性的贪心 token 采样策略。
#include "mini_infer/sampler.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mini_infer {

// 返回 logits 中数值最大位置对应的 token ID。
int GreedySampler::Sample(const Tensor& logits) const
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

}  // namespace mini_infer

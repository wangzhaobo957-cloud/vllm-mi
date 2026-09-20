// 本文件定义拥有连续 FP32 内存的基础张量类型（声明与实现合并）。
#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_infer {

class Tensor
{
public:
    // 创建指定形状并使用同一个数值初始化的连续张量。
    explicit Tensor(std::vector<std::size_t> shape, float initial_value = 0.0F)
        : shape_(std::move(shape)),
          values_(ComputeElementCount(shape_), initial_value)
    {
    }

    // 返回张量的维度信息。
    [[nodiscard]] const std::vector<std::size_t>& Shape() const noexcept
    {
        return shape_;
    }

    // 返回张量包含的维度数量。
    [[nodiscard]] std::size_t Rank() const noexcept
    {
        return shape_.size();
    }

    // 返回张量包含的元素总数。
    [[nodiscard]] std::size_t Size() const noexcept
    {
        return values_.size();
    }

    // 返回可写的连续元素视图。
    [[nodiscard]] std::span<float> Values() noexcept
    {
        return values_;
    }

    // 返回只读的连续元素视图。
    [[nodiscard]] std::span<const float> Values() const noexcept
    {
        return values_;
    }

    // 按连续内存下标访问可写元素。
    [[nodiscard]] float& operator[](std::size_t index)
    {
        return values_.at(index);
    }

    // 按连续内存下标访问只读元素。
    [[nodiscard]] const float& operator[](std::size_t index) const
    {
        return values_.at(index);
    }

private:
    // 根据形状计算元素总数并检查溢出。
    [[nodiscard]] static std::size_t ComputeElementCount(
        const std::vector<std::size_t>& shape)
    {
        std::size_t element_count = 1;
        for (const std::size_t dimension : shape)
        {
            if (dimension == 0)
            {
                throw std::invalid_argument(
                    "tensor dimensions must be greater than zero");
            }
            if (element_count >
                std::numeric_limits<std::size_t>::max() / dimension)
            {
                throw std::overflow_error("tensor element count overflow");
            }
            element_count *= dimension;
        }
        return element_count;
    }

    std::vector<std::size_t> shape_;
    std::vector<float> values_;
};

}  // namespace mini_infer

// 本文件实现连续 FP32 张量的内存分配、形状查询和边界检查。
#include "mini_infer/tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace mini_infer {

// 创建指定形状并使用同一个数值初始化的连续张量。
Tensor::Tensor(std::vector<std::size_t> shape, float initial_value)
    : shape_(std::move(shape)),
      values_(ComputeElementCount(shape_), initial_value)
{
}

// 返回张量的维度信息。
const std::vector<std::size_t>& Tensor::Shape() const noexcept
{
    return shape_;
}

// 返回张量包含的维度数量。
std::size_t Tensor::Rank() const noexcept
{
    return shape_.size();
}

// 返回张量包含的元素总数。
std::size_t Tensor::Size() const noexcept
{
    return values_.size();
}

// 返回可写的连续元素视图。
std::span<float> Tensor::Values() noexcept
{
    return values_;
}

// 返回只读的连续元素视图。
std::span<const float> Tensor::Values() const noexcept
{
    return values_;
}

// 按连续内存下标访问可写元素。
float& Tensor::operator[](std::size_t index)
{
    return values_.at(index);
}

// 按连续内存下标访问只读元素。
const float& Tensor::operator[](std::size_t index) const
{
    return values_.at(index);
}

// 根据形状计算元素总数并检查溢出。
std::size_t Tensor::ComputeElementCount(const std::vector<std::size_t>& shape)
{
    std::size_t element_count = 1;
    for (const std::size_t dimension : shape)
    {
        if (dimension == 0)
        {
            throw std::invalid_argument(
                "tensor dimensions must be greater than zero");
        }
        if (element_count > std::numeric_limits<std::size_t>::max() / dimension)
        {
            throw std::overflow_error("tensor element count overflow");
        }
        element_count *= dimension;
    }
    return element_count;
}

}  // namespace mini_infer

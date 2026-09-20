// 本文件定义拥有连续 FP32 内存的基础张量类型。
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace mini_infer {

class Tensor {
public:
    // 创建指定形状并使用同一个数值初始化的连续张量。
    explicit Tensor(std::vector<std::size_t> shape, float initial_value = 0.0F);

    // 返回张量的维度信息。
    [[nodiscard]] const std::vector<std::size_t>& Shape() const noexcept;

    // 返回张量包含的维度数量。
    [[nodiscard]] std::size_t Rank() const noexcept;

    // 返回张量包含的元素总数。
    [[nodiscard]] std::size_t Size() const noexcept;

    // 返回可写的连续元素视图。
    [[nodiscard]] std::span<float> Values() noexcept;

    // 返回只读的连续元素视图。
    [[nodiscard]] std::span<const float> Values() const noexcept;

    // 按连续内存下标访问可写元素。
    [[nodiscard]] float& operator[](std::size_t index);

    // 按连续内存下标访问只读元素。
    [[nodiscard]] const float& operator[](std::size_t index) const;

private:
    // 根据形状计算元素总数并检查溢出。
    [[nodiscard]] static std::size_t ComputeElementCount(
        const std::vector<std::size_t>& shape);

    std::vector<std::size_t> shape_;
    std::vector<float> values_;
};

}  // namespace mini_infer

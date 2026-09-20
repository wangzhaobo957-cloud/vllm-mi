// 本文件定义按层和 token 位置组织的连续 KV 缓存。
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "mini_infer/model_config.h"

namespace mini_infer {

class KVCache
{
public:
    // 根据模型配置预分配全部 Key 和 Value 存储空间。
    explicit KVCache(const ModelConfig& config);

    // 清空已使用长度并将缓存内容重置为零。
    void Clear() noexcept;

    // 写入指定层和 token 位置对应的 Key 与 Value。
    void Set(std::size_t layer, std::size_t position,
             std::span<const float> key, std::span<const float> value);

    // 返回指定层和 token 位置的只读 Key 视图。
    [[nodiscard]] std::span<const float> Key(std::size_t layer,
                                             std::size_t position) const;

    // 返回指定层和 token 位置的只读 Value 视图。
    [[nodiscard]] std::span<const float> Value(std::size_t layer,
                                               std::size_t position) const;

    // 返回当前已经写入过的最大序列长度。
    [[nodiscard]] std::size_t Length() const noexcept;

    // 返回每个 token 在单层缓存中的元素数量。
    [[nodiscard]] std::size_t Width() const noexcept;

private:
    // 检查层号和位置并返回对应连续内存的起始下标。
    [[nodiscard]] std::size_t Offset(std::size_t layer,
                                     std::size_t position) const;

    std::size_t num_layers_;
    std::size_t max_sequence_length_;
    std::size_t width_;
    std::size_t length_ = 0;
    std::vector<float> keys_;
    std::vector<float> values_;
};

}  // namespace mini_infer

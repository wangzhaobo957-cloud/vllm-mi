// 本文件实现 KV 缓存的预分配、读写、清空和边界检查。
#include "mini_infer/kv_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mini_infer {

// 根据模型配置预分配全部 Key 和 Value 存储空间。
KVCache::KVCache(const ModelConfig& config)
    : num_layers_(config.num_layers),
      max_sequence_length_(config.max_sequence_length), width_(config.KvWidth())
{
    if (num_layers_ >
        std::numeric_limits<std::size_t>::max() / max_sequence_length_) {
        throw std::overflow_error("KV cache dimensions overflow");
    }
    const std::size_t layer_positions = num_layers_ * max_sequence_length_;
    if (layer_positions > std::numeric_limits<std::size_t>::max() / width_) {
        throw std::overflow_error("KV cache dimensions overflow");
    }

    const std::size_t element_count = layer_positions * width_;
    keys_.resize(element_count);
    values_.resize(element_count);
}

// 清空已使用长度并将缓存内容重置为零。
void KVCache::Clear() noexcept
{
    std::fill(keys_.begin(), keys_.end(), 0.0F);
    std::fill(values_.begin(), values_.end(), 0.0F);
    length_ = 0;
}

// 写入指定层和 token 位置对应的 Key 与 Value。
void KVCache::Set(std::size_t layer,
                  std::size_t position,
                  std::span<const float> key,
                  std::span<const float> value)
{
    if (key.size() != width_ || value.size() != width_) {
        throw std::invalid_argument(
            "KV vector width does not match model config");
    }

    const std::size_t offset = Offset(layer, position);
    std::copy(key.begin(), key.end(), keys_.begin() + offset);
    std::copy(value.begin(), value.end(), values_.begin() + offset);
    length_ = std::max(length_, position + 1);
}

// 返回指定层和 token 位置的只读 Key 视图。
std::span<const float> KVCache::Key(std::size_t layer,
                                    std::size_t position) const
{
    const std::size_t offset = Offset(layer, position);
    return std::span<const float>(keys_).subspan(offset, width_);
}

// 返回指定层和 token 位置的只读 Value 视图。
std::span<const float> KVCache::Value(std::size_t layer,
                                      std::size_t position) const
{
    const std::size_t offset = Offset(layer, position);
    return std::span<const float>(values_).subspan(offset, width_);
}

// 返回当前已经写入过的最大序列长度。
std::size_t KVCache::Length() const noexcept
{
    return length_;
}

// 返回每个 token 在单层缓存中的元素数量。
std::size_t KVCache::Width() const noexcept
{
    return width_;
}

// 检查层号和位置并返回对应连续内存的起始下标。
std::size_t KVCache::Offset(std::size_t layer, std::size_t position) const
{
    if (layer >= num_layers_) {
        throw std::out_of_range("KV cache layer is out of range");
    }
    if (position >= max_sequence_length_) {
        throw std::out_of_range("KV cache position is out of range");
    }
    return (layer * max_sequence_length_ + position) * width_;
}

}  // namespace mini_infer

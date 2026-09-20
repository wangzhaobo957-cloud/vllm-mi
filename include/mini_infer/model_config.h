// 本文件定义 Decoder-only 模型的结构参数及其合法性检查。
#pragma once

#include <cstddef>

namespace mini_infer {

struct ModelConfig {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t intermediate_size = 0;
    std::size_t num_layers = 0;
    std::size_t num_attention_heads = 0;
    std::size_t num_kv_heads = 0;
    std::size_t max_sequence_length = 0;
    float rope_theta = 10000.0F;
    float rms_norm_epsilon = 1.0e-5F;

    // 检查模型维度和数值参数是否能够组成合法的 Decoder 模型。
    void Validate() const;

    // 返回每个注意力头包含的隐藏维度。
    [[nodiscard]] std::size_t HeadDimension() const;

    // 返回单个 token 在一层 KV 缓存中占用的元素数量。
    [[nodiscard]] std::size_t KvWidth() const;
};

}  // namespace mini_infer

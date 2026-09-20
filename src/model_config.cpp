// 本文件实现 Decoder-only 模型配置的合法性检查和派生维度计算。
#include "mini_infer/model_config.h"

#include <stdexcept>

namespace mini_infer {

// 检查模型维度和数值参数是否能够组成合法的 Decoder 模型。
void ModelConfig::Validate() const {
    if (vocab_size == 0 || hidden_size == 0 || intermediate_size == 0 ||
        num_layers == 0 || num_attention_heads == 0 || num_kv_heads == 0 ||
        max_sequence_length == 0) {
        throw std::invalid_argument("model dimensions must be greater than zero");
    }
    if (hidden_size % num_attention_heads != 0) {
        throw std::invalid_argument(
            "hidden_size must be divisible by num_attention_heads");
    }
    if (num_attention_heads % num_kv_heads != 0) {
        throw std::invalid_argument(
            "num_attention_heads must be divisible by num_kv_heads");
    }
    if (rope_theta <= 0.0F || rms_norm_epsilon <= 0.0F) {
        throw std::invalid_argument(
            "rope_theta and rms_norm_epsilon must be greater than zero");
    }
}

// 返回每个注意力头包含的隐藏维度。
std::size_t ModelConfig::HeadDimension() const {
    Validate();
    return hidden_size / num_attention_heads;
}

// 返回单个 token 在一层 KV 缓存中占用的元素数量。
std::size_t ModelConfig::KvWidth() const {
    return num_kv_heads * HeadDimension();
}

}  // namespace mini_infer

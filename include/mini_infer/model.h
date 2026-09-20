// 本文件定义 Decoder-only 模型在推理运行时中的最小前向接口。
#pragma once

#include <cstddef>

#include "mini_infer/kv_cache.h"
#include "mini_infer/model_config.h"
#include "mini_infer/tensor.h"

namespace mini_infer {

class DecoderModel {
public:
    // 通过虚析构函数支持从接口安全释放具体模型。
    virtual ~DecoderModel() = default;

    // 返回当前模型使用的静态结构配置。
    [[nodiscard]] virtual const ModelConfig& Config() const noexcept = 0;

    // 输入一个 token 并返回预测下一个 token 的一维 logits。
    [[nodiscard]] virtual Tensor Forward(
        int token_id,
        std::size_t position,
        KVCache& cache) const = 0;
};

}  // namespace mini_infer

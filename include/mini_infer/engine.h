// 本文件定义组织分词、模型前向、KV 缓存和采样的推理引擎。
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "mini_infer/kv_cache.h"
#include "mini_infer/model.h"
#include "mini_infer/sampler.h"
#include "mini_infer/tokenizer.h"

namespace mini_infer {

struct GenerationResult {
    std::vector<int> token_ids;
    std::string text;
};

class InferenceEngine {
public:
    // 绑定推理所需组件并按模型配置创建 KV 缓存。
    InferenceEngine(
        const Tokenizer& tokenizer,
        const DecoderModel& model,
        const Sampler& sampler);

    // 对输入执行 prefill 和逐 token decode，并返回新增内容。
    [[nodiscard]] GenerationResult Generate(
        std::string_view prompt,
        std::size_t max_new_tokens);

private:
    const Tokenizer& tokenizer_;
    const DecoderModel& model_;
    const Sampler& sampler_;
    KVCache cache_;
};

}  // namespace mini_infer

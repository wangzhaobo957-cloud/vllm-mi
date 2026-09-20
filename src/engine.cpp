// 本文件实现 prefill 与 decode 共用的逐 token 自回归推理主循环。
#include "mini_infer/engine.h"

#include <stdexcept>
#include <utility>

namespace mini_infer {
namespace {

// 检查模型输出是否是一份与词表大小匹配的一维 logits。
void ValidateLogits(const Tensor& logits, std::size_t vocab_size) {
    if (logits.Rank() != 1 || logits.Size() != vocab_size) {
        throw std::runtime_error(
            "model must return rank-1 logits matching vocab_size");
    }
}

// 检查 token ID 是否位于模型词表范围内。
void ValidateToken(int token_id, std::size_t vocab_size) {
    if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab_size) {
        throw std::runtime_error("token ID is outside the model vocabulary");
    }
}

}  // namespace

// 绑定推理所需组件并按模型配置创建 KV 缓存。
InferenceEngine::InferenceEngine(
    const Tokenizer& tokenizer,
    const DecoderModel& model,
    const Sampler& sampler)
    : tokenizer_(tokenizer),
      model_(model),
      sampler_(sampler),
      cache_(model.Config()) {}

// 对输入执行 prefill 和逐 token decode，并返回新增内容。
GenerationResult InferenceEngine::Generate(
    std::string_view prompt,
    std::size_t max_new_tokens) {
    GenerationResult result;
    if (max_new_tokens == 0) {
        return result;
    }

    const ModelConfig& config = model_.Config();
    std::vector<int> prompt_tokens = tokenizer_.Encode(prompt);
    if (prompt_tokens.empty()) {
        throw std::invalid_argument("prompt must produce at least one token");
    }
    if (prompt_tokens.size() > config.max_sequence_length ||
        max_new_tokens >
            config.max_sequence_length - prompt_tokens.size()) {
        throw std::length_error(
            "prompt and generated tokens exceed max_sequence_length");
    }

    cache_.Clear();
    for (const int token_id : prompt_tokens) {
        ValidateToken(token_id, config.vocab_size);
    }

    Tensor logits = model_.Forward(prompt_tokens[0], 0, cache_);
    ValidateLogits(logits, config.vocab_size);
    for (std::size_t position = 1; position < prompt_tokens.size(); ++position) {
        logits = model_.Forward(prompt_tokens[position], position, cache_);
        ValidateLogits(logits, config.vocab_size);
    }

    result.token_ids.reserve(max_new_tokens);
    for (std::size_t generated = 0; generated < max_new_tokens; ++generated) {
        const int next_token = sampler_.Sample(logits);
        ValidateToken(next_token, config.vocab_size);
        result.token_ids.push_back(next_token);
        result.text += tokenizer_.DecodeToken(next_token);

        if (generated + 1 < max_new_tokens) {
            const std::size_t position = prompt_tokens.size() + generated;
            logits = model_.Forward(next_token, position, cache_);
            ValidateLogits(logits, config.vocab_size);
        }
    }

    return result;
}

}  // namespace mini_infer

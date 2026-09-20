// 本文件提供一个不依赖真实权重的命令行程序，用于验证完整生成主循环。
#include <array>
#include <charconv>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "mini_infer/engine.h"

namespace {

class ByteTokenizer final : public mini_infer::Tokenizer {
public:
    // 将每个输入字节直接映射为一个 token ID。
    [[nodiscard]] std::vector<int> Encode(
        std::string_view text) const override {
        std::vector<int> tokens;
        tokens.reserve(text.size());
        for (const char character : text) {
            tokens.push_back(static_cast<unsigned char>(character));
        }
        return tokens;
    }

    // 将字节范围内的 token ID 还原为单字符字符串。
    [[nodiscard]] std::string DecodeToken(int token_id) const override {
        if (token_id < 0 || token_id > 255) {
            throw std::out_of_range("byte token is outside [0, 255]");
        }
        return std::string(1, static_cast<char>(token_id));
    }
};

class ToyModel final : public mini_infer::DecoderModel {
public:
    // 创建仅用于验证自回归流程的确定性玩具模型。
    ToyModel() : config_(CreateConfig()) {}

    // 返回玩具模型的固定结构配置。
    [[nodiscard]] const mini_infer::ModelConfig& Config()
        const noexcept override {
        return config_;
    }

    // 令字母 token 预测其后继字母，并写入一项可观察的 KV 数据。
    [[nodiscard]] mini_infer::Tensor Forward(
        int token_id,
        std::size_t position,
        mini_infer::KVCache& cache) const override {
        const std::array<float, 1> key = {
            static_cast<float>(token_id),
        };
        const std::array<float, 1> value = {
            static_cast<float>(position),
        };
        cache.Set(0, position, key, value);

        int next_token = 'a';
        if (token_id >= 'a' && token_id < 'z') {
            next_token = token_id + 1;
        } else if (token_id == 'z') {
            next_token = 'a';
        }

        mini_infer::Tensor logits({config_.vocab_size}, -1000.0F);
        logits[static_cast<std::size_t>(next_token)] = 1.0F;
        return logits;
    }

private:
    // 构造满足接口约束的最小模型配置。
    [[nodiscard]] static mini_infer::ModelConfig CreateConfig() {
        mini_infer::ModelConfig config;
        config.vocab_size = 256;
        config.hidden_size = 1;
        config.intermediate_size = 1;
        config.num_layers = 1;
        config.num_attention_heads = 1;
        config.num_kv_heads = 1;
        config.max_sequence_length = 256;
        config.Validate();
        return config;
    }

    mini_infer::ModelConfig config_;
};

// 将命令行中的生成长度解析为非负整数。
[[nodiscard]] std::size_t ParseCount(std::string_view text) {
    std::size_t value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || position != end) {
        throw std::invalid_argument("max_new_tokens must be an integer");
    }
    return value;
}

}  // namespace

// 解析参数并运行玩具模型的自回归生成流程。
int main(int argc, char** argv) {
    try {
        const std::string_view prompt = argc > 1 ? argv[1] : "a";
        const std::size_t max_new_tokens =
            argc > 2 ? ParseCount(argv[2]) : 8;

        ByteTokenizer tokenizer;
        ToyModel model;
        mini_infer::GreedySampler sampler;
        mini_infer::InferenceEngine engine(tokenizer, model, sampler);

        const mini_infer::GenerationResult result =
            engine.Generate(prompt, max_new_tokens);
        std::cout << result.text << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

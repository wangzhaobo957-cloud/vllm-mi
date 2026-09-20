// 本文件定义文本与 token ID 之间转换的分词器接口。
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace mini_infer {

class Tokenizer
{
public:
    // 通过虚析构函数支持从接口安全释放具体分词器。
    virtual ~Tokenizer() = default;

    // 将输入文本编码为 token ID 序列。
    [[nodiscard]] virtual std::vector<int> Encode(
        std::string_view text) const = 0;

    // 将单个 token ID 解码为可追加的文本片段。
    [[nodiscard]] virtual std::string DecodeToken(int token_id) const = 0;
};

}  // namespace mini_infer

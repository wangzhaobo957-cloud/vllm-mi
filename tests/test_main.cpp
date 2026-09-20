// 本文件提供无第三方依赖的基础测试，验证张量、算子、KV 缓存和采样器。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string_view>

#include "mini_infer/kv_cache.h"
#include "mini_infer/model_config.h"
#include "mini_infer/reference_cpu_backend.h"
#include "mini_infer/sampler.h"
#include "mini_infer/tensor.h"

namespace {

int failures = 0;

// 记录布尔条件不满足时的测试失败信息。
void Expect(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

// 在指定容差内比较两个浮点数。
void ExpectNear(
    float actual,
    float expected,
    float tolerance,
    std::string_view message) {
    Expect(std::fabs(actual - expected) <= tolerance, message);
}

// 构造供 KV 缓存测试使用的小型合法模型配置。
[[nodiscard]] mini_infer::ModelConfig MakeConfig() {
    mini_infer::ModelConfig config;
    config.vocab_size = 16;
    config.hidden_size = 4;
    config.intermediate_size = 8;
    config.num_layers = 2;
    config.num_attention_heads = 2;
    config.num_kv_heads = 1;
    config.max_sequence_length = 4;
    config.Validate();
    return config;
}

// 验证模型配置可以正确计算注意力头和 KV 维度。
void TestModelConfig() {
    const mini_infer::ModelConfig config = MakeConfig();
    Expect(config.HeadDimension() == 2, "head dimension should be 2");
    Expect(config.KvWidth() == 2, "KV width should be 2");
}

// 验证参考后端的二维矩阵乘法结果。
void TestMatMul() {
    mini_infer::Tensor left({2, 3});
    mini_infer::Tensor right({3, 2});
    const std::array<float, 6> left_values = {1, 2, 3, 4, 5, 6};
    const std::array<float, 6> right_values = {7, 8, 9, 10, 11, 12};
    std::copy(left_values.begin(), left_values.end(), left.Values().begin());
    std::copy(right_values.begin(), right_values.end(), right.Values().begin());

    const mini_infer::ReferenceCpuBackend backend;
    const mini_infer::Tensor output = backend.MatMul(left, right);
    const std::array<float, 4> expected = {58, 64, 139, 154};
    Expect(output.Shape() == std::vector<std::size_t>({2, 2}),
           "MatMul output shape should be [2, 2]");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        ExpectNear(output[index], expected[index], 1.0e-6F,
                   "MatMul output value should match");
    }
}

// 验证参考后端沿最后一维计算 RMSNorm。
void TestRmsNorm() {
    mini_infer::Tensor input({1, 2});
    mini_infer::Tensor weight({2}, 1.0F);
    input[0] = 3.0F;
    input[1] = 4.0F;

    const mini_infer::ReferenceCpuBackend backend;
    const mini_infer::Tensor output =
        backend.RmsNorm(input, weight, 1.0e-6F);
    ExpectNear(output[0], 0.848528F, 1.0e-5F,
               "RmsNorm first value should match");
    ExpectNear(output[1], 1.131371F, 1.0e-5F,
               "RmsNorm second value should match");
}

// 验证 KV 缓存可以按层和位置写入、读取并清空。
void TestKvCache() {
    mini_infer::KVCache cache(MakeConfig());
    const std::array<float, 2> key = {1.0F, 2.0F};
    const std::array<float, 2> value = {3.0F, 4.0F};
    cache.Set(1, 2, key, value);

    Expect(cache.Length() == 3, "KV cache length should include position 2");
    Expect(cache.Width() == 2, "KV cache width should be 2");
    Expect(cache.Key(1, 2)[1] == 2.0F, "KV key should round-trip");
    Expect(cache.Value(1, 2)[0] == 3.0F, "KV value should round-trip");

    cache.Clear();
    Expect(cache.Length() == 0, "KV cache length should reset to zero");
    Expect(cache.Key(1, 2)[0] == 0.0F, "KV cache data should reset to zero");
}

// 验证贪心采样器返回最大 logit 对应的位置。
void TestGreedySampler() {
    mini_infer::Tensor logits({4});
    logits[0] = -1.0F;
    logits[1] = 2.0F;
    logits[2] = 0.5F;
    logits[3] = 1.0F;

    const mini_infer::GreedySampler sampler;
    Expect(sampler.Sample(logits) == 1,
           "GreedySampler should select the largest logit");
}

}  // namespace

// 依次运行基础测试并通过进程退出码报告结果。
int main() {
    try {
        TestModelConfig();
        TestMatMul();
        TestRmsNorm();
        TestKvCache();
        TestGreedySampler();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "UNEXPECTED EXCEPTION: " << error.what() << '\n';
    }

    if (failures == 0) {
        std::cout << "all tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}

<!-- 本文件说明学习型 C++ 推理引擎的目标、模块边界和构建方式。 -->
# mini-infer

`mini-infer` 是一个用于学习 Decoder-only 大语言模型推理流程的 C++20 项目。
当前阶段只关注单请求、CPU、FP32 推理主线，不包含 CUDA、连续批处理、分页 KV
缓存和分布式执行。

## 当前结构

```text
include/mini_infer/   公共接口
src/                  核心实现
tools/                可运行示例
tests/                无第三方依赖的基础测试
```

核心模块包括：

- `Tensor`：拥有连续 FP32 数据的基础张量。
- `Backend`：算子后端接口。
- `ReferenceCpuBackend`：便于验证数值正确性的标量 CPU 实现。
- `ModelConfig`：Decoder-only 模型的结构参数。
- `KVCache`：按层、位置存储 Key/Value。
- `DecoderModel`：模型前向接口。
- `Tokenizer`：文本与 token 的转换接口。
- `Sampler`：从 logits 选择下一个 token。
- `InferenceEngine`：组织 prefill 和 decode 自回归循环。

## 构建

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

运行玩具模型，验证推理主循环：

```bash
./build/mini-infer-generate a 5
```

预期生成：

```text
bcdef
```

## 后续里程碑

1. 为基础算子增加与 Python/PyTorch 参考值对齐的测试。
2. 定义权重文件格式并实现模型加载器。
3. 实现 LLaMA 风格的 RoPE、Attention、MLP 和 TransformerLayer。
4. 跑通 TinyStories 小模型并逐层比较 logits。
5. 在结果不变的前提下加入 SIMD、BLAS、量化和分页 KV 缓存。

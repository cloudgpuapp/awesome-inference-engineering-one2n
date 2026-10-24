# Awesome Inference Engineering [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> Tools, papers, benchmarks, talks, and practical guides for building and running AI inference systems.

Explore LLM inference and serving, from model execution and GPU kernel optimization to distributed deployment, multimodal inference, and performance benchmarking. This list brings together open-source engines, research, benchmarks, and practical guides for engineers building and operating production inference systems.

<a id="contents"></a>

## Table of contents

[](#table-of-contents)

- [1. Foundations](#foundations)
  - [1.1 Learning roadmaps](#foundations-learning-roadmaps)
  - [1.2 Inference fundamentals](#foundations-fundamentals)
  - [1.3 Model architectures](#foundations-model-architectures)
  - [1.4 Hardware and accelerators](#foundations-hardware)
  - [1.5 GPU kernel programming](#foundations-gpu-kernel-programming)
- [2. Engines](#engines)
  - [2.1 LLM inference engines](#engines-llm-serving)
  - [2.2 Multimodal and general serving](#engines-multimodal-serving)
  - [2.3 Runtimes and compilers](#engines-runtimes-compilers)
- [3. Optimization](#optimization)
  - [3.1 Attention kernels](#optimization-attention-kernels)
  - [3.2 KV-cache management](#optimization-kv-cache)
  - [3.3 Batching and scheduling](#optimization-batching-scheduling)
  - [3.4 Speculative decoding](#optimization-speculative-decoding)
  - [3.5 Quantization and compression](#optimization-quantization)
  - [3.6 Constrained decoding and sampling](#optimization-constrained-decoding)
  - [3.7 Long-context inference](#optimization-long-context)
  - [3.8 Determinism and numerics](#optimization-determinism)
- [4. Production inference](#production)
  - [4.1 Distributed inference](#production-distributed-inference)
  - [4.2 Gateways and routing](#production-gateways-routing)
  - [4.3 Deployment and orchestration](#production-orchestration)
  - [4.4 Managed Kubernetes inference (EKS, GKE, AKS)](#production-managed-kubernetes)
  - [4.5 Observability and profiling](#production-observability)
  - [4.6 Performance benchmarking and model evaluation](#production-benchmarking)
  - [4.7 Capacity planning and economics](#production-capacity-planning)
  - [4.8 Reliability, security, and output safety](#production-reliability-security)
  - [4.9 Confidential inference and model provenance](#production-confidential-inference)
  - [4.10 Production case studies](#production-case-studies)
- [5. Modalities](#modalities)
  - [5.1 Audio and speech inference](#modalities-audio)
  - [5.2 Vision inference](#modalities-vision)
  - [5.3 Image and video diffusion inference](#modalities-diffusion)
  - [5.4 Embedding and reranking inference](#modalities-embeddings)
- [6. Learning](#learning)
  - [6.1 Research papers](#learning-papers)
  - [6.2 Talks and conferences](#learning-talks)
  - [6.3 Tutorials and guides](#learning-tutorials)
  - [6.4 Engine internals](#learning-engine-internals)
  - [6.5 Courses and books](#learning-courses)
- [7. Contributing](#contributing)
- [8. License](#license)

<a id="foundations"></a>

## 1. Foundations

<a id="foundations-learning-roadmaps"></a>

### 1.1 Learning roadmaps

Learn how a model computes, how a serving system runs it, and how to measure performance.

[← Back to index](#contents)

<a id="foundations-learning-roadmaps-introductory-code"></a>

#### Introductory code

- [nanoGPT](https://github.com/karpathy/nanoGPT) - A compact GPT implementation for inspecting transformer mechanics.
- [llm.c](https://github.com/karpathy/llm.c) - Transformer training and inference implementation with C and CUDA.
- [nanochat](https://github.com/karpathy/nanochat) - Small end-to-end language model stack useful for understanding runtime structure.

<a id="foundations-learning-roadmaps-books-and-notebooks"></a>

#### Books and notebooks

- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) - Guided implementation of LLM components.
- [Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) - Illustrated explanation of language-model architectures and applications.

<a id="foundations-learning-roadmaps-infrastructure-intuition"></a>

#### Infrastructure intuition

- [ai-infra-book](https://github.com/bojieli/ai-infra-book) - Quantitative treatment of AI systems and infrastructure.

<a id="foundations-fundamentals"></a>

### 1.2 Inference fundamentals

Why a model that fits in memory can still run slowly.

[← Back to index](#contents)

<a id="foundations-fundamentals-attention"></a>

#### Attention

- [flash-attention](https://github.com/Dao-AILab/flash-attention) - Practical implementation of IO-aware attention.

<a id="foundations-fundamentals-kv-cache"></a>

#### KV cache

- [vllm](https://github.com/vllm-project/vllm) - Serving engine popularizing paged KV-cache management.

<a id="foundations-fundamentals-prefill-and-decode"></a>

#### Prefill and decode

- [sglang](https://github.com/sgl-project/sglang) - Runtime with optimized scheduling and caching.

<a id="foundations-fundamentals-tokenization"></a>

#### Tokenization

- [tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenizers used in language-model workloads.

<a id="foundations-fundamentals-gpu-profiling"></a>

#### GPU profiling

- [cuda-samples](https://github.com/NVIDIA/cuda-samples) - Examples for understanding CUDA execution.

<a id="foundations-fundamentals-essential-reading"></a>

#### Essential reading

- [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic) - Back-of-the-envelope compute and memory math for transformer inference.
- [How Fast Can We Perform a Forward Pass?](https://bounded-regret.ghost.io/how-fast-can-we-perform-a-forward-pass/) - Theoretical and practical limits on forward-pass latency.
- [How LLM Inference Works](https://arpitbhayani.me/blogs/how-llm-inference-works) - Prompt-to-response walkthrough of tokenization, embedding and autoregressive generation.
- [The Q, K, V Matrices](https://arpitbhayani.me/blogs/qkv-matrices) - Builds the query, key and value matrices of attention from first principles.
- [KV Cache in LLM Inference](https://pub.towardsai.net/kv-cache-in-llm-inference-7b904a2a6982) - Why caching keys and values removes redundant computation during decode.
- [How Do MoE Models Compare to Dense Models in Inference?](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference) - Inference cost and decode dynamics of mixture-of-experts versus dense models.
- [Top 5 AI Model Optimization Techniques](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/) - Vendor overview of model-level optimizations for inference latency and cost.

<a id="foundations-model-architectures"></a>

### 1.3 Model architectures

Architecture determines execution graphs, cache requirements and parallelism options.

[← Back to index](#contents)

<a id="foundations-model-architectures-transformer-implementations"></a>

#### Transformer implementations

- [transformers](https://github.com/huggingface/transformers) - Reference implementations for a broad range of model architectures.

<a id="foundations-model-architectures-dense-and-moe"></a>

#### Dense and MoE

- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) - Technical implementation and materials for a MoE model family.

<a id="foundations-model-architectures-attention-variants"></a>

#### Attention variants

- [flash-attention](https://github.com/Dao-AILab/flash-attention) - Attention kernel implementations.
- [How Attention Got So Efficient: GQA, MLA, DSA](https://youtu.be/Y-o545eYjXM) - Video tracing attention from grouped-query to multi-latent and DeepSeek sparse attention.

<a id="foundations-model-architectures-efficient-transformers"></a>

#### Efficient transformers

- [xformers](https://github.com/facebookresearch/xformers) - Transformer building blocks and attention operators.

<a id="foundations-model-architectures-state-space-models"></a>

#### State-space models

- [mamba](https://github.com/state-spaces/mamba) - Selective state-space architecture and implementations.

<a id="foundations-model-architectures-linear-attention-and-hybrids"></a>

#### Linear attention and hybrids

- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) - Triton kernels for linear-attention and related sequence models.
- [DeltaNet Explained (Part II)](https://sustcsonglin.github.io/blog/2024/deltanet-2/) - How DeltaNet is parallelized across the sequence dimension.
- [Jet-Nemotron](https://hanlab.mit.edu/projects/jet-nemotron) - Hybrid full/linear attention models targeting long-context generation throughput.
- [Titans + MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) - Research architectures that add learned long-term memory to sequence models.

<a id="foundations-model-architectures-diffusion-language-models"></a>

#### Diffusion language models

- [SGLang dLLM RFC](https://github.com/sgl-project/sglang/issues/12766) - Design proposal for serving block-diffusion language models in SGLang.
- [TiDAR](https://arxiv.org/abs/2511.08923) - Hybrid that drafts tokens with diffusion and commits them autoregressively.

<a id="foundations-hardware"></a>

### 1.4 Hardware and accelerators

Choose hardware by memory capacity, bandwidth, interconnect and software compatibility.

[← Back to index](#contents)

<a id="foundations-hardware-nvidia"></a>

#### NVIDIA

- [cuda-samples](https://github.com/NVIDIA/cuda-samples) - CUDA programming examples.
- [cutlass](https://github.com/NVIDIA/cutlass) - High-performance GPU matrix operations.

<a id="foundations-hardware-amd"></a>

#### AMD

- [ROCm](https://rocm.docs.amd.com/en/latest/) - ROCm software platform for AMD accelerators.
- [aiter](https://github.com/ROCm/aiter) - Inference-focused GPU operators.

<a id="foundations-hardware-apple-silicon"></a>

#### Apple Silicon

- [mlx](https://github.com/ml-explore/mlx) - Array framework optimized for Apple Silicon.

<a id="foundations-hardware-collectives"></a>

#### Collectives

- [nccl](https://github.com/NVIDIA/nccl) - GPU collective communication library.

<a id="foundations-hardware-kubernetes-gpus"></a>

#### Kubernetes GPUs

- [gpu-operator](https://github.com/NVIDIA/gpu-operator) - Automated NVIDIA GPU software stack on Kubernetes.

<a id="foundations-hardware-accelerator-architecture"></a>

#### Accelerator architecture

- [Domain-Specific Architectures](https://fleetwood.dev/posts/domain-specific-architectures) - Design principles behind AI accelerators.
- [A Close Look at SRAM for Inference](https://www.viksnewsletter.com/p/a-close-look-at-sram-for-inference) - Trade-offs between SRAM-heavy and HBM-based inference designs.
- [The Chip Made for the AI Inference Era: The Google TPU](https://www.uncoveralpha.com/p/the-chip-made-for-the-ai-inference) - Technical and strategic overview of Google TPUs.
- [Inside the NVIDIA Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) - Vendor overview of the Rubin rack-scale platform.
- [Maia 200](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/) - Vendor announcement of an inference-focused accelerator with FP8/FP4 tensor cores.
- [TinyTPU](https://www.tinytpu.com/) - Build-a-TPU-from-scratch project explaining systolic-array design.
- [ECE298A-TPU](https://github.com/WilliamZhang20/ECE298A-TPU) - Student TPU-style accelerator design intended for tape-out.

<a id="foundations-hardware-hardware-model-co-design"></a>

#### Hardware–model co-design

- [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489) - Shows that hardware-aligned model shapes improve runtime without hurting accuracy.
- [Analog In-Memory Computing Attention](https://www.nature.com/articles/s43588-025-00854-1) - Research on in-memory computing hardware for the attention mechanism.

<a id="foundations-gpu-kernel-programming"></a>

### 1.5 GPU kernel programming

Learn the execution model well enough to read, write and debug inference kernels.

[← Back to index](#contents)

<a id="foundations-gpu-kernel-programming-execution-model-and-architecture"></a>

#### Execution model and architecture

- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) - How GPU chips and their networks work, and what that means for LLMs.
- [Intro to GPUs for the Researcher](https://hackbot.dad/writing/intro-to-gpus/) - Practical GPU mental model aimed at raising model FLOPS utilization.
- [Blocks, Threads, and Kernels: A Deeper Dive](https://vanshnawander.github.io/vansh/posts/blocks-threads-kernels.html) - Thread hierarchy, block organization and kernel launch mechanics.
- [GPU Architecture Deep Dive: From HBM to Tensor Cores](https://www.youtube.com/watch?v=5UWphJWdAHY) - Video tour of the GPU memory hierarchy and tensor cores.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) - Official CUDA programming model reference.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) - Official reference for NVIDIA's virtual instruction set.

<a id="foundations-gpu-kernel-programming-matmul-and-gemm"></a>

#### Matmul and GEMM

- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) - Step-by-step path from a naive kernel to near-cuBLAS performance.
- [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - Worklog of Hopper-specific matmul optimizations.
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) - From PTX/SASS to warp tiling and asynchronous tensor-core pipelines.
- [CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) - High-performance GEMM with CUTLASS and Hopper WGMMA instructions.
- [B200 Blockscaled GEMM: The Setup](https://veitner.bearblog.dev/b200-blockscaled-gemm-the-setup/) - Layouts and MMA setup for block-scaled GEMM on Blackwell.
- [fast.cu](https://github.com/pranjalssh/fast.cu) - Performance-tuned CUDA kernels written from scratch as reference implementations.

<a id="foundations-gpu-kernel-programming-kernel-techniques"></a>

#### Kernel techniques

- [Unweaving Warp Specialization](https://rohany.github.io/blog/warp-specialization/) - Assigning producer and consumer roles to warps for higher throughput.
- [Flash Attention from Scratch Part 4: Bank Conflicts & Swizzling](https://lubits.ch/flash/Part-4) - Shared-memory bank conflicts and swizzling in an attention kernel.
- [Triton Flash Attention Kernel Walkthrough](https://nathanchen.me/public/Triton-Flash-Attention-Kernel-Walkthrough.html) - Line-by-line reading of a Triton attention kernel.
- [Reduction (Sum) Series: Part 1](https://kathsucurry.github.io/cuda/2025/10/14/reduction_sum_part1.html) - Introduction to parallel reduction in CUDA.
- [Reduction (Sum) Series: Part 2](https://kathsucurry.github.io/cuda/2025/10/27/reduction_sum_part2.html) - Implementing and optimizing reduction kernels step by step.
- [Effective Transpose on Hopper GPU](https://github.com/simveit/effective_transpose) - Optimized matrix transpose for Hopper.
- [Blackwell Pipelining with CuTeDSL](https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/) - Overlapping work with CuTeDSL asynchronous pipelines on Blackwell.
- [Categorical Foundations for CuTe Layouts](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) - The mathematics behind CuTe's layout algebra.
- [Demystifying Numeric Conversions in CuTeDSL](https://veitner.bearblog.dev/demystifying-numeric-conversions-in-cutedsl/) - Type-conversion primitives in CuTeDSL.
- [FlexAttention in FlashAttention CuTe DSL](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/) - Writing attention variants such as causal and sliding-window masks.
- [Blackwell NVFP4 Kernel Hackathon Journey](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html) - Practical notes on tuning NVFP4 kernels for Blackwell.

<a id="foundations-gpu-kernel-programming-debugging"></a>

#### Debugging

- [Tracing Hanging GPU Kernels to Source](https://blog.vllm.ai/2025/12/03/improved-cuda-debugging.html) - CUDA core dumps and tooling for locating hung kernels in vLLM.
- [AMD GPU Debugging](https://thegeeko.me/blog/amd-gpu-debugging/) - Debugging GPU programs with ROCm tools.

<a id="foundations-gpu-kernel-programming-beyond-cuda-c"></a>

#### Beyond CUDA C++

- [Rust GPU](https://rust-gpu.github.io/) - Toolchain for writing GPU shaders and compute kernels in Rust.
- [rust-cuda](https://github.com/Rust-GPU/rust-cuda) - Libraries and tools for writing CUDA kernels in Rust.
- [Autocomp Trainium Attention](https://charleshong3.github.io/blog/autocomp_trainium_attention.html) - Writing an attention kernel for AWS Trainium.

<a id="foundations-gpu-kernel-programming-ai-assisted-kernel-development"></a>

#### AI-assisted kernel development

- [AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) - Benchmark environment for coding agents writing GPU kernels.
- [Agent-Assisted Kernel Optimization](https://www.wafer.ai/blog/topk-sigmoid-optimization) - Case study of an agent tuning a kernel on AMD MI300X down to assembly.
- [Automating Algorithm Discovery: Kernel Generation](https://adrs-ucb.notion.site/datadog) - Case study on automated search for GPU kernel algorithms.
- [FlashInfer-Bench paper](https://arxiv.org/abs/2601.00227) - Closed loop connecting generated kernels, benchmarks and serving-engine integration.

<a id="engines"></a>

## 2. Engines

<a id="engines-llm-serving"></a>

### 2.1 LLM inference engines

General-purpose and specialized runtimes for serving language models.

[← Back to index](#contents)

<a id="engines-llm-serving-server-oriented"></a>

#### Server-oriented

- [vllm](https://github.com/vllm-project/vllm) - Serving engine with continuous batching and KV-cache management.
- [sglang](https://github.com/sgl-project/sglang) - High-performance LLM and multimodal serving runtime.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA-optimized language-model inference.
- [lmdeploy](https://github.com/InternLM/lmdeploy) - Deployment toolkit with inference backends.

<a id="engines-llm-serving-local-and-edge"></a>

#### Local and edge

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Portable C/C++ model inference and quantized formats.
- [ollama](https://github.com/ollama/ollama) - Local model distribution and runner.

<a id="engines-llm-serving-apple-silicon"></a>

#### Apple Silicon

- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM inference and fine-tuning on Apple Silicon.

<a id="engines-llm-serving-rust"></a>

#### Rust

- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - Rust inference runtime for language models.

<a id="engines-llm-serving-text-inference"></a>

#### Text inference

- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Fast inference for transformer sequence models.

<a id="engines-llm-serving-plugins-and-hardware-backends"></a>

#### Plugins and hardware backends

- [vLLM plugin system](https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html) - Extending vLLM through plugins instead of maintaining a fork.
- [Awesome vLLM Plugins](https://github.com/BudEcosystem/Awesome-vLLM-plugins) - Curated list of community vLLM plugins.
- [ArcticInference](https://github.com/snowflakedb/ArcticInference) - Snowflake's vLLM plugin with latency and throughput optimizations.
- [vllm-metal](https://github.com/vllm-project/vllm-metal) - Community plugin for running vLLM on Apple Silicon GPUs.
- [vLLM Gaudi](https://vllm-gaudi.readthedocs.io/) - Documentation for running vLLM on Intel Gaudi accelerators.

<a id="engines-llm-serving-consumer-gpu-and-offload"></a>

#### Consumer GPU and offload

- [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) - Consumer-GPU engine exploiting activation sparsity across GPU and CPU.
- [AirLLM](https://github.com/lyogavin/airllm) - Layer-by-layer loading that runs large models on small GPUs at the cost of latency.

<a id="engines-multimodal-serving"></a>

### 2.2 Multimodal and general serving

Serving systems for mixed model types and modalities.

[← Back to index](#contents)

<a id="engines-multimodal-serving-general-serving"></a>

#### General serving

- [server](https://github.com/triton-inference-server/server) - Server supporting multiple model backends.
- [onnxruntime](https://github.com/microsoft/onnxruntime) - Portable inference runtime for ONNX models.
- [TensorRT](https://github.com/NVIDIA/TensorRT) - Optimizing compiler/runtime for NVIDIA inference.

<a id="engines-multimodal-serving-multimodal"></a>

#### Multimodal

- [diffusers](https://github.com/huggingface/diffusers) - Diffusion inference pipelines and optimizations.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Node-based orchestration for image/video model workflows.

<a id="engines-multimodal-serving-text-and-embedding"></a>

#### Text and embedding

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - Embedding and reranker model toolkit.

<a id="engines-multimodal-serving-omni-modal-serving"></a>

#### Omni-modal serving

- [vllm-omni](https://github.com/vllm-project/vllm-omni) - vLLM extension for serving models that mix text, image, audio and video.
- [cornserve](https://github.com/cornserve-ai/cornserve) - Serving system that splits multimodal pipelines into independently scaled components.

<a id="engines-runtimes-compilers"></a>

### 2.3 Runtimes and compilers

Compiler stacks and execution backends that enable efficient inference.

[← Back to index](#contents)

<a id="engines-runtimes-compilers-gpu-programming"></a>

#### GPU programming

- [triton](https://github.com/triton-lang/triton) - Compiler and language for writing GPU kernels.

<a id="engines-runtimes-compilers-cuda-kernels"></a>

#### CUDA kernels

- [cutlass](https://github.com/NVIDIA/cutlass) - Templates for accelerated GEMM operations.

<a id="engines-runtimes-compilers-ml-compilers"></a>

#### ML compilers

- [tvm](https://github.com/apache/tvm) - Machine learning compiler framework.
- [iree](https://github.com/iree-org/iree) - Compiler and runtime for portable model deployment.

<a id="engines-runtimes-compilers-kernel-runtime"></a>

#### Kernel runtime

- [flashinfer](https://github.com/flashinfer-ai/flashinfer) - Inference attention and auxiliary GPU kernels.

<a id="engines-runtimes-compilers-cuda-alternatives"></a>

#### CUDA alternatives

- [ROCm](https://rocm.docs.amd.com/en/latest/) - AMD GPU software stack.

<a id="engines-runtimes-compilers-python-kernel-dsls"></a>

#### Python kernel DSLs

- [helion](https://github.com/pytorch/helion) - Python-embedded DSL from PyTorch for writing ML kernels with autotuning.
- [CUDA Tile](https://developer.nvidia.com/cuda/tile) - NVIDIA's tile-based programming model for tensor-core kernels.
- [cuda-tile](https://github.com/NVIDIA/cuda-tile) - MLIR-based CUDA Tile IR and compiler infrastructure.

<a id="engines-runtimes-compilers-pytorch-compilation"></a>

#### PyTorch compilation

- [AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) - Ahead-of-time compilation of PyTorch models for deployment without a Python runtime.
- [Learn by Doing: TorchInductor Reduction Kernels](https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-Reduction/) - How TorchInductor generates reduction kernels.

<a id="engines-runtimes-compilers-other-compilers"></a>

#### Other compilers

- [ISPC](https://ispc.github.io/index.html) - Intel's implicit-SPMD compiler for SIMD CPU and GPU code.
- [Inside VOLT](https://arxiv.org/abs/2511.13751) - Open-source compiler design for open GPU architectures with custom ISAs.

<a id="optimization"></a>

## 3. Optimization

<a id="optimization-attention-kernels"></a>

### 3.1 Attention kernels

Implementations for reducing memory traffic and improving attention execution.

[← Back to index](#contents)

<a id="optimization-attention-kernels-attention"></a>

#### Attention

- [flash-attention](https://github.com/Dao-AILab/flash-attention) - Memory-efficient attention kernels.
- [flashinfer](https://github.com/flashinfer-ai/flashinfer) - Inference-specialized attention primitives.
- [xformers](https://github.com/facebookresearch/xformers) - Efficient attention and transformer operations.

<a id="optimization-attention-kernels-gpu-math"></a>

#### GPU math

- [cutlass](https://github.com/NVIDIA/cutlass) - Composable GPU math operators.

<a id="optimization-attention-kernels-amd-kernels"></a>

#### AMD kernels

- [aiter](https://github.com/ROCm/aiter) - AMD inference operators.

<a id="optimization-attention-kernels-kernel-programming"></a>

#### Kernel programming

- [triton](https://github.com/triton-lang/triton) - GPU kernel language used for custom optimizations.

<a id="optimization-kv-cache"></a>

### 3.2 KV-cache management

How attention state is stored, reused, transferred, compressed, and released.

[← Back to index](#contents)

<a id="optimization-kv-cache-paged-cache"></a>

#### Paged cache

- [vllm](https://github.com/vllm-project/vllm) - Runtime with paged KV-cache abstractions.

<a id="optimization-kv-cache-prefix-caching"></a>

#### Prefix caching

- [sglang](https://github.com/sgl-project/sglang) - Runtime with prefix reuse and efficient serving.

<a id="optimization-kv-cache-cache-offload"></a>

#### Cache offload

- [LMCache](https://github.com/LMCache/LMCache) - KV-cache storage and reuse across tiers.

<a id="optimization-kv-cache-disaggregated-cache"></a>

#### Disaggregated cache

- [Mooncake](https://github.com/kvcache-ai/Mooncake) - Cache-centric infrastructure for disaggregated inference.

<a id="optimization-kv-cache-distributed-serving"></a>

#### Distributed serving

- [llm-d](https://github.com/llm-d/llm-d) - Infrastructure integrating cache-aware routing and serving.

<a id="optimization-kv-cache-offload-and-tiering"></a>

#### Offload and tiering

- [vLLM KV Offloading Connector](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html) - Offloading KV-cache to CPU DRAM and tuning host-to-device throughput.
- [PrisKV](https://aibrix.github.io/posts/2025-11-26-priskv-intro/) - Colocated, tiered KV-cache store from the AIBrix project.
- [How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/) - KV-cache offload and reuse in Dynamo.

<a id="optimization-kv-cache-cache-quantization"></a>

#### Cache quantization

- [Optimizing Inference with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) - FP4 KV-cache for long contexts and large batches.

<a id="optimization-kv-cache-prompt-caching"></a>

#### Prompt caching

- [Prompt Caching](https://ngrok.com/blog/prompt-caching/) - Practical explanation of how provider-side prompt caching works and when it pays off.

<a id="optimization-batching-scheduling"></a>

### 3.3 Batching and scheduling

Techniques for sharing accelerator cycles across uneven request workloads.

[← Back to index](#contents)

<a id="optimization-batching-scheduling-continuous-batching"></a>

#### Continuous batching

- [vllm](https://github.com/vllm-project/vllm) - Serving and request scheduler implementations.

<a id="optimization-batching-scheduling-scheduling"></a>

#### Scheduling

- [sglang](https://github.com/sgl-project/sglang) - Efficient request processing and caching.

<a id="optimization-batching-scheduling-production-patterns"></a>

#### Production patterns

- [production-stack](https://github.com/vllm-project/production-stack) - Deployment and autoscaling patterns for vLLM.

<a id="optimization-batching-scheduling-simulation"></a>

#### Simulation

- [vidur](https://github.com/microsoft/vidur) - LLM inference performance simulator.

<a id="optimization-speculative-decoding"></a>

### 3.4 Speculative decoding

Propose several tokens at once, then verify them to reduce decode time.

[← Back to index](#contents)

<a id="optimization-speculative-decoding-draft-tree"></a>

#### Draft tree

- [EAGLE](https://github.com/SafeAILab/EAGLE) - Speculative sampling family and associated code.

<a id="optimization-speculative-decoding-multiple-heads"></a>

#### Multiple heads

- [Medusa](https://github.com/FasterDecoding/Medusa) - Multiple decoding heads for candidate token generation.

<a id="optimization-speculative-decoding-engine-integration"></a>

#### Engine integration

- [vllm](https://github.com/vllm-project/vllm) - Serving implementation with decoding optimizations.
- [sglang](https://github.com/sgl-project/sglang) - Runtime supporting advanced decoding optimizations.

<a id="optimization-speculative-decoding-training-free-drafting"></a>

#### Training-free drafting

- [SuffixDecoding](https://suffix-decoding.github.io/) - CPU-side suffix trees for draft-model-free speculation on repetitive agentic workloads.

<a id="optimization-speculative-decoding-libraries-and-benchmarks"></a>

#### Libraries and benchmarks

- [speculators](https://github.com/vllm-project/speculators) - Library for building, evaluating and packaging speculative decoding models for vLLM.
- [Speculators: Production-Ready Speculative Decoding](https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding) - Why the speculators library standardizes draft-model formats.
- [Spec-Bench](https://github.com/hemingkx/Spec-Bench) - Unified benchmark for comparing speculative decoding methods.

<a id="optimization-quantization"></a>

### 3.5 Quantization and compression

Lower memory bandwidth and storage requirements while preserving acceptable quality.

[← Back to index](#contents)

<a id="optimization-quantization-awq"></a>

#### AWQ

- [llm-awq](https://github.com/mit-han-lab/llm-awq) - Activation-aware weight quantization.

<a id="optimization-quantization-gptq"></a>

#### GPTQ

- [gptq](https://github.com/IST-DASLab/gptq) - Post-training weight quantization research implementation.

<a id="optimization-quantization-gptq-historical"></a>

#### GPTQ historical

- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) - Formerly popular GPTQ implementation; review maintenance before adoption.

<a id="optimization-quantization-runtime-kernels"></a>

#### Runtime kernels

- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - Low-precision GPU operations.

<a id="optimization-quantization-compression"></a>

#### Compression

- [llm-compressor](https://github.com/vllm-project/llm-compressor) - LLM quantization and compression toolkit.

<a id="optimization-quantization-edge-formats"></a>

#### Edge formats

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Quantized formats and implementations for portable inference.

<a id="optimization-quantization-low-precision-formats"></a>

#### Low-precision formats

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Illustrated introduction to LLM quantization.
- [Post Training Quantization](https://liyuan24.github.io/writings/2026_01_06_post_training_quantization.html) - Detailed walkthrough of post-training weight quantization.
- [Survey of Quantization Formats](https://github.com/vipulSharma18/Survey-of-Quantization-Formats) - MXFP8 and NVFP4 formats with TorchAO and GemLite benchmarks on an RTX 4090.
- [INT vs FP](https://github.com/ChenMnZ/INT_vs_FP) - Framework comparing low-bit integer and floating-point formats.
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) - FP8 and FP4 transformer building blocks for Hopper, Ada and Blackwell.
- [torchao Quantized Models on Hugging Face Hub](https://pytorch.org/blog/torchao-quantized-models-and-quantization-recipes-now-available-on-huggingface-hub/) - PyTorch-native quantization recipes with pre-quantized checkpoints.
- [torchao float8](https://github.com/pytorch/ao/blob/main/torchao/float8/README.md) - Float8 workflows in torchao.

<a id="optimization-quantization-low-bit-toolkits"></a>

#### Low-bit toolkits

- [GPTQModel](https://github.com/ModelCloud/GPTQModel) - Maintained GPTQ toolkit with CUDA, ROCm, XPU and CPU backends.
- [AutoRound x LLM Compressor](https://blog.vllm.ai/2025/12/09/intel-autoround-llmc.html) - Intel AutoRound integrated with LLM Compressor for vLLM serving.
- [AutoRound Meets SGLang](https://lmsys.org/blog/2025-11-13-AutoRound/) - Serving AutoRound low-bit models in SGLang.

<a id="optimization-quantization-low-bit-kernels"></a>

#### Low-bit kernels

- [GemLite](https://dropbox.github.io/gemlite_blogpost/) - Fused low-bit GEMV kernels with bit-packing and dequantization.

<a id="optimization-quantization-extreme-compression-and-pruning"></a>

#### Extreme compression and pruning

- [BitNet](https://github.com/microsoft/BitNet) - Inference framework for 1-bit (ternary) LLMs.
- [LLM Pruning Collection](https://github.com/zlab-princeton/llm-pruning-collection) - Pruning implementations and evaluation scripts for LLMs.

<a id="optimization-constrained-decoding"></a>

### 3.6 Constrained decoding and sampling

Constrain token selection for structured output and tool calls while keeping throughput in view.

[← Back to index](#contents)

<a id="optimization-constrained-decoding-guided-decoding"></a>

#### Guided decoding

- [Guided Decoding Performance on vLLM and SGLang](https://blog.squeezebits.com/70642) - Benchmarks XGrammar and LLGuidance backends across both engines.
- [Combining Guided Decoding and Speculative Decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.md) - How TensorRT-LLM applies grammar constraints to speculative drafts.

<a id="optimization-constrained-decoding-tool-calling"></a>

#### Tool calling

- [Tool Calling in Inference](https://www.baseten.co/blog/tool-calling-in-inference/) - Why tool-call quality differs between providers and how to make it reliable.

<a id="optimization-constrained-decoding-token-efficient-formats"></a>

#### Token-efficient formats

- [TSON](https://github.com/zenoaihq/tson) - Experimental compact object notation that reduces tokens spent on structured output.

<a id="optimization-constrained-decoding-sampling-kernels"></a>

#### Sampling kernels

- [Sorting-Free GPU Kernels for LLM Sampling](https://flashinfer.ai/2025/03/10/sampling.html) - Top-k/top-p sampling on GPU without a full sort.

<a id="optimization-long-context"></a>

### 3.7 Long-context inference

Serve long prompts without letting prefill time or KV-cache growth dominate cost.

[← Back to index](#contents)

<a id="optimization-long-context-position-encoding"></a>

#### Position encoding

- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) - Explains RoPE, the position encoding most context-extension methods modify.
- [DroPE](https://pub.sakana.ai/DroPE/) - Extends context by dropping positional embeddings instead of long-context fine-tuning.

<a id="optimization-long-context-attention-at-length"></a>

#### Attention at length

- [Long Context Attention](https://nrehiew.github.io/blog/long_context/) - Compute and memory challenges of scaling attention to long sequences.

<a id="optimization-long-context-prefill-scaling"></a>

#### Prefill scaling

- [Optimizing Long-Context Prefill on Older GPU Nodes](https://moreh.io/blog/optimizing-long-context-prefill-on-multiple-older-generation-gpu-nodes-251226/) - Distributing long-context prefill across older-generation GPUs.

<a id="optimization-long-context-kv-compression"></a>

#### KV compression

- [PQCache](https://sky-light.eecs.berkeley.edu/#/blog/pqcache) - Product quantization of the KV-cache for long-context inference.

<a id="optimization-determinism"></a>

### 3.8 Determinism and numerics

Understand when and why identical requests produce different outputs.

[← Back to index](#contents)

<a id="optimization-determinism-batch-invariance"></a>

#### Batch invariance

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) - Traces nondeterminism to batch-dependent kernels and shows batch-invariant fixes.
- [Enabling Deterministic Inference for SGLang](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) - Integrating batch-invariant kernels into SGLang.

<a id="optimization-determinism-train-inference-consistency"></a>

#### Train–inference consistency

- [Bitwise Consistent On-Policy RL](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) - Matching vLLM inference numerics with TorchTitan training.
- [When Speed Kills Stability](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) - How training–inference numeric mismatch collapses RL runs.

<a id="optimization-determinism-numerics"></a>

#### Numerics

- [Why Stochastic Rounding is Essential](https://cloud.google.com/blog/topics/developers-practitioners/why-stochastic-rounding-is-essential-for-modern-generative-ai) - Stochastic rounding for quality in low-precision arithmetic.
- [Future Leakage in Block-Quantized Attention](https://matx.com/research/leaky_quantization) - How block quantization can leak future-token information into causal attention.

<a id="production"></a>

## 4. Production inference

<a id="production-distributed-inference"></a>

### 4.1 Distributed inference

Distributed inference across GPUs and nodes, including the cost of communication.

[← Back to index](#contents)

<a id="production-distributed-inference-distributed-inference"></a>

#### Distributed serving frameworks

- [llm-d](https://github.com/llm-d/llm-d) - Kubernetes-oriented distributed LLM inference.

<a id="production-distributed-inference-distributed-runtime"></a>

#### Distributed runtime

- [dynamo](https://github.com/ai-dynamo/dynamo) - Distributed inference serving components.

<a id="production-distributed-inference-kv-cache-transport"></a>

#### KV-cache transport

- [Mooncake](https://github.com/kvcache-ai/Mooncake) - Disaggregated KV-cache and data movement.

<a id="production-distributed-inference-expert-parallelism"></a>

#### Expert parallelism

- [DeepEP](https://github.com/deepseek-ai/DeepEP) - MoE communication implementation.

<a id="production-distributed-inference-collectives"></a>

#### Collectives

- [nccl](https://github.com/NVIDIA/nccl) - GPU collective communication.

<a id="production-distributed-inference-cache-tiers"></a>

#### Cache tiers

- [LMCache](https://github.com/LMCache/LMCache) - KV-cache lifecycle and reuse.

<a id="production-distributed-inference-disaggregation-in-practice"></a>

#### Disaggregation in practice

- [Disaggregated Inference: 18 Months Later](https://hao-ai-lab.github.io/blogs/distserve-retro/) - DistServe authors' retrospective on prefill/decode disaggregation adoption.
- [Disaggregated Inference at Scale with PyTorch and vLLM](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/) - Running separate prefill and decode pools with vLLM.
- [Disaggregated Serving in TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog05_Disaggregated_Serving_in_TensorRT-LLM.md) - TensorRT-LLM's disaggregated serving design.
- [llm-d Architecture](https://llm-d.ai/docs/architecture) - Component design and topology of llm-d.

<a id="production-distributed-inference-parallelism-strategies"></a>

#### Parallelism strategies

- [Visualizing Parallelism in Transformer](https://ailzhang.github.io/posts/distributed-compute-in-transformer/) - Diagrams of tensor, pipeline and data parallelism across devices.
- [The vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) - Combining TP, DP, PP and expert parallelism for MoE models in vLLM.
- [Shift Parallelism](https://arxiv.org/abs/2509.16495) - Switching between tensor and data parallelism to balance latency and throughput.
- [Learning to Love Mesh-Oriented Sharding](https://blog.ezyang.com/2025/12/learning-to-love-mesh-oriented-sharding/) - Mental model for device-mesh sharding.
- [Tensor Parallel in NanoVLLM](https://liyuan24.github.io/writings/2025_12_18_nanovllm_tensor_parallel_kernel_fusion.html) - Tensor parallelism and kernel fusion in a minimal engine.

<a id="production-distributed-inference-communication"></a>

#### Communication

- [uccl](https://github.com/uccl-project/uccl) - GPU communication library for collectives, P2P KV transfer and expert parallelism.
- [Low Latency Communication Kernels with NVSHMEM](https://www.youtube.com/live/rZBF2PuycLQ) - GPU MODE lecture on GPU-initiated communication.
- [RoCEv2 for Deep Learning](https://iodized-hawthorn-94a.notion.site/RoCEv2-26954c8e1f0a80b78bf1c6adc583e670) - Introduction to RDMA over Converged Ethernet for GPU clusters.

<a id="production-distributed-inference-moe-serving"></a>

#### MoE serving

- [vLLM Large Scale Serving with Wide-EP](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) - Wide expert parallelism for DeepSeek on H200 clusters.
- [LPLB](https://github.com/deepseek-ai/LPLB) - Early-stage linear-programming MoE expert load balancer.

<a id="production-distributed-inference-heterogeneous-accelerators"></a>

#### Heterogeneous accelerators

- [Distributed Inference on Heterogeneous Accelerators](https://moreh.io/blog/distributed-inference-on-heterogeneous-accelerators-including-gpus-rubin-cpx-and-ai-accelerators-250923/) - Splitting inference across mixed AMD and NVIDIA hardware.

<a id="production-gateways-routing"></a>

### 4.2 Gateways and routing

Route requests among models and backends while meeting budgets and SLOs.

[← Back to index](#contents)

<a id="production-gateways-routing-model-gateway"></a>

#### Model gateway

- [litellm](https://github.com/BerriAI/litellm) - Provider abstraction, routing, quotas and API gateway features.

<a id="production-gateways-routing-envoy-ecosystem"></a>

#### Envoy ecosystem

- [Agent Router (formerly Envoy AI Gateway)](https://github.com/theagentrouter/agent-router) - AI request routing using Envoy infrastructure.

<a id="production-gateways-routing-kubernetes"></a>

#### Kubernetes inference routing

- [gateway-api-inference-extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) - Inference-specific routing APIs for Kubernetes Gateway API.

<a id="production-gateways-routing-distributed-routing"></a>

#### Distributed routing

- [llm-d](https://github.com/llm-d/llm-d) - Cache and load-aware routing patterns.

<a id="production-gateways-routing-gateway"></a>

#### Open-source AI gateway

- [gateway](https://github.com/Portkey-AI/gateway) - Open-source AI gateway.

<a id="production-gateways-routing-semantic-and-model-routing"></a>

#### Semantic and model routing

- [semantic-router](https://github.com/vllm-project/semantic-router) - vLLM's mixture-of-models router that classifies requests and picks a model.
- [vLLM Semantic Router: Improving Efficiency in AI Reasoning](https://developers.redhat.com/articles/2025/09/11/vllm-semantic-router-improving-efficiency-ai-reasoning) - Cost-aware routing between reasoning and non-reasoning paths.
- [Signal-Decision Driven Architecture](https://blog.vllm.ai/2025/11/19/signal-decision.html) - How the semantic router evolved from classification to signal-based decisions.
- [LLM Routing](https://www.liuxunzhuo.com/llm-routing) - Overview of strategies for routing requests between models.

<a id="production-gateways-routing-cache-aware-routing"></a>

#### Cache-aware routing

- [vLLM router](https://github.com/vllm-project/router) - Lightweight router for large vLLM deployments.
- [Ray Serve Custom Request Routing](https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing) - Prefix-aware routing to cut TTFT in multi-turn chat.
- [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see) - Prefix-, load- and SLO-aware routing in llm-d.

<a id="production-gateways-routing-inference-proxies"></a>

#### Inference proxies

- [optillm](https://github.com/algorithmicsuperintelligence/optillm) - OpenAI-compatible proxy that applies inference-time reasoning techniques to improve answers.

<a id="production-orchestration"></a>

### 4.3 Deployment and orchestration

Deploy, update, and manage inference services.

[← Back to index](#contents)

<a id="production-orchestration-kubernetes-serving"></a>

#### Kubernetes serving

- [kserve](https://github.com/kserve/kserve) - Kubernetes model serving framework.

<a id="production-orchestration-ray-serving"></a>

#### Ray serving

- [ray](https://github.com/ray-project/ray) - Distributed Python runtime including Ray Serve.

<a id="production-orchestration-packaging"></a>

#### Packaging

- [BentoML](https://github.com/bentoml/BentoML) - Inference packaging and service deployment.

<a id="production-orchestration-production-vllm"></a>

#### Production vLLM

- [production-stack](https://github.com/vllm-project/production-stack) - Production-oriented manifests and components.

<a id="production-orchestration-multi-node-workloads"></a>

#### Multi-node workloads

- [lws](https://github.com/kubernetes-sigs/lws) - LeaderWorkerSet API for multi-pod workloads.

<a id="production-orchestration-gpu-management"></a>

#### GPU management

- [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) - Expose NVIDIA GPUs as Kubernetes resources.
- [gpu-operator](https://github.com/NVIDIA/gpu-operator) - Manage NVIDIA GPU drivers and components.

<a id="production-orchestration-multi-cloud-and-cluster-schedulers"></a>

#### Multi-cloud and cluster schedulers

- [SkyPilot + NVIDIA Dynamo](https://github.com/skypilot-org/skypilot/tree/master/examples/serve/nvidia-dynamo) - Example of running Dynamo across clouds with SkyPilot.
- [Democratizing AI Compute with AMD Using SkyPilot](https://rocm.blogs.amd.com/ecosystems-and-partners/democratizing-multicloud-skypi/README.html) - Multi-cloud deployment on AMD GPUs with SkyPilot.
- [dstack 0.20](https://dstack.ai/blog/0_20/) - Fleet-based orchestration for AI workloads across clouds and on-prem.
- [Migrating from Slurm to dstack](https://github.com/dstackai/migrate-from-slurm/blob/main/guide.md) - Step-by-step guide from Slurm to dstack.
- [Ray on TPUs with GKE](https://cloud.google.com/blog/products/containers-kubernetes/ray-on-tpus-with-gke-a-more-native-experience) - TPU-aware scheduling and slice reservations for Ray on GKE.
- [Ray Symmetric-Run](https://blog.vllm.ai/2025/11/22/ray-symmetric-run.html) - Launching multi-node vLLM on HPC-style clusters with one command.
- [Kubernetes + Ray + PyTorch + vLLM](https://www.youtube.com/watch?v=4o2amJxMHUc) - CNCF talk on an open-source AI compute stack.

<a id="production-orchestration-gpu-sharing-and-virtualization"></a>

#### GPU sharing and virtualization

- [HAMi](https://github.com/Project-HAMi/HAMi) - CNCF middleware for sharing and isolating heterogeneous accelerators on Kubernetes.
- [kvcached](https://github.com/ovg-project/kvcached) - Elastic, virtualized KV-cache that lets multiple models share a GPU.
- [kv-marketplace](https://github.com/neelsomani/kv-marketplace) - Experimental cross-GPU KV-cache sharing.

<a id="production-managed-kubernetes"></a>

### 4.4 Managed Kubernetes inference (EKS, GKE, AKS)

Provider-specific guides for serving models on Amazon EKS, Google GKE and Azure AKS.

[← Back to index](#contents)

<a id="production-managed-kubernetes-amazon-eks"></a>

#### Amazon EKS

- [EKS AI/ML Inference Guide](https://docs.aws.amazon.com/eks/latest/userguide/ml-inference.html) - Official entry point for serving on EKS with NVIDIA GPUs or Trainium/Inferentia, Karpenter and GPU node health monitoring.
- [Quickstart: vLLM on Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/ml-realtime-inference-llm-inference-vllm.html) - Deploying vLLM and tuning its configuration with GuideLLM benchmarks.
- [Load and Serve Models on EKS](https://docs.aws.amazon.com/eks/latest/userguide/ml-inference-load-serve-model.html) - Streaming weights from S3 to GPU memory with the Run:ai Model Streamer to cut cold starts.
- [EKS Best Practices: AI/ML Performance](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-performance.html) - Model artifact management, serving frameworks and container startup optimization.
- [GPU Inference on EKS Auto Mode](https://aws.amazon.com/blogs/containers/how-to-run-ai-model-inference-with-gpus-on-amazon-eks-auto-mode/) - GPU node pools with AWS-managed node lifecycle and patching.
- [ai-on-eks](https://github.com/awslabs/ai-on-eks) - AWS Labs blueprints and Terraform for AI workloads on EKS.
- [AI on EKS Inference Charts](https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/inference-charts) - Helm charts for vLLM, Ray-vLLM, Triton, AIBrix and LeaderWorkerSet on GPUs and Neuron.
- [vLLM with Ray Serve on Inferentia2](https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/framework-guides/Neuron/vllm-ray-inf2) - Serving LLMs on AWS Inferentia2 from EKS.
- [awsome-inference](https://github.com/aws-samples/awsome-inference) - Reference architectures for NIM, TensorRT-LLM, Triton and RayService, with multi-node LWS and MIG examples.
- [aws-neuron-eks-samples](https://github.com/aws-neuron/aws-neuron-eks-samples) - Samples for Inferentia and Trainium workloads on EKS.
- [EKS AI/ML Resources](https://docs.aws.amazon.com/eks/latest/userguide/ml-resources.html) - AWS index of EKS AI/ML repos, workshops and guides.

<a id="production-managed-kubernetes-google-gke"></a>

#### Google GKE

- [About AI/ML Model Inference on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/machine-learning/inference) - Official overview of GKE inference features, accelerators and routing.
- [GKE Inference Quickstart](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/inference/inference-quickstart) - Tested manifests and performance data by model, accelerator and latency target.
- [Serve an LLM with GKE Inference Gateway](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-with-gke-inference-gateway) - InferencePool-based, model-aware load balancing in front of vLLM.
- [Deploy GKE Inference Gateway Powered by llm-d](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/deploy-gke-inference-gateway) - Managed gateway using llm-d scheduling.
- [GKE Inference Autoscaling Best Practices](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling) - Choosing HPA signals such as queue size and batch size over GPU utilization.
- [Tuning the GKE HPA to Run Inference on GPUs](https://cloud.google.com/blog/products/containers-kubernetes/tuning-the-gke-hpa-to-run-inference-on-gpus) - Experiments comparing autoscaling metrics for LLM servers.
- [Serve an LLM Using TPU Trillium on GKE with vLLM](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu) - vLLM on TPU v6e with autoscaling on server metrics.
- [Serve an LLM Using TPUs on GKE with KubeRay](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray) - Ray Operator add-on with vLLM on TPU slices.
- [tpu-inference](https://github.com/vllm-project/tpu-inference) - vLLM backend for running on Google TPUs.
- [GKE Inference Reference Architecture](https://cloud.google.com/blog/topics/developers-practitioners/supercharge-your-ai-gke-inference-reference-architecture-your-blueprint-for-production-ready-inference) - Opinionated, Terraform-based blueprint for production inference on GKE.
- [Serve Open LLMs on GKE with a Pre-configured Architecture](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-open-models-terraform) - Hands-on deployment of the reference architecture.
- [accelerated-platforms](https://github.com/GoogleCloudPlatform/accelerated-platforms) - Source code for the GKE reference architecture and tutorials.

<a id="production-managed-kubernetes-azure-aks"></a>

#### Azure AKS

- [AI Toolchain Operator (KAITO) Add-on for AKS](https://learn.microsoft.com/azure/aks/ai-toolchain-operator) - Managed add-on that deploys open models with vLLM and provisions GPU nodes.
- [kaito](https://github.com/kaito-project/kaito) - CNCF Kubernetes AI Toolchain Operator for model inference and fine-tuning.
- [KAITO with vLLM](https://docs.vllm.ai/en/latest/deployment/integrations/kaito/) - vLLM's integration notes for KAITO.
- [KAITO Tool Calling](https://learn.microsoft.com/azure/aks/ai-toolchain-operator-tool-calling) - OpenAI-compatible endpoints with tool calling on AKS.
- [KAITO Monitoring](https://learn.microsoft.com/azure/aks/ai-toolchain-operator-monitoring) - Collecting vLLM inference metrics from KAITO workloads.
- [AKS-Managed GPU Nodes](https://learn.microsoft.com/azure/aks/aks-managed-gpu-nodes) - GPU node pools with AKS-managed drivers and device plugin.
- [GPU Partitioning on AKS](https://learn.microsoft.com/azure/aks/concepts-gpu-partitioning) - Comparison of MIG, time-slicing and MPS for sharing GPUs.
- [Multi-Instance GPU Node Pools](https://learn.microsoft.com/azure/aks/gpu-multi-instance) - Creating MIG-partitioned node pools on AKS.
- [Multi-Instance GPU with DRA on AKS](https://blog.aks.azure.com/2026/03/03/multi-instance-gpu-with-dra-on-aks) - Claim-based MIG allocation with the NVIDIA GPU Operator and DRA driver.

<a id="production-managed-kubernetes-cross-cloud"></a>

#### Cross-cloud

- [llm-d Infrastructure & Environments](https://llm-d.ai/docs/infrastructure) - Provider-specific cluster notes for running llm-d on GKE, AKS and other platforms.

<a id="production-observability"></a>

### 4.5 Observability and profiling

Monitor model requests, queues, and accelerator use.

[← Back to index](#contents)

<a id="production-observability-gpu-monitoring"></a>

#### GPU monitoring

- [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) - Prometheus exporter for NVIDIA GPU telemetry.

<a id="production-observability-tracing"></a>

#### Tracing

- [opentelemetry-collector](https://github.com/open-telemetry/opentelemetry-collector) - Telemetry collection and processing.

<a id="production-observability-dashboards"></a>

#### Dashboards

- [grafana](https://github.com/grafana/grafana) - Metrics and trace visualization.

<a id="production-observability-metrics"></a>

#### Metrics

- [prometheus](https://github.com/prometheus/prometheus) - Time-series monitoring and alerting.

<a id="production-observability-llm-request-tracing"></a>

#### LLM request tracing

- [langfuse](https://github.com/langfuse/langfuse) - Request tracing and model observability.

<a id="production-observability-gpu-profiling"></a>

#### GPU profiling

- [NVTX](https://github.com/NVIDIA/NVTX) - Instrumentation annotations for profiling.

<a id="production-observability-pytorch-profiling"></a>

#### PyTorch profiling

- [pytorch](https://github.com/pytorch/pytorch) - Profiler tooling available in PyTorch.

<a id="production-observability-host-overhead-and-gpu-visibility"></a>

#### Host overhead and GPU visibility

- [Host Overhead is Killing Your Inference Efficiency](https://modal.com/blog/host-overhead-inference-efficiency) - How CPU-side work starves the GPU and how to find it.
- [CPU-GPU Synchronization](https://tomasruizt.github.io/posts/08_cpu_gpu_synchronization/) - Spotting and removing hidden synchronization points.
- [The GPU Observability Gap](https://eunomia.dev/blog/2025/10/14/the-gpu-observability-gap-why-we-need-ebpf-on-gpu-devices/) - Case for eBPF-style visibility inside GPU devices.

<a id="production-observability-reference-setups"></a>

#### Reference setups

- [vllm-concurrency-demo](https://github.com/Regan-Milne/vllm-concurrency-demo) - Single-GPU vLLM concurrency tests with Prometheus and Grafana dashboards.

<a id="production-benchmarking"></a>

### 4.6 Performance benchmarking and model evaluation

Benchmark with representative workloads and report enough detail for others to reproduce the results.

[← Back to index](#contents)

<a id="production-benchmarking-load-testing"></a>

#### Load testing

- [aiperf](https://github.com/ai-dynamo/aiperf) - Generative AI load testing, trace replay and SLO analysis.

<a id="production-benchmarking-engine-suite"></a>

#### Engine suite

- [vllm](https://github.com/vllm-project/vllm) - Built-in serving benchmark scripts.
- [sglang](https://github.com/sgl-project/sglang) - Serving benchmark tools.

<a id="production-benchmarking-quality"></a>

#### Model-quality evaluation

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Model capability evaluation.

<a id="production-benchmarking-industry-benchmark"></a>

#### Industry performance benchmark

- [inference](https://github.com/mlcommons/inference) - MLPerf Inference benchmarks.

<a id="production-benchmarking-simulation"></a>

#### Simulation

- [vidur](https://github.com/microsoft/vidur) - Predictive performance simulation for LLM serving.

<a id="production-benchmarking-comparative-toolkit"></a>

#### Comparative benchmarking tools

- [llm-inference-benchmark](https://github.com/winderai/llm-inference-benchmark) - Inference benchmark tools and examples.

<a id="production-benchmarking-load-generators"></a>

#### Load generators

- [tokenflood](https://github.com/twerkmeister/tokenflood) - Load generator for simulating arbitrary request shapes against LLM endpoints.

<a id="production-benchmarking-kernel-benchmarks"></a>

#### Kernel benchmarks

- [FlashInfer-Bench](https://bench.flashinfer.ai/) - Standardized benchmark platform for inference kernels.

<a id="production-benchmarking-hardware-comparisons"></a>

#### Hardware comparisons

- [Artificial Analysis Hardware Benchmarks](https://artificialanalysis.ai/benchmarks/hardware) - Third-party comparisons of accelerator systems under concurrent LLM load.

<a id="production-benchmarking-evaluation-methodology"></a>

#### Evaluation methodology

- [Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook) - Hugging Face guide to designing and interpreting model evaluations.

<a id="production-capacity-planning"></a>

### 4.7 Capacity planning and economics

Use benchmark results to estimate GPU capacity and serving costs.

[← Back to index](#contents)

<a id="production-capacity-planning-simulator"></a>

#### Simulator

- [vidur](https://github.com/microsoft/vidur) - Estimate LLM serving behavior across configurations.

<a id="production-capacity-planning-serving-metrics"></a>

#### Serving metrics

- [vllm](https://github.com/vllm-project/vllm) - Runtime metrics and memory accounting.

<a id="production-capacity-planning-benchmarks"></a>

#### Benchmarks

- [aiperf](https://github.com/ai-dynamo/aiperf) - SLO-aware load testing and performance data.

<a id="production-capacity-planning-local-measurements"></a>

#### Local measurements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Includes local benchmarks and memory-conscious runtime.

<a id="production-capacity-planning-cost-and-energy-analysis"></a>

#### Cost and energy analysis

- [The Next 1000x Cost Saving of LLM](https://ralphmao.github.io/token-cost/) - Where past per-token cost reductions came from and where the next ones may.
- [Energy Use of AI Inference](https://arxiv.org/abs/2509.20241) - Per-query energy estimates and efficiency pathways at scale.
- [GPU (In)efficiency in AI Workloads](https://www.anyscale.com/blog/gpu-in-efficiency-in-ai-workloads) - Why production GPUs sit underutilized.
- [Revealing the Hidden Economics of Open Models](https://www.linuxfoundation.org/blog/revealing-the-hidden-economics-of-open-models-in-the-ai-era) - MIT and Linux Foundation research on open-model economics.

<a id="production-reliability-security"></a>

### 4.8 Reliability, security, and output safety

Keep model servers available during failures and overload, including in multi-tenant deployments.

[← Back to index](#contents)

<a id="production-reliability-security-gateway-protection"></a>

#### Gateway protection

- [Agent Router (formerly Envoy AI Gateway)](https://github.com/theagentrouter/agent-router) - Traffic controls for AI endpoints.

<a id="production-reliability-security-policy-and-routing"></a>

#### Policy and routing

- [gateway-api-inference-extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) - Kubernetes inference traffic interfaces.

<a id="production-reliability-security-serving-health"></a>

#### Serving health

- [kserve](https://github.com/kserve/kserve) - Deployment readiness and lifecycle patterns.

<a id="production-reliability-security-monitoring"></a>

#### Monitoring

- [prometheus](https://github.com/prometheus/prometheus) - Alerting inputs for resource and service health.

<a id="production-reliability-security-supply-chain-security"></a>

#### Supply-chain security

- [cosign](https://github.com/sigstore/cosign) - Sign and verify model artifact or container provenance.

<a id="production-reliability-security-failure-analyses"></a>

#### Failure analyses

- [vLLM: Why Requests Take Hours Under Load](https://blog.dotieuthien.com/posts/vllm) - Diagnosing KV block exhaustion and queue starvation.
- [One Token to Corrupt Them All](https://www.ai21.com/blog/vllm-debugging-mamba-bug/) - Debugging a state-corruption bug in vLLM's Mamba support.

<a id="production-reliability-security-runtime-output-checks"></a>

#### Runtime output safety checks

- [HaluGate](https://blog.vllm.ai/2025/12/14/halugate.html) - Token-level hallucination detection in the vLLM serving path.

<a id="production-confidential-inference"></a>

### 4.9 Confidential inference and model provenance

Protect prompts, outputs, and weights from infrastructure operators. Track model origins and changes.

[← Back to index](#contents)

<a id="production-confidential-inference-trusted-execution"></a>

#### Trusted execution

- [Confidential Compute for AI Inference with TEEs](https://chutes.ai/news/confidential-compute-for-ai-inference-how-chutes-delivers-verifiable-privacy-with-trusted-execution-environments) - Verifiable private inference on untrusted, permissionless GPU providers.

<a id="production-confidential-inference-model-provenance"></a>

#### Model provenance


<a id="production-case-studies"></a>

### 4.10 Production case studies

Case studies of large inference deployments. Most come from vendors, so treat their performance figures as vendor-reported unless independently reproduced.

[← Back to index](#contents)

<a id="production-case-studies-low-latency-agents"></a>

#### Low-latency agents

- [Cursor and Together AI](https://www.together.ai/blog/learn-how-cursor-partnered-with-together-ai-to-deliver-real-time-low-latency-inference-at-scale) - Bringing up Blackwell for in-editor agents, including kernel tuning and FP4 (vendor-reported).

<a id="production-case-studies-recommendation-systems"></a>

#### Recommendation systems

- [Turbocharging LinkedIn's Recommendation Systems with SGLang](https://www.linkedin.com/blog/engineering/ai/turbocharging-linkedins-recommendation-systems-with-sglang) - Using an LLM serving engine for ranking workloads.

<a id="production-case-studies-large-moe-serving"></a>

#### Large MoE serving

- [DeepSeek R1 671B on AMD MI300X](https://docs.moreh.io/benchmarking/deepseek_r1_671b_on_amd_mi300x_gpus_maximum_throughput/) - Throughput evaluation across 40 MI300X GPUs (vendor-reported).
- [FriendliAI Qwen3 235B Benchmark](https://friendli.ai/blog/Qwen3-235B-benchmark) - MoE-aware serving compared with a vLLM baseline (vendor-reported).
- [Fastest Inference for Top Open-Source Models](https://www.together.ai/blog/fastest-inference-for-the-top-open-source-models) - Speculative decoding and FP4 on Blackwell (vendor-reported).

<a id="production-case-studies-kv-caching"></a>

#### KV caching

- [GMI Cloud 4x TTFT with TensorMesh](https://www.gmicloud.ai/blog/gmi-cloud-achieves-4x-llm-performance-boost-with-tensormesh) - SSD-backed prefix caching for lower TTFT (vendor-reported).

<a id="modalities"></a>

## 5. Modalities

<a id="modalities-audio"></a>

### 5.1 Audio and speech inference

Streaming speech recognition, synthesis and latency-sensitive audio inference.

[← Back to index](#contents)

<a id="modalities-audio-asr"></a>

#### ASR

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Efficient Whisper speech-to-text.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) - Portable Whisper inference.

<a id="modalities-audio-tts"></a>

#### TTS

- [piper](https://github.com/rhasspy/piper) - Local neural text-to-speech.

<a id="modalities-audio-audio-models"></a>

#### Audio models

- [TTS](https://github.com/coqui-ai/TTS) - Speech synthesis toolkit; verify project maintenance before deployment.

<a id="modalities-audio-vad"></a>

#### VAD

- [silero-vad](https://github.com/snakers4/silero-vad) - Lightweight voice activity detection.

<a id="modalities-vision"></a>

### 5.2 Vision inference

Inference for image encoders, detectors, vision-language models and video streams.

[← Back to index](#contents)

<a id="modalities-vision-detection"></a>

#### Detection

- [ultralytics](https://github.com/ultralytics/ultralytics) - Object detection and computer vision model tooling.

<a id="modalities-vision-computer-vision"></a>

#### Computer vision

- [opencv](https://github.com/opencv/opencv) - Image/video processing and inference integrations.

<a id="modalities-vision-onnx"></a>

#### ONNX

- [onnxruntime](https://github.com/microsoft/onnxruntime) - Portable execution for vision graphs.

<a id="modalities-vision-multimodal-models"></a>

#### Multimodal models

- [transformers](https://github.com/huggingface/transformers) - Vision-language model architectures and processors.

<a id="modalities-diffusion"></a>

### 5.3 Image and video diffusion inference

Optimize the denoising loop, scheduler, attention, memory and decoding stages.

[← Back to index](#contents)

<a id="modalities-diffusion-pipelines"></a>

#### Pipelines

- [diffusers](https://github.com/huggingface/diffusers) - Diffusion pipelines for image, audio and video tasks.

<a id="modalities-diffusion-workflow-ui"></a>

#### Workflow UI

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Configurable generative media execution graphs.

<a id="modalities-diffusion-video-models"></a>

#### Video models

- [CogVideo](https://github.com/THUDM/CogVideo) - Video-generation model implementations.

<a id="modalities-diffusion-kernels"></a>

#### Kernels

- [flash-attention](https://github.com/Dao-AILab/flash-attention) - Optimized attention applicable to some diffusion components.

<a id="modalities-diffusion-general-runtime"></a>

#### General runtime

- [TensorRT](https://github.com/NVIDIA/TensorRT) - Inference optimizations for supported diffusion architectures.

<a id="modalities-diffusion-serving-engines"></a>

#### Serving engines

- [SGLang Diffusion](https://lmsys.org/blog/2025-11-07-sglang-diffusion/) - SGLang's serving stack applied to image and video generation.
- [vLLM-Omni Diffusion Guide](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion) - Attention backends and acceleration options for diffusion models in vLLM-Omni.
- [Diffusers Attention Backends](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends) - Choosing attention backends for diffusion pipelines.

<a id="modalities-diffusion-production-guides"></a>

#### Production guides

- [Productionizing Diffusion Models](https://a-r-r-o-w.github.io/blog/3_blossom/00001_productionizing_diffusion-1/) - Taking diffusion models from research code to production serving.

<a id="modalities-embeddings"></a>

### 5.4 Embedding and reranking inference

Serve non-generative and ranking models efficiently.

[← Back to index](#contents)

<a id="modalities-embeddings-embedding-models"></a>

#### Embedding models

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - Embeddings and cross-encoder reranking.
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) - Embedding and reranking model implementations.

<a id="modalities-embeddings-general-runtime"></a>

#### General runtime

- [onnxruntime](https://github.com/microsoft/onnxruntime) - Optimized inference for exported encoders.

<a id="modalities-embeddings-text-inference"></a>

#### Text inference

- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Inference for transformer sequence models.

<a id="modalities-embeddings-benchmarking"></a>

#### Benchmarking

- [aiperf](https://github.com/ai-dynamo/aiperf) - Benchmarking supports embedding-oriented endpoints.

<a id="learning"></a>

## 6. Learning

<a id="learning-papers"></a>

### 6.1 Research papers

Papers grouped by the inference bottleneck they address.

[← Back to index](#contents)

<a id="learning-papers-efficient-attention"></a>

#### Efficient attention

- [flash-attention](https://github.com/Dao-AILab/flash-attention) - Code and references for FlashAttention papers.

<a id="learning-papers-serving"></a>

#### Serving

- [vllm](https://github.com/vllm-project/vllm) - PagedAttention and vLLM paper references.

<a id="learning-papers-speculative-decoding"></a>

#### Speculative decoding

- [EAGLE](https://github.com/SafeAILab/EAGLE) - EAGLE research and implementations.

<a id="learning-papers-quantization"></a>

#### Quantization

- [llm-awq](https://github.com/mit-han-lab/llm-awq) - Activation-aware quantization research.

<a id="learning-papers-disaggregation"></a>

#### Disaggregation

- [Mooncake](https://github.com/kvcache-ai/Mooncake) - Disaggregated cache infrastructure research.

<a id="learning-papers-simulation"></a>

#### Simulation

- [vidur](https://github.com/microsoft/vidur) - Serving simulation research and implementation.

<a id="learning-papers-kv-cache-systems"></a>

#### KV-cache systems

- [LMCache: An Efficient KV Cache Layer for Enterprise-Scale Inference](https://arxiv.org/abs/2510.09665) - Paper behind LMCache's cross-query and cross-engine cache reuse.

<a id="learning-papers-cpu-gpu-offload"></a>

#### CPU–GPU offload

- [Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs](https://arxiv.org/abs/2506.03296) - Offloading KV-cache and attention to the CPU during decode.

<a id="learning-talks"></a>

### 6.2 Talks and conferences

Talks, slides, and discussions about inference systems.

[← Back to index](#contents)

<a id="learning-talks-recorded-engine-talks"></a>

#### Recorded engine talks

- [The Rise of vLLM](https://www.youtube.com/watch?v=WLl8D1nyaW8) - How vLLM grew from a research project into a widely used engine.
- [SGLang: Large-Scale LLM Serving](https://www.youtube.com/watch?v=w9-AYqIhHRo) - SGLang architecture and performance work.
- [Inference OSS Ecosystem featuring vLLM](https://www.nvidia.com/en-us/on-demand/session/other25-dynamoday06/) - Disaggregation and wide-EP for MoE on GB200 racks.
- [Inside the MAX Framework](https://www.youtube.com/watch?v=WK5dVQ8vhbU) - Modular meetup walkthrough of the MAX inference stack.
- [How to Make vLLM 13x Faster with LMCache + NVIDIA Dynamo](https://www.youtube.com/watch?v=iuoOpOQkURo) - Hands-on KV-cache reuse demo; reproduce the speedup on your workload.

<a id="learning-tutorials"></a>

### 6.3 Tutorials and guides

Self-paced courses and practical walkthroughs.

[← Back to index](#contents)

- [LLM Inference at Scale](https://github.com/harshuljain13/llm-inference-at-scale) - A production-focused guide to serving LLMs, from inference fundamentals to optimization and deployment.

<a id="learning-tutorials-vllm"></a>

#### vLLM

- [vllm](https://github.com/vllm-project/vllm) - Examples and documentation for serving and benchmarking.

<a id="learning-tutorials-sglang"></a>

#### SGLang

- [sglang](https://github.com/sgl-project/sglang) - Runtime and performance tutorials.
- [SGLang Cookbook](https://cookbook.sglang.io/) - Recipes for deploying specific models and features with SGLang.

<a id="learning-tutorials-inference-kernels"></a>

#### Inference kernels

- [triton](https://github.com/triton-lang/triton) - Triton kernel tutorials.

<a id="learning-tutorials-tensorrt-llm"></a>

#### TensorRT-LLM

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - Serving and optimized runtime examples.

<a id="learning-tutorials-local-inference"></a>

#### Local inference

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Local model execution and benchmarks.

<a id="learning-tutorials-distributed"></a>

#### Distributed

- [llm-d](https://github.com/llm-d/llm-d) - Distributed inference architecture examples.

<a id="learning-engine-internals"></a>

### 6.4 Engine internals

Guides for reading inference-engine source code.

[← Back to index](#contents)

<a id="learning-engine-internals-minimal-engines"></a>

#### Minimal engines

- [Inside Nano-vLLM (Part 1)](https://neutree.ai/blog/nano-vllm-part-1) - Scheduling, batching and memory management in a minimal vLLM reimplementation.
- [mini-sglang](https://github.com/sgl-project/mini-sglang) - Compact educational reimplementation of SGLang.
- [vLLM-Style Inference Engine from Scratch on CPU](https://medium.com/@alaminibrahim433/vllm-style-fast-inference-engine-building-from-scratch-on-cpu-1f2a1f31f02a) - Implementing PagedAttention and continuous batching by hand.

<a id="learning-engine-internals-vllm"></a>

#### vLLM

- [vLLM Architectural Deep Dive](https://docs.google.com/presentation/d/1dMnZyXDff1zh1bfV0v5bku4e7iWBaOkaGNsegnpWe3w/edit?usp=sharing) - Slides on vLLM's architecture and design decisions.
- [Paged Attention from First Principles](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) - PagedAttention explained from virtual-memory basics.
- [GPU Model Runner V2](https://github.com/vllm-project/vllm/pull/25266) - Pull request redesigning vLLM's model runner and input preparation.
- [vllm-daily](https://github.com/vllm-project/vllm-daily) - Daily summaries of merged vLLM pull requests.

<a id="learning-courses"></a>

### 6.5 Courses and books

Longer courses and books on inference systems.

[← Back to index](#contents)

<a id="learning-courses-university-courses"></a>

#### University courses

- [MIT 6.5940: TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) - Efficient inference techniques from edge devices to the cloud.
- [ECE 5545: ML Hardware and Systems](https://abdelfattah-class.github.io/ece5545/) - Course on ML hardware, systems design and algorithm–hardware interaction.
- [Stanford CS149 Assignment 5: Kernels](https://github.com/stanford-cs149/asst5-kernels) - Hands-on GPU kernel assignment.

<a id="learning-courses-books-and-playbooks"></a>

#### Books and playbooks

- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) - TPU/GPU architecture, communication and parallelism for training and inference.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) - Interactive guide to parallelism strategies and communication patterns.
- [Build an LLM from Scratch with MAX](https://llm.modular.com/) - Interactive tutorial on LLM internals using Modular's MAX.

<a id="contributing"></a>

## 7. Contributing

Read [the contribution guide](CONTRIBUTING.md) before submitting. Use the issue templates to propose a topic.

<a id="license"></a>

## 8. License

This repository's original content is licensed under the [MIT License](LICENSE). Linked projects, documentation, papers, images, and code retain their respective licenses.

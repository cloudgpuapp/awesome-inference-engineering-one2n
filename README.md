# Awesome Inference Engineering [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Explore LLM inference and serving, from model execution and GPU kernel optimization to distributed deployment, multimodal inference, and performance benchmarking. This list brings together open-source engines, research, benchmarks, and practical guides for engineers building and operating production inference systems.

The sections follow the models and performance, hardware, software, inference techniques, inference beyond LLMs, and production.

<a id="contents"></a>

## Table of contents

- [1. Models and performance](#models-and-performance)
  - [1.1 Inference fundamentals](#inference-fundamentals)
  - [1.2 Transformer and LLM architecture](#transformer-and-llm-architecture)
  - [1.3 Mixture of experts and emerging architectures](#emerging-architectures)
  - [1.4 Image and video models](#image-and-video-models)
  - [1.5 Bottleneck analysis](#bottleneck-analysis)
  - [1.6 Attention optimization](#attention-optimization)
  - [1.7 Model selection, evaluation and compression](#model-strategy)
- [2. Hardware](#hardware)
  - [2.1 GPU architecture](#gpu-architecture)
  - [2.2 Accelerators by vendor](#accelerators-by-vendor)
  - [2.3 Accelerator design and emerging hardware](#accelerator-design)
  - [2.4 Instances, interconnects and sizing](#instances-and-interconnects)
  - [2.5 Local and edge inference](#local-and-edge-inference)
- [3. Software](#software)
  - [3.1 GPU programming fundamentals](#gpu-programming)
  - [3.2 Kernel libraries](#kernel-libraries)
  - [3.3 GEMM and matmul kernels](#gemm-kernels)
  - [3.4 Attention and memory-access kernels](#attention-kernels)
  - [3.5 CuTe DSL and tile programming](#cute-and-tile)
  - [3.6 Kernel languages and compilers](#kernel-compilers)
  - [3.7 AI-assisted kernel generation](#ai-kernel-generation)
  - [3.8 Kernel debugging](#kernel-debugging)
  - [3.9 Frameworks and libraries](#frameworks-and-libraries)
  - [3.10 Model formats and runtimes](#model-formats-and-runtimes)
  - [3.11 Inference engines](#inference-engines)
  - [3.12 Engine internals and extensions](#engine-internals)
  - [3.13 Benchmarking](#benchmarking)
  - [3.14 Profiling](#profiling)
- [4. Inference techniques](#inference-techniques)
  - [4.1 Quantization concepts and formats](#quantization-concepts)
  - [4.2 Quantization algorithms](#quantization-algorithms)
  - [4.3 Quantization toolkits](#quantization-toolkits)
  - [4.4 Speculative decoding](#speculative-decoding)
  - [4.5 Structured outputs and sampling](#structured-outputs)
  - [4.6 KV-cache storage and offload](#kv-cache-storage)
  - [4.7 Prefix caching and cache-aware routing](#prefix-caching)
  - [4.8 Long-context handling](#long-context)
  - [4.9 Parallelism strategies](#parallelism-strategies)
  - [4.10 Communication libraries](#communication-libraries)
  - [4.11 Disaggregation](#disaggregation)
  - [4.12 Determinism and numerics](#determinism-and-numerics)
- [5. Inference beyond LLMs](#beyond-llms)
  - [5.1 Vision and vision-language models](#vision-language-models)
  - [5.2 Embedding models](#embedding-models)
  - [5.3 Speech](#speech)
  - [5.4 Image and video generation](#image-and-video-generation)
- [6. Production](#production)
  - [6.1 Containerization and packaging](#containerization)
  - [6.2 Kubernetes serving platforms](#kubernetes-serving)
  - [6.3 GPU enablement on Kubernetes](#gpu-enablement)
  - [6.4 Gateways and request routing](#gateways-and-routing)
  - [6.5 Model routing](#model-routing)
  - [6.6 Autoscaling and cold starts](#autoscaling)
  - [6.7 Cloud platforms](#cloud-platforms)
  - [6.8 Multi-cloud and GPU capacity](#multi-cloud)
  - [6.9 Observability](#observability)
  - [6.10 Security, quality and incident debugging](#security-and-reliability)
  - [6.11 Cost and efficiency](#cost-and-efficiency)
  - [6.12 Case studies](#case-studies)
- [7. Learning resources](#learning)
  - [7.1 Courses and books](#courses-and-books)
  - [7.2 Talks](#talks)
- [8. Contributing](#contributing)
- [9. License](#license)

<a id="models-and-performance"></a>

## 1. Models and performance

<a id="inference-fundamentals"></a>

### 1.1 Inference fundamentals

The request path, prefill and decode phases, KV cache and the latency metrics used to judge a serving system.

- [How LLM Inference Works](https://arpitbhayani.me/blogs/how-llm-inference-works) - Prompt-to-response walkthrough of tokenization, embedding and autoregressive generation.
- [Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - Prefill and decode, compute and memory limits, batching and KV-cache trade-offs.
- [LLM Inference Benchmarking: Fundamental Concepts](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/) - Definitions of TTFT, inter-token latency, tokens per second and end-to-end latency.
- [AI Model Optimization Techniques](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/) -  Overview of model-level optimizations for inference latency and cost.
- [KV Cache in LLM Inference](https://sebastianraschka.com/faq/docs/kv-cache.html) - Why KV cache is important in LLM inference

<a id="emerging-architectures"></a>

### 1.4 Image and video models

Diffusion, few-step generation and video models, covering the architecture side. Serving resources are in [Image and video generation](#image-and-video-generation).

- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Mathematical walkthrough of the forward noising and reverse denoising processes.
- [Diffusion Models](https://huggingface.co/blog/Esmail-AGumaan/diffusion-models) - Introductory explanation of how diffusion models are trained and sampled.
- [Latent Consistency Models](https://latent-consistency-models.github.io/) - Few-step image generation by distilling a diffusion model into a consistency model.
- [CogVideo](https://github.com/THUDM/CogVideo) - Video-generation model implementations.

<a id="bottleneck-analysis"></a>

### 1.5 Bottleneck analysis

Ops-to-byte ratio, arithmetic intensity and the roofline model for deciding whether a workload is compute- or memory-bound.

- [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic) - Back-of-the-envelope compute and memory math for transformer inference.
- [How Fast Can We Perform a Forward Pass?](https://bounded-regret.ghost.io/how-fast-can-we-perform-a-forward-pass/) - Theoretical and practical limits on forward-pass latency.
- [Making Deep Learning Model Performant From First Principles](https://horace.io/brrr_intro.html) - Separates compute, memory-bandwidth and overhead bottlenecks.
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) - Roofline analysis and arithmetic intensity applied to model workloads.

<a id="attention-optimization"></a>

### 1.6 Attention optimization

FlashAttention, PagedAttention and attention variants that cut the memory and compute cost of attention.

- [FlashAttention](https://arxiv.org/html/2205.14135v2) - IO-aware exact attention that tiles computation to avoid materializing the attention matrix.

- [PagedAttention](https://arxiv.org/abs/2309.06180) - The vLLM paper introducing paged KV-cache memory management.
- [Paged Attention from First Principles](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) - PagedAttention explained from virtual-memory basics.
- [flashinfer](https://github.com/flashinfer-ai/flashinfer) - Inference-specialized attention primitives.
- [xformers](https://github.com/facebookresearch/xformers) - Efficient attention and transformer operations.

<a id="model-strategy"></a>

### 1.7 Model selection, evaluation and compression

How to chose a model, finetuning, reducing the overall size through pruning. 

- [Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook) - Hugging Face guide to designing and interpreting model evaluations.
- [peft](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning methods such as LoRA, whose adapters engines can serve side by side.
- [LLM Model Pruning and Knowledge Distillation with NeMo](https://developer.nvidia.com/blog/llm-model-pruning-and-knowledge-distillation-with-nvidia-nemo-framework/) - Producing a smaller student model through pruning and distillation.
- [LLM Pruning Collection](https://github.com/zlab-princeton/llm-pruning-collection) - Pruning implementations and evaluation scripts for LLMs.

<a id="hardware"></a>

## 2. Hardware

<a id="gpu-architecture"></a>

### 2.1 GPU architecture

How GPU compute units, memory hierarchy and tensor cores determine inference performance.

- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) - How GPU chips and their networks work, and what that means for LLMs.
- [Intro to GPUs for the Researcher](https://hackbot.dad/writing/intro-to-gpus/) - Practical GPU mental model aimed at raising model FLOPS utilization.
- [GPU Architecture Deep Dive: From HBM to Tensor Cores](https://www.youtube.com/watch?v=5UWphJWdAHY) - Video tour of the GPU memory hierarchy and tensor cores.
- [GPU Glossary](https://modal.com/gpu-glossary) - Linked definitions of GPU hardware and software terms, from SMs to warps.

<a id="accelerators-by-vendor"></a>

### 2.2 Accelerators by vendor

Each vendor has the same mix of resources: architecture material, the software platform, and how to run LLM serving on it.

#### NVIDIA

- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) - H100 streaming multiprocessors, FP8 tensor cores and the Transformer Engine.
- [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) - Blackwell Ultra architecture, NVFP4 and attention-layer acceleration.
- [Inside the NVIDIA Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) - Vendor overview of the Rubin rack-scale platform, including the Vera CPU.
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) - Index of NVIDIA's CUDA libraries, compilers and developer tools.

#### AMD

- [AMD CDNA 3 Architecture White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) - Architecture of the MI300 series, including chiplets and memory.
- [AMD CDNA 4 Architecture White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) - Architecture of the MI350 series and its low-precision data types.
- [ROCm](https://rocm.docs.amd.com/en/latest/) - ROCm software platform for AMD accelerators.
- [vLLM on ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html) - AMD's guide to installing, running and benchmarking vLLM on Instinct GPUs.

#### Google TPU

- [The Chip Made for the AI Inference Era: The Google TPU](https://www.uncoveralpha.com/p/the-chip-made-for-the-ai-inference) - Technical and strategic overview of Google TPUs.
- [TPU System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) - Official reference for TPU chips, slices, pods and the TPU VM model.
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) - TPU compute, memory and interconnect explained for model workloads.
- [tpu-inference](https://github.com/vllm-project/tpu-inference) - vLLM backend for running on Google TPUs.

#### AWS Trainium and Inferentia

- [Trainium2 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) - Official description of Trainium2 NeuronCores, memory and interconnect.
- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) - SDK, compiler and runtime for Trainium and Inferentia.
- [vllm-neuron](https://github.com/vllm-project/vllm-neuron) - vLLM plugin for running on Trainium and Inferentia.
- [Autocomp Trainium Attention](https://charleshong3.github.io/blog/autocomp_trainium_attention.html) - Writing an attention kernel for AWS Trainium.

<a id="accelerator-design"></a>

### 2.3 Accelerator design and emerging hardware

Design principles behind AI accelerators, and research hardware outside the mainstream GPU vendors.

- [Domain-Specific Architectures](https://fleetwood.dev/posts/domain-specific-architectures) - Design principles behind AI accelerators.
- [A Close Look at SRAM for Inference](https://www.viksnewsletter.com/p/a-close-look-at-sram-for-inference) - Trade-offs between SRAM-heavy and HBM-based inference designs.
- [Maia 200](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/) - Vendor announcement of an inference-focused accelerator with FP8/FP4 tensor cores.
- [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489) - Shows that hardware-aligned model shapes improve runtime without hurting accuracy.
- [Analog In-Memory Computing Attention](https://www.nature.com/articles/s43588-025-00854-1) - Research on in-memory computing hardware for the attention mechanism.
- [TinyTPU](https://www.tinytpu.com/) - Build-a-TPU-from-scratch project explaining systolic-array design.
- [ECE298A-TPU](https://github.com/WilliamZhang20/ECE298A-TPU) - Student TPU-style accelerator design intended for tape-out.

<a id="instances-and-interconnects"></a>

### 2.4 Instances, interconnects and sizing

Picking instance types, understanding the interconnect between accelerators, and comparing systems before procurement.

- [NVLink and NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/) - Vendor overview of NVIDIA's GPU-to-GPU interconnect and switch fabric.
- [RoCEv2 for Deep Learning](https://iodized-hawthorn-94a.notion.site/RoCEv2-26954c8e1f0a80b78bf1c6adc583e670) - Introduction to RDMA over Converged Ethernet for GPU clusters.
- [Amazon EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/) - AWS accelerated-computing families for GPU, Trainium and Inferentia.
- [Google Cloud GPU Machine Types](https://docs.cloud.google.com/compute/docs/gpus) - GPU models, machine families and regional availability on Compute Engine.
- [Azure VM Sizes](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview) - Azure VM families, including the GPU-accelerated N-series.
- [Artificial Analysis Hardware Benchmarks](https://artificialanalysis.ai/benchmarks/hardware) - Third-party comparisons of accelerator systems under concurrent LLM load.

<a id="local-and-edge-inference"></a>

### 2.5 Local and edge inference

Running models on desktops, laptops and phones, where memory and power limits drive the design.

#### Desktop

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Portable C/C++ model inference and quantized formats.
- [ollama](https://github.com/ollama/ollama) - Local model distribution and runner.
- [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) - Consumer-GPU engine exploiting activation sparsity across GPU and CPU.
- [AirLLM](https://github.com/lyogavin/airllm) - Layer-by-layer loading that runs large models on small GPUs at the cost of latency.

#### Apple Silicon

- [mlx](https://github.com/ml-explore/mlx) - Array framework optimized for Apple Silicon.
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM inference and fine-tuning on Apple Silicon.
- [vllm-metal](https://github.com/vllm-project/vllm-metal) - Community plugin for running vLLM on Apple Silicon GPUs.
- [On-Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama) - Apple's walkthrough of optimizing an LLM for on-device Core ML inference.

#### Mobile

- [executorch](https://github.com/pytorch/executorch) - PyTorch runtime for on-device inference on mobile and embedded hardware.
- [mlc-llm](https://github.com/mlc-ai/mlc-llm) - Compiler-based deployment of LLMs to phones, browsers and GPUs.
- [MediaPipe LLM Inference](https://developers.google.com/edge/mediapipe/solutions/genai/llm_inference) - Google's API for running LLMs on Android, iOS and the web.

<a id="software"></a>

## 3. Software

<a id="gpu-programming"></a>

### 3.1 GPU programming fundamentals

Programming models, instruction sets and examples for NVIDIA and AMD GPUs, plus the core parallel patterns.

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) - Official CUDA programming model reference.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) - Official reference for NVIDIA's virtual instruction set.
- [cuda-samples](https://github.com/NVIDIA/cuda-samples) - Examples for understanding CUDA execution.
- [HIP Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/) - Official HIP programming model reference for AMD GPUs.
- [AMD CDNA 3 Instruction Set Architecture](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) - Official reference for the MI300 instruction set.
- [rocm-examples](https://github.com/ROCm/rocm-examples) - Examples for HIP and ROCm libraries.
- [Blocks, Threads, and Kernels: A Deeper Dive](https://vanshnawander.github.io/vansh/posts/blocks-threads-kernels.html) - Thread hierarchy, block organization and kernel launch mechanics.
- [Reduction (Sum) Series: Part 1](https://kathsucurry.github.io/cuda/2025/10/14/reduction_sum_part1.html) - Introduction to parallel reduction in CUDA.
- [Reduction (Sum) Series: Part 2](https://kathsucurry.github.io/cuda/2025/10/27/reduction_sum_part2.html) - Implementing and optimizing reduction kernels step by step.

<a id="kernel-libraries"></a>

### 3.2 Kernel libraries

Vendor-maintained building blocks that engines call for GEMM, convolution and attention.

- [cutlass](https://github.com/NVIDIA/cutlass) - Templates for accelerated GEMM operations.
- [cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend) - C++ and Python API for cuDNN graph operations, including fused attention.
- [aiter](https://github.com/ROCm/aiter) - AMD inference operators.
- [composable_kernel](https://github.com/ROCm/composable_kernel) - AMD's templated library for fused GEMM, attention and other operators.

<a id="gemm-kernels"></a>

### 3.3 GEMM and matmul kernels

Step-by-step optimization of the matrix multiplications that dominate inference compute.

- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) - Step-by-step path from a naive kernel to near-cuBLAS performance.
- [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - Worklog of Hopper-specific matmul optimizations.
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) - From PTX/SASS to warp tiling and asynchronous tensor-core pipelines.
- [CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) - High-performance GEMM with CUTLASS and Hopper WGMMA instructions.
- [B200 Blockscaled GEMM: The Setup](https://veitner.bearblog.dev/b200-blockscaled-gemm-the-setup/) - Layouts and MMA setup for block-scaled GEMM on Blackwell.
- [fast.cu](https://github.com/pranjalssh/fast.cu) - Performance-tuned CUDA kernels written from scratch as reference implementations.
- [Blackwell NVFP4 Kernel Hackathon Journey](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html) - Practical notes on tuning NVFP4 kernels for Blackwell.

<a id="attention-kernels"></a>

### 3.4 Attention and memory-access kernels

Warp roles, shared-memory layout and data movement in attention and transpose kernels.

- [Unweaving Warp Specialization](https://rohany.github.io/blog/warp-specialization/) - Assigning producer and consumer roles to warps for higher throughput.
- [Flash Attention from Scratch Part 4: Bank Conflicts & Swizzling](https://lubits.ch/flash/Part-4) - Shared-memory bank conflicts and swizzling in an attention kernel.
- [Triton Flash Attention Kernel Walkthrough](https://nathanchen.me/public/Triton-Flash-Attention-Kernel-Walkthrough.html) - Line-by-line reading of a Triton attention kernel.
- [FlexAttention in FlashAttention CuTe DSL](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/) - Writing attention variants such as causal and sliding-window masks.
- [Effective Transpose on Hopper GPU](https://github.com/simveit/effective_transpose) - Optimized matrix transpose for Hopper.

<a id="cute-and-tile"></a>

### 3.5 CuTe DSL and tile programming

Layout algebra and tile-based programming models for tensor-core kernels.

- [Blackwell Pipelining with CuTeDSL](https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/) - Overlapping work with CuTeDSL asynchronous pipelines on Blackwell.
- [Categorical Foundations for CuTe Layouts](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) - The mathematics behind CuTe's layout algebra.
- [Demystifying Numeric Conversions in CuTeDSL](https://veitner.bearblog.dev/demystifying-numeric-conversions-in-cutedsl/) - Type-conversion primitives in CuTeDSL.
- [CUDA Tile](https://developer.nvidia.com/cuda/tile) - NVIDIA's tile-based programming model for tensor-core kernels.
- [cuda-tile](https://github.com/NVIDIA/cuda-tile) - MLIR-based CUDA Tile IR and compiler infrastructure.

<a id="kernel-compilers"></a>

### 3.6 Kernel languages and compilers

Higher-level languages and compilers that generate kernels instead of hand-writing them.

- [triton](https://github.com/triton-lang/triton) - Compiler and language for writing GPU kernels.
- [helion](https://github.com/pytorch/helion) - Python-embedded DSL from PyTorch for writing ML kernels with autotuning.
- [Learn by Doing: TorchInductor Reduction Kernels](https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-Reduction/) - How TorchInductor generates reduction kernels.
- [tvm](https://github.com/apache/tvm) - Machine learning compiler framework.
- [iree](https://github.com/iree-org/iree) - Compiler and runtime for portable model deployment.
- [ISPC](https://ispc.github.io/index.html) - Intel's implicit-SPMD compiler for SIMD CPU and GPU code.
- [Inside VOLT](https://arxiv.org/abs/2511.13751) - Open-source compiler design for open GPU architectures with custom ISAs.
- [Rust GPU](https://rust-gpu.github.io/) - Toolchain for writing GPU shaders and compute kernels in Rust.
- [rust-cuda](https://github.com/Rust-GPU/rust-cuda) - Libraries and tools for writing CUDA kernels in Rust.

<a id="ai-kernel-generation"></a>

### 3.7 AI-assisted kernel generation

Coding agents and automated search applied to writing and tuning kernels.

- [AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) - Benchmark environment for coding agents writing GPU kernels.
- [Agent-Assisted Kernel Optimization](https://www.wafer.ai/blog/topk-sigmoid-optimization) - Case study of an agent tuning a kernel on AMD MI300X down to assembly.
- [Automating Algorithm Discovery: Kernel Generation](https://adrs-ucb.notion.site/datadog) - Case study on automated search for GPU kernel algorithms.
- [FlashInfer-Bench paper](https://arxiv.org/abs/2601.00227) - Closed loop connecting generated kernels, benchmarks and serving-engine integration.

<a id="kernel-debugging"></a>

### 3.8 Kernel debugging

Finding hangs, memory errors and crashes in GPU code on NVIDIA and AMD.

- [Tracing Hanging GPU Kernels to Source](https://blog.vllm.ai/2025/12/03/improved-cuda-debugging.html) - CUDA core dumps and tooling for locating hung kernels in vLLM.
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/) - NVIDIA tool for detecting memory, race and synchronization errors in CUDA kernels.
- [AMD GPU Debugging](https://thegeeko.me/blog/amd-gpu-debugging/) - Debugging GPU programs with ROCm tools.
- [ROCgdb](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/) - AMD's source-level debugger for HIP and ROCm programs.

<a id="frameworks-and-libraries"></a>

### 3.9 Frameworks and libraries

The model frameworks that engines build on and that most model code is written in.

- [pytorch](https://github.com/pytorch/pytorch) - Core framework, including torch.compile and the PyTorch profiler.
- [jax](https://github.com/jax-ml/jax) - Composable transformations and XLA compilation, widely used on TPUs.
- [transformers](https://github.com/huggingface/transformers) - Reference implementations for a broad range of model architectures.
- [tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenizers used in language-model workloads.

<a id="model-formats-and-runtimes"></a>

### 3.10 Model formats and runtimes

Weight formats and compiled runtimes for deploying models outside a training framework.

- [safetensors](https://github.com/safetensors/safetensors) - Safe, zero-copy tensor serialization format used by most open checkpoints.
- [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) - Specification of the single-file quantized model format used by llama.cpp.
- [onnxruntime](https://github.com/microsoft/onnxruntime) - Portable inference runtime for ONNX models.
- [TensorRT](https://github.com/NVIDIA/TensorRT) - Optimizing compiler/runtime for NVIDIA inference.
- [AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) - Ahead-of-time compilation of PyTorch models for deployment without a Python runtime.

<a id="inference-engines"></a>

### 3.11 Inference engines

Serving engines that handle batching, KV-cache management and model execution.

- [vllm](https://github.com/vllm-project/vllm) - Serving engine with continuous batching and KV-cache management.
- [sglang](https://github.com/sgl-project/sglang) - High-performance LLM and multimodal serving runtime.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA-optimized language-model inference.
- [lmdeploy](https://github.com/InternLM/lmdeploy) - Deployment toolkit with inference backends.
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - Rust inference runtime for language models.
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Fast inference for transformer sequence models.
- [server](https://github.com/triton-inference-server/server) - Triton Inference Server, supporting multiple model backends.

<a id="engine-internals"></a>

### 3.12 Engine internals and extensions

How engines are built, how to follow their development, and how to extend them without forking.

- [Inside Nano-vLLM (Part 1)](https://neutree.ai/blog/nano-vllm-part-1) - Scheduling, batching and memory management in a minimal vLLM reimplementation.
- [mini-sglang](https://github.com/sgl-project/mini-sglang) - Compact educational reimplementation of SGLang.
- [vLLM-Style Inference Engine from Scratch on CPU](https://medium.com/@alaminibrahim433/vllm-style-fast-inference-engine-building-from-scratch-on-cpu-1f2a1f31f02a) - Implementing PagedAttention and continuous batching by hand.
- [vLLM Architectural Deep Dive](https://docs.google.com/presentation/d/1dMnZyXDff1zh1bfV0v5bku4e7iWBaOkaGNsegnpWe3w/edit?usp=sharing) - Slides on vLLM's architecture and design decisions.
- [GPU Model Runner V2](https://github.com/vllm-project/vllm/pull/25266) - Pull request redesigning vLLM's model runner and input preparation.
- [vllm-daily](https://github.com/vllm-project/vllm-daily) - Daily summaries of merged vLLM pull requests.
- [SGLang Cookbook](https://cookbook.sglang.io/) - Recipes for deploying specific models and features with SGLang.
- [vLLM plugin system](https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html) - Extending vLLM through plugins instead of maintaining a fork.
- [Awesome vLLM Plugins](https://github.com/BudEcosystem/Awesome-vLLM-plugins) - Curated list of community vLLM plugins.
- [ArcticInference](https://github.com/snowflakedb/ArcticInference) - Snowflake's vLLM plugin with latency and throughput optimizations.

<a id="benchmarking"></a>

### 3.13 Benchmarking

Load generators, standard suites and simulators for measuring serving performance. Benchmark the workload you intend to serve.

- [aiperf](https://github.com/ai-dynamo/aiperf) - Generative AI load testing, trace replay and SLO analysis.
- [genai-bench](https://github.com/sgl-project/genai-bench) - SGLang's token-level benchmark for LLM serving endpoints.
- [guidellm](https://github.com/vllm-project/guidellm) - vLLM project's tool for benchmarking deployments against realistic traffic.
- [locust](https://github.com/locustio/locust) - General-purpose, scriptable load-testing framework.
- [tokenflood](https://github.com/twerkmeister/tokenflood) - Load generator for simulating arbitrary request shapes against LLM endpoints.
- [inference](https://github.com/mlcommons/inference) - MLPerf Inference benchmarks.
- [llm-inference-benchmark](https://github.com/winderai/llm-inference-benchmark) - Inference benchmark tools and examples.
- [FlashInfer-Bench](https://bench.flashinfer.ai/) - Standardized benchmark platform for inference kernels.
- [vidur](https://github.com/microsoft/vidur) - Estimate LLM serving behavior across configurations.

<a id="profiling"></a>

### 3.14 Profiling

System and kernel profilers for NVIDIA and AMD, and the host-side overheads they reveal.

- [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) - NVIDIA's system-wide timeline profiler for CPU, GPU and communication activity.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) - Profiling CUDA kernels and interpreting hardware metrics.
- [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/) - AMD's system-wide timeline profiler for CPU and GPU activity.
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) - Kernel-level performance analysis for AMD Instinct GPUs.
- [NVTX](https://github.com/NVIDIA/NVTX) - Instrumentation annotations for profiling.
- [Host Overhead is Killing Your Inference Efficiency](https://modal.com/blog/host-overhead-inference-efficiency) - How CPU-side work starves the GPU and how to find it.
- [CPU-GPU Synchronization](https://tomasruizt.github.io/posts/08_cpu_gpu_synchronization/) - Spotting and removing hidden synchronization points.
- [The GPU Observability Gap](https://eunomia.dev/blog/2025/10/14/the-gpu-observability-gap-why-we-need-ebpf-on-gpu-devices/) - Case for eBPF-style visibility inside GPU devices.

<a id="inference-techniques"></a>

## 4. Inference techniques

<a id="quantization-concepts"></a>

### 4.1 Quantization concepts and formats

How low-precision formats work and how they affect model quality.

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Illustrated introduction to LLM quantization.
- [Post Training Quantization](https://liyuan24.github.io/writings/2026_01_06_post_training_quantization.html) - Detailed walkthrough of post-training weight quantization.
- [Quantization-Aware Training for LLMs with PyTorch](https://pytorch.org/blog/quantization-aware-training/) - Recovering accuracy lost to quantization by simulating it during training.
- [Survey of Quantization Formats](https://github.com/vipulSharma18/Survey-of-Quantization-Formats) - MXFP8 and NVFP4 formats with TorchAO and GemLite benchmarks on an RTX 4090.
- [INT vs FP](https://github.com/ChenMnZ/INT_vs_FP) - Framework comparing low-bit integer and floating-point formats.
- [Why Stochastic Rounding is Essential](https://cloud.google.com/blog/topics/developers-practitioners/why-stochastic-rounding-is-essential-for-modern-generative-ai) - Stochastic rounding for quality in low-precision arithmetic.
- [Future Leakage in Block-Quantized Attention](https://matx.com/research/leaky_quantization) - How block quantization can leak future-token information into causal attention.

<a id="quantization-algorithms"></a>

### 4.2 Quantization algorithms

Implementations of specific weight-quantization methods and the kernels that run them.

- [llm-awq](https://github.com/mit-han-lab/llm-awq) - Activation-aware weight quantization.
- [gptq](https://github.com/IST-DASLab/gptq) - Post-training weight quantization research implementation.
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) - Formerly popular GPTQ implementation; review maintenance before adoption.
- [GPTQModel](https://github.com/ModelCloud/GPTQModel) - Maintained GPTQ toolkit with CUDA, ROCm, XPU and CPU backends.
- [AutoRound x LLM Compressor](https://blog.vllm.ai/2025/12/09/intel-autoround-llmc.html) - Intel AutoRound integrated with LLM Compressor for vLLM serving.
- [AutoRound Meets SGLang](https://lmsys.org/blog/2025-11-13-AutoRound/) - Serving AutoRound low-bit models in SGLang.
- [GemLite](https://dropbox.github.io/gemlite_blogpost/) - Fused low-bit GEMV kernels with bit-packing and dequantization.
- [BitNet](https://github.com/microsoft/BitNet) - Inference framework for 1-bit (ternary) LLMs.

<a id="quantization-toolkits"></a>

### 4.3 Quantization toolkits

Framework and vendor toolkits for producing quantized checkpoints that engines can load.

- [llm-compressor](https://github.com/vllm-project/llm-compressor) - LLM quantization and compression toolkit.
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - Low-precision GPU operations.
- [torchao Quantized Models on Hugging Face Hub](https://pytorch.org/blog/torchao-quantized-models-and-quantization-recipes-now-available-on-huggingface-hub/) - PyTorch-native quantization recipes with pre-quantized checkpoints.
- [torchao float8](https://github.com/pytorch/ao/blob/main/torchao/float8/README.md) - Float8 workflows in torchao.
- [Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) - NVIDIA ModelOpt library for quantization, pruning and distillation targeting TensorRT-LLM and vLLM.
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) - FP8 and FP4 transformer building blocks for Hopper, Ada and Blackwell.
- [Quark](https://github.com/amd/Quark) - AMD's quantization toolkit for producing models that run on Instinct GPUs.
- [TransformerEngine for ROCm](https://github.com/ROCm/TransformerEngine) - AMD's port of TransformerEngine for FP8 on Instinct GPUs.

<a id="speculative-decoding"></a>

### 4.4 Speculative decoding

Draft-target, EAGLE, Medusa and n-gram methods that propose several tokens per forward pass.

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - The original draft-target speculative decoding paper.
- [EAGLE](https://github.com/SafeAILab/EAGLE) - Speculative sampling family and associated code.
- [Medusa](https://github.com/FasterDecoding/Medusa) - Multiple decoding heads for candidate token generation.
- [prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) - N-gram drafting from the prompt, with no draft model.
- [SuffixDecoding](https://suffix-decoding.github.io/) - CPU-side suffix trees for draft-model-free speculation on repetitive agentic workloads.
- [speculators](https://github.com/vllm-project/speculators) - Library for building, evaluating and packaging speculative decoding models for vLLM.
- [Speculators: Production-Ready Speculative Decoding](https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding) - Why the speculators library standardizes draft-model formats.
- [Spec-Bench](https://github.com/hemingkx/Spec-Bench) - Unified benchmark for comparing speculative decoding methods.

<a id="structured-outputs"></a>

### 4.5 Structured outputs and sampling

Constrained decoding, tool calling and the sampling step at the end of each forward pass.

- [xgrammar](https://github.com/mlc-ai/xgrammar) - Grammar-constrained decoding engine used by vLLM, SGLang and TensorRT-LLM.
- [llguidance](https://github.com/guidance-ai/llguidance) - Low-latency constrained decoding library for JSON schemas and grammars.
- [Guided Decoding Performance on vLLM and SGLang](https://blog.squeezebits.com/70642) - Benchmarks XGrammar and LLGuidance backends across both engines.
- [Combining Guided Decoding and Speculative Decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.md) - How TensorRT-LLM applies grammar constraints to speculative drafts.
- [Tool Calling in Inference](https://www.baseten.co/blog/tool-calling-in-inference/) - Why tool-call quality differs between providers and how to make it reliable.
- [TSON](https://github.com/zenoaihq/tson) - Experimental compact object notation that reduces tokens spent on structured output.
- [Sorting-Free GPU Kernels for LLM Sampling](https://flashinfer.ai/2025/03/10/sampling.html) - Top-k/top-p sampling on GPU without a full sort.

<a id="kv-cache-storage"></a>

### 4.6 KV-cache storage and offload

Storing, compressing and moving the KV cache across GPU, CPU and remote tiers.

- [LMCache](https://github.com/LMCache/LMCache) - KV-cache storage and reuse across tiers.
- [LMCache: An Efficient KV Cache Layer for Enterprise-Scale Inference](https://arxiv.org/abs/2510.09665) - Paper behind LMCache's cross-query and cross-engine cache reuse.
- [vLLM KV Offloading Connector](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html) - Offloading KV-cache to CPU DRAM and tuning host-to-device throughput.
- [How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/) - KV-cache offload and reuse in Dynamo.
- [PrisKV](https://aibrix.github.io/posts/2025-11-26-priskv-intro/) - Colocated, tiered KV-cache store from the AIBrix project.
- [Optimizing Inference with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) - FP4 KV-cache for long contexts and large batches.
- [PQCache](https://sky-light.eecs.berkeley.edu/#/blog/pqcache) - Product quantization of the KV-cache for long-context inference.
- [kvcached](https://github.com/ovg-project/kvcached) - Elastic, virtualized KV-cache that lets multiple models share a GPU.
- [kv-marketplace](https://github.com/neelsomani/kv-marketplace) - Experimental cross-GPU KV-cache sharing.
- [Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs](https://arxiv.org/abs/2506.03296) - Offloading KV-cache and attention to the CPU during decode.

<a id="prefix-caching"></a>

### 4.7 Prefix caching and cache-aware routing

Reusing computed prefixes within an engine and sending requests to the replica that already holds them.

- [Prompt Caching](https://ngrok.com/blog/prompt-caching/) - Practical explanation of how provider-side prompt caching works and when it pays off.
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) - Design of vLLM's hash-based prefix cache.
- [SGLang RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) - Radix-tree KV-cache reuse across requests in SGLang.
- [Ray Serve Custom Request Routing](https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing) - Prefix-aware routing to cut TTFT in multi-turn chat.
- [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see) - Prefix-, load- and SLO-aware routing in llm-d.

<a id="long-context"></a>

### 4.8 Long-context handling

Position encodings and attention changes that make long sequences tractable.

- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) - Explains RoPE, the position encoding most context-extension methods modify.
- [DroPE](https://pub.sakana.ai/DroPE/) - Extends context by dropping positional embeddings instead of long-context fine-tuning.
- [Long Context Attention](https://nrehiew.github.io/blog/long_context/) - Compute and memory challenges of scaling attention to long sequences.
- [Optimizing Long-Context Prefill on Older GPU Nodes](https://moreh.io/blog/optimizing-long-context-prefill-on-multiple-older-generation-gpu-nodes-251226/) - Distributing long-context prefill across older-generation GPUs.

<a id="parallelism-strategies"></a>

### 4.9 Parallelism strategies

Tensor, pipeline, data and expert parallelism for models that do not fit on one device.

- [Visualizing Parallelism in Transformer](https://ailzhang.github.io/posts/distributed-compute-in-transformer/) - Diagrams of tensor, pipeline and data parallelism across devices.
- [The vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) - Combining TP, DP, PP and expert parallelism for MoE models in vLLM.
- [Shift Parallelism](https://arxiv.org/abs/2509.16495) - Switching between tensor and data parallelism to balance latency and throughput.
- [Learning to Love Mesh-Oriented Sharding](https://blog.ezyang.com/2025/12/learning-to-love-mesh-oriented-sharding/) - Mental model for device-mesh sharding.
- [Tensor Parallel in NanoVLLM](https://liyuan24.github.io/writings/2025_12_18_nanovllm_tensor_parallel_kernel_fusion.html) - Tensor parallelism and kernel fusion in a minimal engine.
- [vLLM Large Scale Serving with Wide-EP](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) - Wide expert parallelism for DeepSeek on H200 clusters.
- [LPLB](https://github.com/deepseek-ai/LPLB) - Early-stage linear-programming MoE expert load balancer.
- [Ray Symmetric-Run](https://blog.vllm.ai/2025/11/22/ray-symmetric-run.html) - Launching multi-node vLLM on HPC-style clusters with one command.

<a id="communication-libraries"></a>

### 4.10 Communication libraries

Collective and point-to-point libraries that move activations, experts and KV cache between GPUs.

- [nccl](https://github.com/NVIDIA/nccl) - GPU collective communication.
- [Low Latency Communication Kernels with NVSHMEM](https://www.youtube.com/live/rZBF2PuycLQ) - GPU MODE lecture on GPU-initiated communication.
- [rccl](https://github.com/ROCm/rccl) - AMD's collective communication library for ROCm GPUs.
- [rocSHMEM](https://github.com/ROCm/rocSHMEM) - AMD's GPU-initiated OpenSHMEM communication library.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) - MoE communication implementation.
- [uccl](https://github.com/uccl-project/uccl) - GPU communication library for collectives, P2P KV transfer and expert parallelism.

<a id="disaggregation"></a>

### 4.11 Disaggregation

Running prefill and decode on separate worker pools and moving the KV cache between them.

- [dynamo](https://github.com/ai-dynamo/dynamo) - Distributed inference serving components.
- [Disaggregated Inference: 18 Months Later](https://hao-ai-lab.github.io/blogs/distserve-retro/) - DistServe authors' retrospective on prefill/decode disaggregation adoption.
- [Disaggregated Inference at Scale with PyTorch and vLLM](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/) - Running separate prefill and decode pools with vLLM.
- [Disaggregated Serving in TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog05_Disaggregated_Serving_in_TensorRT-LLM.md) - TensorRT-LLM's disaggregated serving design.
- [SGLang PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) - Configuring prefill/decode disaggregation in SGLang.
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - Disaggregated KV-cache and data movement.
- [llm-d](https://github.com/llm-d/llm-d) - Kubernetes-oriented distributed LLM inference.
- [llm-d Architecture](https://llm-d.ai/docs/architecture) - Component design and topology of llm-d.
- [Distributed Inference on Heterogeneous Accelerators](https://moreh.io/blog/distributed-inference-on-heterogeneous-accelerators-including-gpus-rubin-cpx-and-ai-accelerators-250923/) - Splitting inference across mixed AMD and NVIDIA hardware.

<a id="determinism-and-numerics"></a>

### 4.12 Determinism and numerics

Why identical requests can produce different outputs, and why inference numerics must match training for RL.

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) - Traces nondeterminism to batch-dependent kernels and shows batch-invariant fixes.
- [Enabling Deterministic Inference for SGLang](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) - Integrating batch-invariant kernels into SGLang.
- [Bitwise Consistent On-Policy RL](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) - Matching vLLM inference numerics with TorchTitan training.
- [When Speed Kills Stability](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) - How training–inference numeric mismatch collapses RL runs.

<a id="beyond-llms"></a>

## 5. Inference beyond LLMs

<a id="vision-language-models"></a>

### 5.1 Vision and vision-language models

Image and video inputs, omni-modal pipelines and classic computer vision.

- [vLLM Multimodal Inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) - Passing images, video and audio to vision-language models in vLLM.
- [vllm-omni](https://github.com/vllm-project/vllm-omni) - vLLM extension for serving models that mix text, image, audio and video.
- [vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models](https://arxiv.org/abs/2602.02204) - Research on serving multimodal pipelines as independently scheduled stages.
- [cornserve](https://github.com/cornserve-ai/cornserve) - Serving system that splits multimodal pipelines into independently scaled components.
- [ultralytics](https://github.com/ultralytics/ultralytics) - Object detection and computer vision model tooling.
- [opencv](https://github.com/opencv/opencv) - Image/video processing and inference integrations.

<a id="embedding-models"></a>

### 5.2 Embedding models

BERT-style and LLM-based embedding and reranking models, and the servers built for them.

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - Embeddings and cross-encoder reranking.
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) - Embedding and reranking model implementations.
- [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) - Hugging Face's serving engine for embedding and reranker models.
- [infinity](https://github.com/michaelfeil/infinity) - High-throughput REST server for embeddings, reranking and classification.

<a id="speech"></a>

### 5.3 Speech

Voice activity detection, speech recognition, text-to-speech and speech-to-speech models.

- [silero-vad](https://github.com/snakers4/silero-vad) - Lightweight voice activity detection.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Efficient Whisper speech-to-text.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) - Portable Whisper inference.
- [piper](https://github.com/rhasspy/piper) - Local neural text-to-speech.
- [TTS](https://github.com/coqui-ai/TTS) - Speech synthesis toolkit; verify project maintenance before deployment.
- [moshi](https://github.com/kyutai-labs/moshi) - Full-duplex speech-to-speech model with streaming inference.

<a id="image-and-video-generation"></a>

### 5.4 Image and video generation

Serving diffusion pipelines, where step count, attention backends and parallelism set the cost.

- [diffusers](https://github.com/huggingface/diffusers) - Diffusion pipelines for image, audio and video tasks.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Configurable generative media execution graphs.
- [SGLang Diffusion](https://lmsys.org/blog/2025-11-07-sglang-diffusion/) - SGLang's serving stack applied to image and video generation.
- [vLLM-Omni Diffusion Guide](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion) - Attention backends and acceleration options for diffusion models in vLLM-Omni.
- [Diffusers Attention Backends](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends) - Choosing attention backends for diffusion pipelines.
- [xDiT](https://github.com/xdit-project/xDiT) - Parallel inference engine for diffusion transformers across multiple GPUs.
- [Productionizing Diffusion Models](https://a-r-r-o-w.github.io/blog/3_blossom/00001_productionizing_diffusion-1/) - Taking diffusion models from research code to production serving.

<a id="production"></a>

## 6. Production

<a id="containerization"></a>

### 6.1 Containerization and packaging

GPU-enabled containers, prebuilt inference microservices from NVIDIA and AMD, and model packaging.

- [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) - Runtime hooks that expose NVIDIA GPUs to containers.
- [NVIDIA NIM](https://docs.nvidia.com/nim/) - Prebuilt, optimized inference microservice containers for NVIDIA GPUs.
- [Running ROCm Docker Containers](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) - Exposing AMD GPUs to containers.
- [AMD Inference Microservice (AIM)](https://rocm.blogs.amd.com/artificial-intelligence/enterprise-ai-aims/README.html) - Prebuilt inference containers for AMD Instinct GPUs.
- [BentoML](https://github.com/bentoml/BentoML) - Inference packaging and service deployment.

<a id="kubernetes-serving"></a>

### 6.2 Kubernetes serving platforms

Controllers and runtimes for deploying model servers on Kubernetes.

- [kserve](https://github.com/kserve/kserve) - Kubernetes model serving framework.
- [ray](https://github.com/ray-project/ray) - Distributed Python runtime including Ray Serve.
- [production-stack](https://github.com/vllm-project/production-stack) - Production-oriented manifests and components.
- [lws](https://github.com/kubernetes-sigs/lws) - LeaderWorkerSet API for multi-pod workloads.
- [HAMi](https://github.com/Project-HAMi/HAMi) - CNCF middleware for sharing and isolating heterogeneous accelerators on Kubernetes.
- [Kubernetes + Ray + PyTorch + vLLM](https://www.youtube.com/watch?v=4o2amJxMHUc) - CNCF talk on an open-source AI compute stack.

<a id="gpu-enablement"></a>

### 6.3 GPU enablement on Kubernetes

Device plugins and operators that make accelerators schedulable.

- [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) - Expose NVIDIA GPUs as Kubernetes resources.
- [gpu-operator](https://github.com/NVIDIA/gpu-operator) - Manage NVIDIA GPU drivers and components.
- [ROCm k8s-device-plugin](https://github.com/ROCm/k8s-device-plugin) - Expose AMD GPUs as Kubernetes resources.
- [AMD GPU Operator](https://github.com/ROCm/gpu-operator) - Manage AMD GPU drivers, device plugin and metrics on Kubernetes.

<a id="gateways-and-routing"></a>

### 6.4 Gateways and request routing

API gateways and load balancers that sit in front of model servers.

- [litellm](https://github.com/BerriAI/litellm) - Provider abstraction, routing, quotas and API gateway features.
- [Agent Router (formerly Envoy AI Gateway)](https://github.com/theagentrouter/agent-router) - AI request routing using Envoy infrastructure.
- [gateway-api-inference-extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) - Inference-specific routing APIs for Kubernetes Gateway API.
- [gateway](https://github.com/Portkey-AI/gateway) - Open-source AI gateway.
- [vLLM router](https://github.com/vllm-project/router) - Lightweight router for large vLLM deployments.
- [vLLM LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora/) - Serving many LoRA adapters on one base model and routing requests to them.
- [optillm](https://github.com/algorithmicsuperintelligence/optillm) - OpenAI-compatible proxy that applies inference-time reasoning techniques to improve answers.

<a id="model-routing"></a>

### 6.5 Model routing

Choosing which model should answer a request, based on cost, difficulty or intent.

- [semantic-router](https://github.com/vllm-project/semantic-router) - vLLM's mixture-of-models router that classifies requests and picks a model.
- [vLLM Semantic Router: Improving Efficiency in AI Reasoning](https://developers.redhat.com/articles/2025/09/11/vllm-semantic-router-improving-efficiency-ai-reasoning) - Cost-aware routing between reasoning and non-reasoning paths.
- [Signal-Decision Driven Architecture](https://blog.vllm.ai/2025/11/19/signal-decision.html) - How the semantic router evolved from classification to signal-based decisions.
- [LLM Routing](https://www.liuxunzhuo.com/llm-routing) - Overview of strategies for routing requests between models.

<a id="autoscaling"></a>

### 6.6 Autoscaling and cold starts

Scaling on queue depth and load, provisioning GPU nodes quickly, and loading weights fast.

- [keda](https://github.com/kedacore/keda) - Event-driven autoscaling, including scale to zero, on custom metrics such as queue length.
- [karpenter](https://github.com/kubernetes-sigs/karpenter) - Node autoscaler that provisions right-sized GPU nodes on demand.
- [model-streamer](https://github.com/dsx-ai-factory/model-streamer) - Run:ai Model Streamer for streaming weights from object storage straight to GPU memory.
- [vLLM: Why Requests Take Hours Under Load](https://blog.dotieuthien.com/posts/vllm) - Diagnosing KV block exhaustion and queue starvation.

<a id="cloud-platforms"></a>

### 6.7 Cloud platforms

Managed Kubernetes guides, blueprints and accelerator integrations for each of the three major clouds.

#### AWS (EKS)

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

#### Google Cloud (GKE)

- [About AI/ML Model Inference on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/machine-learning/inference) - Official overview of GKE inference features, accelerators and routing.
- [GKE Inference Quickstart](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/inference/inference-quickstart) - Tested manifests and performance data by model, accelerator and latency target.
- [Serve an LLM with GKE Inference Gateway](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-with-gke-inference-gateway) - InferencePool-based, model-aware load balancing in front of vLLM.
- [Deploy GKE Inference Gateway Powered by llm-d](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/deploy-gke-inference-gateway) - Managed gateway using llm-d scheduling.
- [GKE Inference Autoscaling Best Practices](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling) - Choosing HPA signals such as queue size and batch size over GPU utilization.
- [Tuning the GKE HPA to Run Inference on GPUs](https://cloud.google.com/blog/products/containers-kubernetes/tuning-the-gke-hpa-to-run-inference-on-gpus) - Experiments comparing autoscaling metrics for LLM servers.
- [Serve an LLM Using TPU Trillium on GKE with vLLM](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu) - vLLM on TPU v6e with autoscaling on server metrics.
- [Serve an LLM Using TPUs on GKE with KubeRay](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray) - Ray Operator add-on with vLLM on TPU slices.
- [Ray on TPUs with GKE](https://cloud.google.com/blog/products/containers-kubernetes/ray-on-tpus-with-gke-a-more-native-experience) - TPU-aware scheduling and slice reservations for Ray on GKE.
- [GKE Inference Reference Architecture](https://cloud.google.com/blog/topics/developers-practitioners/supercharge-your-ai-gke-inference-reference-architecture-your-blueprint-for-production-ready-inference) - Opinionated, Terraform-based blueprint for production inference on GKE.
- [Serve Open LLMs on GKE with a Pre-configured Architecture](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-open-models-terraform) - Hands-on deployment of the reference architecture.
- [accelerated-platforms](https://github.com/GoogleCloudPlatform/accelerated-platforms) - Source code for the GKE reference architecture and tutorials.

#### Azure (AKS)

- [AI Toolchain Operator (KAITO) Add-on for AKS](https://learn.microsoft.com/azure/aks/ai-toolchain-operator) - Managed add-on that deploys open models with vLLM and provisions GPU nodes.
- [kaito](https://github.com/kaito-project/kaito) - CNCF Kubernetes AI Toolchain Operator for model inference and fine-tuning.
- [KAITO with vLLM](https://docs.vllm.ai/en/latest/deployment/integrations/kaito/) - vLLM's integration notes for KAITO.
- [KAITO Tool Calling](https://learn.microsoft.com/azure/aks/ai-toolchain-operator-tool-calling) - OpenAI-compatible endpoints with tool calling on AKS.
- [KAITO Monitoring](https://learn.microsoft.com/azure/aks/ai-toolchain-operator-monitoring) - Collecting vLLM inference metrics from KAITO workloads.
- [Use NVIDIA GPUs on AKS](https://learn.microsoft.com/en-us/azure/aks/use-nvidia-gpu) - Creating GPU node pools and installing the NVIDIA device plugin.
- [AKS-Managed GPU Nodes](https://learn.microsoft.com/azure/aks/aks-managed-gpu-nodes) - GPU node pools with AKS-managed drivers and device plugin.
- [GPU Partitioning on AKS](https://learn.microsoft.com/azure/aks/concepts-gpu-partitioning) - Comparison of MIG, time-slicing and MPS for sharing GPUs.
- [Multi-Instance GPU Node Pools](https://learn.microsoft.com/azure/aks/gpu-multi-instance) - Creating MIG-partitioned node pools on AKS.
- [Multi-Instance GPU with DRA on AKS](https://blog.aks.azure.com/2026/03/03/multi-instance-gpu-with-dra-on-aks) - Claim-based MIG allocation with the NVIDIA GPU Operator and DRA driver.
- [KEDA Add-on for AKS](https://learn.microsoft.com/en-us/azure/aks/keda-about) - Managed KEDA for event-driven autoscaling of inference workloads.

<a id="multi-cloud"></a>

### 6.8 Multi-cloud and GPU capacity

Finding GPU capacity across clouds and on-prem, and running one stack on all of them.

- [skypilot](https://github.com/skypilot-org/skypilot) - Runs AI workloads across clouds and Kubernetes clusters, picking available and cheaper GPUs.
- [SkyPilot + NVIDIA Dynamo](https://github.com/skypilot-org/skypilot/tree/master/examples/serve/nvidia-dynamo) - Example of running Dynamo across clouds with SkyPilot.
- [Democratizing AI Compute with AMD Using SkyPilot](https://rocm.blogs.amd.com/ecosystems-and-partners/democratizing-multicloud-skypi/README.html) - Multi-cloud deployment on AMD GPUs with SkyPilot.
- [dstack 0.20](https://dstack.ai/blog/0_20/) - Fleet-based orchestration for AI workloads across clouds and on-prem.
- [Migrating from Slurm to dstack](https://github.com/dstackai/migrate-from-slurm/blob/main/guide.md) - Step-by-step guide from Slurm to dstack.
- [llm-d Infrastructure & Environments](https://llm-d.ai/docs/infrastructure) - Provider-specific cluster notes for running llm-d on GKE, AKS and other platforms.

<a id="observability"></a>

### 6.9 Observability

GPU telemetry from both vendors, plus metrics, traces and dashboards for the serving path.

- [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) - Prometheus exporter for NVIDIA GPU telemetry.
- [device-metrics-exporter](https://github.com/ROCm/device-metrics-exporter) - Prometheus exporter for AMD GPU telemetry.
- [prometheus](https://github.com/prometheus/prometheus) - Time-series monitoring and alerting.
- [grafana](https://github.com/grafana/grafana) - Metrics and trace visualization.
- [opentelemetry-collector](https://github.com/open-telemetry/opentelemetry-collector) - Telemetry collection and processing.
- [langfuse](https://github.com/langfuse/langfuse) - Request tracing and model observability.
- [vllm-concurrency-demo](https://github.com/Regan-Milne/vllm-concurrency-demo) - Single-GPU vLLM concurrency tests with Prometheus and Grafana dashboards.

<a id="security-and-reliability"></a>

### 6.10 Security, quality and incident debugging

Artifact provenance, private inference, output guards, and write-ups of real serving bugs.

- [cosign](https://github.com/sigstore/cosign) - Sign and verify model artifact or container provenance.
- [Confidential Compute for AI Inference with TEEs](https://chutes.ai/news/confidential-compute-for-ai-inference-how-chutes-delivers-verifiable-privacy-with-trusted-execution-environments) - Verifiable private inference on untrusted, permissionless GPU providers.
- [HaluGate](https://blog.vllm.ai/2025/12/14/halugate.html) - Token-level hallucination detection in the vLLM serving path.
- [One Token to Corrupt Them All](https://www.ai21.com/blog/vllm-debugging-mamba-bug/) - Debugging a state-corruption bug in vLLM's Mamba support.

<a id="cost-and-efficiency"></a>

### 6.11 Cost and efficiency

Where inference cost and energy go, and why GPUs sit idle.

- [The Next 1000x Cost Saving of LLM](https://ralphmao.github.io/token-cost/) - Where past per-token cost reductions came from and where the next ones may.
- [Energy Use of AI Inference](https://arxiv.org/abs/2509.20241) - Per-query energy estimates and efficiency pathways at scale.
- [GPU (In)efficiency in AI Workloads](https://www.anyscale.com/blog/gpu-in-efficiency-in-ai-workloads) - Why production GPUs sit underutilized.
- [Revealing the Hidden Economics of Open Models](https://www.linuxfoundation.org/blog/revealing-the-hidden-economics-of-open-models-in-the-ai-era) - MIT and Linux Foundation research on open-model economics.

<a id="case-studies"></a>

### 6.12 Case studies

Case studies of large inference deployments. Most come from vendors, so treat their performance figures as vendor-reported unless independently reproduced.

- [Cursor and Together AI](https://www.together.ai/blog/learn-how-cursor-partnered-with-together-ai-to-deliver-real-time-low-latency-inference-at-scale) - Bringing up Blackwell for in-editor agents, including kernel tuning and FP4 (vendor-reported).
- [Turbocharging LinkedIn's Recommendation Systems with SGLang](https://www.linkedin.com/blog/engineering/ai/turbocharging-linkedins-recommendation-systems-with-sglang) - Using an LLM serving engine for ranking workloads.
- [DeepSeek R1 671B on AMD MI300X](https://docs.moreh.io/benchmarking/deepseek_r1_671b_on_amd_mi300x_gpus_maximum_throughput/) - Throughput evaluation across 40 MI300X GPUs (vendor-reported).
- [FriendliAI Qwen3 235B Benchmark](https://friendli.ai/blog/Qwen3-235B-benchmark) - MoE-aware serving compared with a vLLM baseline (vendor-reported).
- [Fastest Inference for Top Open-Source Models](https://www.together.ai/blog/fastest-inference-for-the-top-open-source-models) - Speculative decoding and FP4 on Blackwell (vendor-reported).
- [GMI Cloud 4x TTFT with TensorMesh](https://www.gmicloud.ai/blog/gmi-cloud-achieves-4x-llm-performance-boost-with-tensormesh) - SSD-backed prefix caching for lower TTFT (vendor-reported).

<a id="learning"></a>

## 7. Learning resources

<a id="courses-and-books"></a>

### 7.1 Courses and books

- [Inference Engineering](https://www.baseten.co/inference-engineering/book/) - Philip Kiely's book on building faster, cheaper and more reliable generative AI inference systems.
- [LLM Inference at Scale](https://github.com/harshuljain13/llm-inference-at-scale) - A production-focused guide to serving LLMs, from inference fundamentals to optimization and deployment.
- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) - TPU/GPU architecture, communication and parallelism for training and inference.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) - Interactive guide to parallelism strategies and communication patterns.
- [ai-infra-book](https://github.com/bojieli/ai-infra-book) - Quantitative treatment of AI systems and infrastructure.
- [MIT 6.5940: TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) - Efficient inference techniques from edge devices to the cloud.
- [ECE 5545: ML Hardware and Systems](https://abdelfattah-class.github.io/ece5545/) - Course on ML hardware, systems design and algorithm–hardware interaction.
- [Stanford CS149 Assignment 5: Kernels](https://github.com/stanford-cs149/asst5-kernels) - Hands-on GPU kernel assignment.
- [Build an LLM from Scratch with MAX](https://llm.modular.com/) - Interactive tutorial on LLM internals using Modular's MAX.

<a id="talks"></a>

### 7.2 Talks

- [The Rise of vLLM](https://www.youtube.com/watch?v=WLl8D1nyaW8) - How vLLM grew from a research project into a widely used engine.
- [SGLang: Large-Scale LLM Serving](https://www.youtube.com/watch?v=w9-AYqIhHRo) - SGLang architecture and performance work.
- [Inference OSS Ecosystem featuring vLLM](https://www.nvidia.com/en-us/on-demand/session/other25-dynamoday06/) - Disaggregation and wide-EP for MoE on GB200 racks.
- [Inside the MAX Framework](https://www.youtube.com/watch?v=WK5dVQ8vhbU) - Modular meetup walkthrough of the MAX inference stack.
- [How to Make vLLM 13x Faster with LMCache + NVIDIA Dynamo](https://www.youtube.com/watch?v=iuoOpOQkURo) - Hands-on KV-cache reuse demo; reproduce the speedup on your workload.

<a id="contributing"></a>

## 8. Contributing

Read [the contribution guide](CONTRIBUTING.md) before submitting. Use the issue templates to propose a topic.

<a id="license"></a>

## 9. License

This repository's original content is licensed under the [MIT License](LICENSE). Linked projects, documentation, papers, images, and code retain their respective licenses.

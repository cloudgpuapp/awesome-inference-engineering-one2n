# Awesome Inference Engineering [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This is a curated collection of inference fundamentals, open-source inference engines, research papers, benchmarks, and practical guides for engineers who want to understand what happens when a model runs, and what it takes to run it well in production.

It starts with the basics of model serving, then goes deeper into the hardware and systems underneath and the optimisations that make inference faster and cheaper. Later sections cover multimodal inference, including computer vision, video analytics and image and video generation, and how to run and scale all of these workloads in production.

<p align="center">
  <a href="assets/inference-landscape.png"><img src="assets/inference-landscape.png" alt="Inference Engineering landscape" width="100%"></a>
</p>

<a id="contents"></a>

## Table of contents

- [1. Inference fundamentals](#fundamentals)
  - [1.1 What inference is](#inference-basics)
  - [1.2 Transformers and attention](#transformers-and-attention)
  - [1.3 The KV cache](#kv-cache-basics)
  - [1.4 Prefill, decode and serving modes](#prefill-and-decode)
  - [1.5 Batching and request scheduling](#batching-and-scheduling)
  - [1.6 Latency and throughput metrics](#latency-metrics)
  - [1.7 Bottleneck analysis](#bottleneck-analysis)
  - [1.8 Inference optimization overview](#optimization-overview)
  - [1.9 Model selection, evaluation and fine-tuning](#model-strategy)
- [2. Hardware](#hardware)
  - [2.1 GPU architecture](#gpu-architecture)
  - [2.2 Accelerators by vendor](#accelerators-by-vendor)
  - [2.3 Instances, interconnects and sizing](#instances-and-interconnects)
- [3. Software](#software)
  - [3.1 Frameworks, runtimes and model formats](#frameworks-and-formats)
  - [3.2 Inference engines](#inference-engines)
  - [3.3 Benchmarking and profiling](#benchmarking-and-profiling)
  - [3.4 Kernel libraries, languages and compilers](#kernel-libraries)
  - [3.5 GEMM kernels and tile programming](#gemm-kernels)
- [4. Inference techniques](#inference-techniques)
  - [4.1 Quantization](#quantization)
  - [4.2 Speculative decoding](#speculative-decoding)
  - [4.3 KV-cache reuse, routing and offload](#kv-cache)
  - [4.4 Long-context handling](#long-context)
  - [4.5 Parallelism and communication](#parallelism-and-communication)
  - [4.6 Disaggregation](#disaggregation)
- [5. Omni-modal inference](#beyond-llms)
  - [5.1 Omni-modal inference](#omni-modal-inference)
  - [5.2 Vision-language models](#vision-language-models)
  - [5.3 Speech](#speech)
  - [5.4 Computer vision and video analytics](#vision-and-video-analytics)
  - [5.5 Image and video generation](#image-and-video-generation)
- [6. Production inference](#production)
  - [6.1 Containers and GPU enablement](#containers-and-gpus)
  - [6.2 Kubernetes serving platforms](#kubernetes-serving)
  - [6.3 Gateways, routing and autoscaling](#routing-and-autoscaling)
  - [6.4 Cloud and GPU hosting platforms](#cloud-platforms)
  - [6.5 Observability, security and debugging](#observability-and-reliability)
  - [6.6 Cost and efficiency](#cost-and-efficiency)
- [7. Learning resources](#learning)
  - [7.1 Courses and books](#courses-and-books)
  - [7.2 GitHub repositories](#github-repositories)
  - [7.3 Talks](#talks)
- [Contributing](#contributing)
- [License](#license)

<a id="fundamentals"></a>

## 1. Inference fundamentals

<a id="inference-basics"></a>

### 1.1 What inference is

- [The Difference Between AI Training and Inference](https://nebius.com/blog/posts/difference-between-ai-training-and-inference) - How serving a trained model differs from training it.
- [How LLM Inference Works](https://arpitbhayani.me/blogs/how-llm-inference-works) - Prompt-to-response walkthrough of tokenization, embedding and autoregressive generation.

<a id="transformers-and-attention"></a>

### 1.2 Transformers and attention

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Diagram-led walkthrough of self-attention and the encoder-decoder transformer.
- [FlashAttention](https://arxiv.org/html/2205.14135v2) - IO-aware exact attention that tiles computation to avoid materializing the attention matrix.

<a id="kv-cache-basics"></a>

### 1.3 The KV cache

- [KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching) - How caching keys and values speeds up transformer decoding.
- [What Is a KV Cache, and Why Does It Make LLM Inference Faster?](https://sebastianraschka.com/faq/docs/kv-cache.html) - What the KV cache stores and why LLM inference depends on it.
- [Paged Attention from First Principles](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) - PagedAttention explained from virtual-memory basics.

<a id="prefill-and-decode"></a>

### 1.4 Prefill, decode and serving modes

- [Prefill vs Decode](https://redis.io/blog/prefill-vs-decode/) - Why prefill is compute-bound and decode is memory-bound.
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) - How sparse expert layers work, and what they change for training and inference.
- [Using Asynchronous Inference in Production](https://www.baseten.co/blog/using-asynchronous-inference-in-production/) - Queue-based inference for long-running or bursty workloads.

<a id="batching-and-scheduling"></a>

### 1.5 Batching and request scheduling

- [Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching) - Hugging Face walkthrough from static batching to iteration-level scheduling.
- [Continuous Batching: Optimizing LLM Inference Throughput](https://mbrenndoerfer.com/writing/continuous-batching) - Interactive explanation of how requests join and leave a running batch.
- [Efficient Request Queueing](https://huggingface.co/blog/tngtech/llm-performance-request-queueing) - How queueing in front of an LLM server affects latency and throughput under load.

<a id="latency-metrics"></a>

### 1.6 Latency and throughput metrics

- [LLM Inference Benchmarking: Fundamental Concepts](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/) - Definitions of TTFT, inter-token latency, tokens per second and end-to-end latency.
- [NIM LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/metrics) - How NVIDIA defines and measures TTFT, inter-token latency, throughput and request latency.

<a id="bottleneck-analysis"></a>

### 1.7 Bottleneck analysis

- [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic) - Back-of-the-envelope compute and memory math for transformer inference.
- [How Fast Can We Perform a Forward Pass?](https://bounded-regret.ghost.io/how-fast-can-we-perform-a-forward-pass/) - Theoretical and practical limits on forward-pass latency.
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) - Separates compute, memory-bandwidth and overhead bottlenecks.
- [A Guide to LLM Inference and Performance](https://www.baseten.co/blog/llm-transformer-inference-guide/) - Ops:byte ratio, arithmetic intensity and memory bandwidth applied to choosing a GPU.
- [What Is the Roofline Model?](https://modal.com/gpu-glossary/perf/roofline-model) - Short definition of the roofline model with a worked diagram.
- [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) - NVIDIA's background guide to math-bound and memory-bound operations.
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) - Roofline analysis and arithmetic intensity applied to model workloads.

<a id="optimization-overview"></a>

### 1.8 Inference optimization overview

- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - Prefill and decode, compute and memory limits, batching and KV-cache trade-offs.
- [Top 5 AI Model Optimization Techniques for Faster, Smarter Inference](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/) - Post-training quantization, quantization-aware training, quantization-aware distillation, speculative decoding and pruning with distillation.

<a id="model-strategy"></a>

### 1.9 Model selection, evaluation and fine-tuning

#### Model selection

- [Choosing the Right Model](https://handbook.modular.com/getting-started/choosing-the-right-model/) - Criteria for picking an open model for a given task, size and budget.
- [LLM Benchmarking Framework with LLM-as-a-JURY](https://github.com/aws-samples/sample-bedrock-migration-and-modernization-tools/blob/main/360-eval/README.md) - AWS 360-Eval, which benchmarks candidate models on your own data using a jury of LLM judges.

#### Evaluation

- [LLM Evaluation: A Beginner's Guide](https://www.evidentlyai.com/llm-guide/llm-evaluation) - Evaluation methods for LLM systems, from benchmarks to LLM-as-a-judge.
- [Langfuse](https://github.com/langfuse/langfuse) - Open-source platform for tracing, evaluating and monitoring LLM applications.
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - Open-source LLM tracing and evaluation built on OpenTelemetry.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - EleutherAI's framework for running standard academic benchmarks against a model.
- [Lighteval](https://github.com/huggingface/lighteval) - Hugging Face toolkit for evaluating LLMs across inference backends such as vLLM and SGLang.
- [DeepEval](https://github.com/confident-ai/deepeval) - Pytest-style framework for unit-testing LLM outputs with metrics and LLM-as-a-judge.
- [promptfoo](https://github.com/promptfoo/promptfoo) - CLI for testing prompts, models and agents, including red-teaming.
- [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) - UK AI Security Institute's framework for LLM and agent evaluations.

#### Fine-tuning and adapters

- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning methods such as LoRA, which produce small adapters that engines can serve together on one base model.
- [TRL](https://github.com/huggingface/trl) - Hugging Face library for supervised fine-tuning, DPO and RL-based post-training.
- [Unsloth](https://github.com/unslothai/unsloth) - Memory-efficient fine-tuning and RL for open models, with export to GGUF and vLLM.
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - Config-driven fine-tuning for open models, from LoRA to multi-GPU full fine-tuning.
- [LlamaFactory](https://github.com/hiyouga/LlamaFactory) - Unified fine-tuning for 100+ LLMs and VLMs with a CLI and web UI.

#### Pruning and distillation

- [LLM Model Pruning and Knowledge Distillation with NVIDIA NeMo Framework](https://developer.nvidia.com/blog/llm-model-pruning-and-knowledge-distillation-with-nvidia-nemo-framework/) - Producing a smaller student model through pruning and distillation.

<a id="hardware"></a>

## 2. Hardware

<a id="gpu-architecture"></a>

### 2.1 GPU architecture

- [Intro to GPUs for the Researcher](https://hackbot.dad/writing/intro-to-gpus/) - Practical GPU mental model aimed at raising model FLOPS utilization.
- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) - How GPU chips and their networks work, and what that means for LLMs.
- [What Is GPU Memory and Why It Matters for LLM Inference](https://www.bentoml.com/blog/what-is-gpu-memory-and-why-it-matters-for-llm-inference) - How weights, KV cache and activations compete for VRAM.

<a id="accelerators-by-vendor"></a>

### 2.2 Accelerators by vendor

#### NVIDIA

- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) - Architecture guide to the NVIDIA H100 Tensor Core GPU.
- [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) - Blackwell Ultra architecture, NVFP4 and attention-layer acceleration.
- [Inside the NVIDIA Vera Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) - Vendor overview of the Vera Rubin rack-scale platform and its chips, including the Vera CPU.

#### AMD

- [AMD CDNA 3 Architecture White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) - Architecture of the MI300 series, including chiplets and memory.
- [AMD CDNA 4 Architecture White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) - Architecture of the MI350 series and its low-precision data types.
- [ROCm Documentation](https://rocm.docs.amd.com/en/latest/) - Documentation for AMD's ROCm software platform for Instinct accelerators.

#### Google TPU

- [Ironwood: The first Google TPU for the age of inference](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/) - Google's announcement of its seventh-generation, inference-focused TPU.
- [TPU Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) - Official reference for TensorCores, MXUs, slices, pods and multislice topologies.
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) - TPU compute, memory and interconnect explained for model workloads.

#### AWS Trainium and Inferentia

- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) - SDK, compiler and vLLM integration for AWS Trainium and Inferentia accelerators.

<a id="instances-and-interconnects"></a>

### 2.3 Instances, interconnects and sizing

- [NVLink and NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/) - Vendor overview of NVIDIA's GPU-to-GPU interconnect and switch fabric.
- [RoCE networks for distributed AI training at scale](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/) - How Meta designs and runs RDMA over Converged Ethernet networks for its GPU clusters.
- [Amazon EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/) - AWS accelerated-computing families for GPU, Trainium and Inferentia.
- [Google Cloud GPU Machine Types](https://docs.cloud.google.com/compute/docs/gpus) - GPU models, machine families and regional availability on Compute Engine.
- [Azure VM Sizes](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview) - Azure VM families, including the GPU-accelerated N-series.
- [Artificial Analysis Hardware Benchmarks](https://artificialanalysis.ai/hardware-inference-stack/) - Independent throughput, speed and cost benchmarks of inference hardware, from datacenter systems to laptops and phones.

<a id="software"></a>

## 3. Software

<a id="frameworks-and-formats"></a>

### 3.1 Frameworks, runtimes and model formats

- [PyTorch](https://github.com/pytorch/pytorch) - Core framework, including torch.compile and the PyTorch profiler.
- [JAX](https://github.com/jax-ml/jax) - Composable transformations and XLA compilation, widely used on TPUs.
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face model definitions for text, vision, audio and multimodal models, used for both inference and training.
- [Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenizer implementations used by most open language models.
- [Common AI Model Formats](https://huggingface.co/blog/ngxson/common-ai-model-formats) - Comparison of GGUF, safetensors, PyTorch and ONNX formats.
- [safetensors](https://github.com/safetensors/safetensors) - Safe, zero-copy tensor serialization format used by most open checkpoints.
- [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) - Specification of the single-file quantized model format used by llama.cpp.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Cross-platform inference runtime for ONNX models.
- [TensorRT](https://github.com/NVIDIA/TensorRT) - NVIDIA SDK that compiles and optimizes models for inference on NVIDIA GPUs.
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel toolkit for optimizing and running models on Intel CPUs, GPUs and NPUs.

<a id="inference-engines"></a>

### 3.2 Inference engines

#### Datacenter serving

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput, memory-efficient LLM serving engine with continuous batching and PagedAttention.
- [Inside vLLM](https://www.aleksagordic.com/blog/vllm) - Walkthrough of vLLM internals, from paged attention, continuous batching, prefix caching and speculative decoding to multi-GPU and multi-node serving.
- [SGLang](https://github.com/sgl-project/sglang) - High-performance serving framework for language and multimodal models.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA library and runtime for optimized LLM inference on NVIDIA GPUs.
- [Triton Inference Server](https://github.com/triton-inference-server/server) - NVIDIA's multi-framework model server, with TensorRT-LLM, vLLM, PyTorch and ONNX backends.
- [LMDeploy](https://github.com/InternLM/lmdeploy) - Toolkit for compressing, deploying and serving LLMs, built on its own TurboMind and PyTorch inference engines.
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - Fast LLM inference engine written in Rust.

#### Local and desktop

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - LLM inference in C/C++ for CPUs and GPUs, using the GGUF model format.
- [Ollama](https://github.com/ollama/ollama) - Downloads and runs open models locally behind a simple API.
- [llamafile](https://github.com/mozilla-ai/llamafile) - Packages a model and llama.cpp into a single executable that runs on most operating systems.
- [LocalAI](https://github.com/mudler/LocalAI) - Self-hosted, OpenAI-compatible server for running LLM, image and audio models locally.
- [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) - Consumer-GPU engine exploiting activation sparsity across GPU and CPU.
- [AirLLM](https://github.com/lyogavin/airllm) - Layer-by-layer loading that runs large models on small GPUs at the cost of latency.

#### Apple Silicon

- [MLX](https://github.com/ml-explore/mlx) - Array framework for machine learning on Apple Silicon.
- [vllm-metal](https://github.com/vllm-project/vllm-metal) - Community plugin for running vLLM on Apple Silicon GPUs.
- [On-Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama) - Apple's walkthrough of optimizing an LLM for on-device Core ML inference.

#### Mobile and edge

- [ExecuTorch](https://github.com/pytorch/executorch) - PyTorch runtime for on-device inference on mobile and embedded hardware.
- [MLC LLM](https://github.com/mlc-ai/mlc-llm) - Compiler-based deployment of LLMs to phones, browsers and GPUs.
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) - Google AI Edge's open-source inference framework for running LLMs on edge devices.
- [MediaPipe LLM Inference](https://developers.google.com/edge/mediapipe/solutions/genai/llm_inference) - Google AI Edge API for running LLMs on Android, iOS and the web.

<a id="benchmarking-and-profiling"></a>

### 3.3 Benchmarking and profiling

- [GuideLLM](https://github.com/vllm-project/guidellm) - vLLM project's tool for benchmarking deployments against realistic traffic.
- [genai-bench](https://github.com/sgl-project/genai-bench) - SGLang's token-level benchmark for LLM serving endpoints.
- [AIPerf](https://github.com/ai-dynamo/aiperf) - Generative AI load testing, trace replay and SLO analysis.
- [MLPerf Inference](https://github.com/mlcommons/inference) - Reference implementations of the MLPerf Inference benchmarks.
- [Vidur](https://github.com/microsoft/vidur) - Simulator that estimates LLM serving performance across configurations without running them.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) - NVIDIA's system-wide timeline profiler for CPU, GPU and communication activity.
- [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/) - AMD's system-wide timeline profiler for CPU and GPU activity.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) - Profiling CUDA kernels and interpreting hardware metrics.
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) - Kernel-level performance analysis for AMD Instinct GPUs.

<a id="kernel-libraries"></a>

### 3.4 Kernel libraries, languages and compilers

- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA C++ templates and Python DSLs for high-performance GEMM and related kernels.
- [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) - C++ and Python API for cuDNN graph operations, including fused attention.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Official FlashAttention kernels for NVIDIA and AMD GPUs.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - Attention, sampling and MoE kernels for LLM serving, used by vLLM and SGLang.
- [AITER](https://github.com/ROCm/aiter) - AI Tensor Engine for ROCm: optimized inference operators for AMD GPUs.
- [Composable Kernel](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel) - AMD's templated library for fused GEMM, attention and other operators.
- [Triton](https://github.com/triton-lang/triton) - Language and compiler for writing GPU kernels in Python.
- [Learn by Doing: TorchInductor Reduction Kernels](https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-Reduction/) - How TorchInductor generates reduction kernels.
- [Apache TVM](https://github.com/apache/tvm) - Open machine learning compiler framework.

<a id="gemm-kernels"></a>

### 3.5 GEMM kernels and tile programming

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - Step-by-step path from a naive kernel to near-cuBLAS performance.
- [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - Worklog of Hopper-specific matmul optimizations.
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) - From PTX/SASS to warp tiling and asynchronous tensor-core pipelines.
- [Categorical Foundations for CuTe Layouts](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) - The mathematics behind CuTe's layout algebra.
- [Demystifying Numeric Conversions in CuTeDSL](https://veitner.bearblog.dev/demystifying-numeric-conversions-in-cutedsl/) - Type-conversion primitives in CuTeDSL.
- [Blackwell Pipelining with CuTeDSL](https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/) - Overlapping work with CuTeDSL asynchronous pipelines on Blackwell.

<a id="inference-techniques"></a>

## 4. Inference techniques

<a id="quantization"></a>

### 4.1 Quantization

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Illustrated introduction to LLM quantization.
- [Post Training Quantization](https://liyuan24.github.io/writings/2026_01_06_post_training_quantization.html) - Walkthrough of post-training weight quantization.
- [Quantization-Aware Training for Large Language Models with PyTorch](https://pytorch.org/blog/quantization-aware-training/) - Recovering accuracy lost to quantization by simulating it during training.
- [LLM Compressor](https://github.com/vllm-project/llm-compressor) - vLLM project's library for producing GPTQ, AWQ, FP8 and other quantized checkpoints.
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) - Quantization, pruning, distillation and sparsity for deployment with TensorRT-LLM, vLLM and SGLang.
- [torchao](https://github.com/pytorch/ao) - PyTorch-native quantization and sparsity for training and inference.
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - 8-bit and 4-bit quantization for PyTorch, widely used for QLoRA.
- [GPTQModel](https://github.com/ModelCloud/GPTQModel) - GPTQ quantization toolkit with CUDA, ROCm, XPU and CPU backends.
- [AutoRound](https://github.com/intel/auto-round) - Intel's sign-gradient post-training quantization for low-bit LLMs and VLMs.

<a id="speculative-decoding"></a>

### 4.2 Speculative decoding

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/html/2211.17192v2) - The original draft-target speculative decoding paper.
- [EAGLE](https://github.com/SafeAILab/EAGLE) - Official implementation of the EAGLE-1, EAGLE-2 and EAGLE-3 speculative decoding methods.
- [Medusa](https://github.com/FasterDecoding/Medusa) - Multiple decoding heads for candidate token generation.
- [Speculators: Production-Ready Speculative Decoding](https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding) - Why the speculators library standardizes draft-model formats.

<a id="kv-cache"></a>

### 4.3 KV-cache reuse, routing and offload

- [Prompt Caching](https://ngrok.com/blog/prompt-caching/) - How cached prompt tokens skip recomputation and why that makes them cheaper.
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) - Design of vLLM's hash-based prefix cache.
- [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) - Radix-tree KV-cache reuse across requests in SGLang.
- [Ray Serve: Reduce LLM Inference Latency by 60% with Custom Request Routing](https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing) - Prefix-aware routing to cut TTFT in multi-turn chat.
- [How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/) - KV-cache offload and reuse in Dynamo.
- [Inside vLLM's New KV Offloading Connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) - Offloading KV-cache to CPU DRAM and tuning host-to-device throughput.
- [LMCache](https://github.com/LMCache/LMCache) - KV-cache layer that stores and reuses KV cache across GPU, CPU, disk and remote tiers.
- [PrisKV](https://aibrix.github.io/posts/2025-11-26-priskv-intro/) - Colocated, tiered KV-cache store from the AIBrix project.
- [kvcached](https://github.com/ovg-project/kvcached) - Elastic, virtualized KV-cache that lets multiple models share a GPU.
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - KV-cache-centric serving platform behind Moonshot AI's Kimi, whose Mooncake Store pools, reuses and offloads KV cache across serving instances.

<a id="long-context"></a>

### 4.4 Long-context handling

- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) - Explains RoPE, the position encoding most context-extension methods modify.
- [RoFormer](https://arxiv.org/abs/2104.09864) - The paper that introduced rotary position embeddings (RoPE).
- [DroPE](https://pub.sakana.ai/DroPE/) - Extends context by dropping positional embeddings instead of long-context fine-tuning.
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) - Splitting long sequences across devices with blockwise attention passed around a ring.
- [Evaluating Long Context (Reasoning) Ability](https://nrehiew.github.io/blog/long_context/) - Why models degrade well before their advertised context length, measured on LongCodeEdit.

<a id="parallelism-and-communication"></a>

### 4.5 Parallelism and communication

- [How Multi-Node Inference Works for Massive LLMs like DeepSeek-R1](https://www.baseten.co/blog/how-multi-node-inference-works-llms-deepseek-r1/) - Why DeepSeek-R1 needs 16 H100s across two nodes, and how multi-node serving connects them.
- [Shift Parallelism](https://arxiv.org/html/2509.16495v2) - Switching between tensor and data parallelism to balance latency and throughput.
- [vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP](https://vllm.ai/blog/2025-12-17-large-scale-serving) - Wide expert parallelism, dual-batch overlap and disaggregated serving for DeepSeek on H200 clusters.
- [NCCL](https://github.com/NVIDIA/nccl) - NVIDIA collective communication library for multi-GPU and multi-node transfers.
- [RCCL](https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl) - AMD's collective communication library for ROCm GPUs.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) - Expert-parallel communication library for MoE dispatch and combine.
- [UCCL](https://github.com/uccl-project/uccl) - GPU communication library for collectives, P2P KV transfer and expert parallelism.
- [llm-d Wide Expert Parallelism](https://llm-d.ai/docs/architecture/disaggregation/wide-expert-parallelism) - How llm-d deploys large MoE models with data and expert parallelism across nodes.

<a id="disaggregation"></a>

### 4.6 Disaggregation

- [Disaggregated Inference at Scale with PyTorch and vLLM](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/) - Running separate prefill and decode pools with vLLM.
- [Disaggregated Serving in TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog05_Disaggregated_Serving_in_TensorRT-LLM.md) - TensorRT-LLM's disaggregated serving design.
- [SGLang PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) - Configuring prefill/decode disaggregation in SGLang.
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) - Datacenter-scale distributed inference serving framework with disaggregated prefill and decode.
- [llm-d Architecture](https://llm-d.ai/docs/architecture) - Component design and topology of llm-d.
- [llm-d](https://github.com/llm-d/llm-d) - Kubernetes-native distributed LLM inference with KV-cache-aware scheduling.
- [llm-d P/D Disaggregation](https://llm-d.ai/docs/architecture/disaggregation/pd-disaggregation) - When to split prefill and decode in llm-d and how KV-cache is transferred between them.

<a id="beyond-llms"></a>

## 5. Omni-modal inference

<a id="omni-modal-inference"></a>

### 5.1 Omni-modal inference

- [What Is an Omni-Model?](https://www.nvidia.com/en-us/glossary/omni-model/) - NVIDIA definition of models that take and produce several modalities.
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) - Thinker-talker architecture that reads text, images, audio and video and streams speech back.
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) - End-to-end omni-modal model that understands text, audio, images and video and generates speech in real time.
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) - vLLM framework for serving omni-modal models.
- [vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models](https://arxiv.org/abs/2602.02204) - Paper on serving multimodal pipelines as independently scheduled stages.

<a id="vision-language-models"></a>

### 5.2 Vision-language models

- [Vision Language Models Explained](https://huggingface.co/blog/vlms) - Hugging Face introduction to VLM architectures, training and how to choose one.
- [Vision Language Models (Better, Faster, Stronger)](https://huggingface.co/blog/vlms-2025) - Hugging Face update on newer VLM trends, including any-to-any models, reasoning, small models and MoE decoders.
- [Vision Language Models](https://www.baseten.co/inference-engineering/book/06-modalities/vision-language-models/) - Inference Engineering book chapter on serving models that take images and video, and the cost of the vision encoder.
- [vLLM Multimodal Inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) - Passing images, video and audio to multimodal models in vLLM.
- [Vision Language Model Prompt Engineering Guide for Image and Video Understanding](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/) - NVIDIA guide to feeding single images, multiple images and video frames to VLMs.

<a id="speech"></a>

### 5.3 Speech

- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) - The Whisper paper.
- [Silero VAD](https://github.com/snakers4/silero-vad) - Lightweight voice activity detector for finding speech before transcription.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Whisper transcription reimplemented on CTranslate2 for faster inference.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) - Port of Whisper to C/C++ for CPU, GPU and on-device transcription.
- [Piper](https://github.com/OHF-Voice/piper1-gpl) - Fast, local neural text-to-speech engine.
- [What Is Speech-to-Speech?](https://deepgram.com/blog/what-is-speech-to-speech) - Cascaded versus end-to-end speech-to-speech pipelines and where each fits.
- [Moshi](https://github.com/kyutai-labs/moshi) - Speech-text foundation model and full-duplex dialogue framework with a streaming audio codec.

<a id="vision-and-video-analytics"></a>

### 5.4 Computer vision and video analytics

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Training, export and inference for YOLO detection, segmentation, pose and tracking models.
- [Ultralytics Export Formats](https://docs.ultralytics.com/modes/export/) - Exporting YOLO models to ONNX, TensorRT, OpenVINO, Core ML and other deployment formats.
- [Triton Inference Server with Ultralytics YOLO](https://docs.ultralytics.com/guides/triton-inference-server/) - Exporting a YOLO model to ONNX and serving it from a Triton model repository.
- [NVIDIA DeepStream SDK](https://github.com/NVIDIA/DeepStream) - GStreamer-based toolkit for multi-stream video analytics pipelines on NVIDIA GPUs and Jetson.
- [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) - Community implementation for running YOLO models inside DeepStream pipelines.
- [Savant](https://github.com/insight-platform/Savant) - Python framework for video analytics pipelines, built on DeepStream for NVIDIA GPUs and Jetson.
- [DL Streamer](https://github.com/open-edge-platform/dlstreamer) - Intel's GStreamer-based pipeline framework for video analytics on Intel CPUs, GPUs and NPUs.
- [Roboflow Inference](https://github.com/roboflow/inference) - Self-hosted server for running vision models and video-stream workflows on cloud or edge devices.

<a id="image-and-video-generation"></a>

### 5.5 Image and video generation

- [Diffusion Models](https://huggingface.co/blog/Esmail-AGumaan/diffusion-models) - Introductory explanation of how diffusion models are trained and sampled.
- [Diffusers](https://github.com/huggingface/diffusers) - Hugging Face library of diffusion models and pipelines for image, video and audio generation.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) - Node-graph interface, API and backend for running diffusion pipelines.
- [SGLang Diffusion](https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/) - SGLang's serving stack applied to image and video generation.
- [xDiT](https://github.com/xdit-project/xDiT) - Parallel inference engine for diffusion transformers, splitting image and video generation across GPUs.
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) - Inference and post-training framework for accelerated video generation.
- [vLLM-Omni Diffusion Guide](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion) - Attention backends and acceleration options for diffusion models in vLLM-Omni.

<a id="production"></a>

## 6. Production inference

<a id="containers-and-gpus"></a>

### 6.1 Containers and GPU enablement

- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) - Runtime hooks that expose NVIDIA GPUs to containers.
- [AMD Container Toolkit](https://github.com/ROCm/container-toolkit) - Runtime and tools that expose AMD GPUs to containers.
- [NVIDIA NIM](https://docs.nvidia.com/nim/) - Prebuilt, optimized inference microservice containers for NVIDIA GPUs.
- [BentoML](https://github.com/bentoml/BentoML) - Framework for packaging models as inference APIs, job queues and multi-model pipelines.
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) - Exposes NVIDIA GPUs as schedulable Kubernetes resources.
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) - Installs and manages NVIDIA drivers, device plugin and monitoring on Kubernetes.
- [AMD GPU Device Plugin](https://github.com/ROCm/k8s-device-plugin) - Exposes AMD GPUs as schedulable Kubernetes resources.
- [AMD GPU Operator](https://github.com/ROCm/gpu-operator) - Installs and manages AMD GPU drivers, device plugin and metrics on Kubernetes.

<a id="kubernetes-serving"></a>

### 6.2 Kubernetes serving platforms

- [KServe](https://github.com/kserve/kserve) - Kubernetes platform for deploying generative and predictive models across frameworks.
- [Ray](https://github.com/ray-project/ray) - Distributed compute engine whose Ray Serve library serves models at scale.
- [vLLM Production Stack](https://github.com/vllm-project/production-stack) - vLLM's reference system for Kubernetes-native, cluster-wide deployment.
- [AIBrix](https://github.com/vllm-project/aibrix) - vLLM project's Kubernetes control plane for LLM serving, with LoRA management, routing and autoscaling.
- [KAITO](https://github.com/kaito-project/kaito) - CNCF Kubernetes AI Toolchain Operator that provisions GPU nodes and deploys open models.
- [HAMi](https://github.com/Project-HAMi/HAMi) - CNCF project for sharing and isolating heterogeneous GPUs on Kubernetes.

<a id="routing-and-autoscaling"></a>

### 6.3 Gateways, routing and autoscaling

- [The Baseten Inference Stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack/) - Runtime and infrastructure layers of a production stack, including geo-aware load balancing.
- [LiteLLM](https://github.com/BerriAI/litellm) - AI gateway and SDK that calls 100+ LLM APIs in OpenAI format, with load balancing and cost tracking.
- [Agent Router (formerly Envoy AI Gateway)](https://github.com/theagentrouter/agent-router) - Unified access to generative AI services, built on Envoy Gateway.
- [Portkey Gateway](https://github.com/Portkey-AI/gateway) - Open-source AI gateway with routing, retries, fallbacks and guardrails across LLM providers.
- [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) - Kubernetes SIG project that adds model-aware routing to Gateway API through InferencePool.
- [llm-d Router](https://github.com/llm-d/llm-d-router) - llm-d's scheduler that routes requests by prefix-cache hits, load and predicted latency.
- [KEDA](https://github.com/kedacore/keda) - Event-driven Kubernetes autoscaler, often used to scale inference pods on queue depth.
- [vLLM LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora/) - Serving many LoRA adapters on one base model and routing requests to them.
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) - Programmable layer that classifies requests and routes them to the right model.
- [Autoscaling Endpoints for LLM Inference](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference) - Choosing scaling metrics, tuning scaling windows and budgeting for cold starts.
- [vLLM: Why Requests Take Hours Under Load](https://blog.dotieuthien.com/posts/vllm) - Diagnosing KV block exhaustion and queue starvation.

<a id="cloud-platforms"></a>

### 6.4 Cloud and GPU hosting platforms

#### AWS (EKS)

- [Load and Serve Models on Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/ml-inference-load-serve-model.html) - Deploying vLLM on GPUs and streaming weights from S3 with the Run:ai Model Streamer to cut cold starts.
- [EKS Best Practices: Application Scaling and Performance](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-performance.html) - Model artifact management, serving frameworks and container startup optimization.
- [GPU Inference on EKS Auto Mode](https://aws.amazon.com/blogs/containers/how-to-run-ai-model-inference-with-gpus-on-amazon-eks-auto-mode/) - GPU node pools with AWS-managed node lifecycle and patching.
- [Introducing Disaggregated Inference on AWS Powered by llm-d](https://aws.amazon.com/blogs/machine-learning/introducing-disaggregated-inference-on-aws-powered-by-llm-d/) - Running llm-d prefill/decode disaggregation and wide expert parallelism on AWS.

#### Google Cloud (GKE)

- [Deploy GKE Inference Gateway Powered by llm-d](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/deploy-gke-inference-gateway) - Managed gateway using llm-d scheduling.
- [Best Practices for Autoscaling LLM Inference Workloads on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling) - Choosing HPA signals such as queue size and batch size over GPU utilization.
- [Tuning the GKE HPA to Run Inference on GPUs](https://cloud.google.com/blog/products/containers-kubernetes/tuning-the-gke-hpa-to-run-inference-on-gpus) - Experiments comparing autoscaling metrics for LLM servers.
- [Serve an LLM Using TPUs on GKE with KubeRay](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray) - Deploying a RayCluster on a TPU node pool and serving an LLM with Ray Serve.

#### Azure (AKS)

- [Deploy an AI Model on AKS with the AI Toolchain Operator](https://learn.microsoft.com/en-us/azure/aks/ai-toolchain-operator) - Managed KAITO add-on that deploys open models with vLLM and provisions GPU nodes.
- [Use GPUs on AKS](https://learn.microsoft.com/en-us/azure/aks/use-nvidia-gpu) - Creating GPU node pools and installing the NVIDIA device plugin.
- [GPU Node Partitioning Strategies in AKS](https://learn.microsoft.com/en-us/azure/aks/concepts-gpu-partitioning) - Comparison of MIG, time-slicing and MPS for sharing GPUs.
- [Create a Multi-Instance GPU Node Pool in AKS](https://learn.microsoft.com/en-us/azure/aks/gpu-multi-instance) - Creating MIG-partitioned node pools on AKS.

#### GPU clouds

- [Serving Llama 3.1 with vLLM on Lambda](https://docs.lambda.ai/education/large-language-models/serving-llama-3-1-docker/) - Lambda's guide to running vLLM in Docker on on-demand A100 and H100 instances.
- [About CoreWeave Inference](https://docs.coreweave.com/products/inference) - CoreWeave's options for serving models: serverless, dedicated, or self-managed on its Kubernetes service.
- [vLLM on Vast.ai](https://docs.vast.ai/vllm-llm-inference-and-serving) - Vast.ai's template for serving LLMs with vLLM on marketplace GPUs.
- [Serve LLM Inference on E2E GPU Nodes](https://docs.e2enetworks.com/docs/myaccount/gpu/guides/serve-llm-inference) - E2E Networks' guide to serving an LLM on its GPU virtual machines.

#### Managed inference endpoints

- [RunPod Serverless vLLM](https://docs.runpod.io/serverless/vllm/overview) - Deploying an OpenAI-compatible vLLM endpoint on RunPod Serverless that scales with traffic.
- [Replicate: Deploy a Custom Model](https://replicate.com/docs/get-started/deploy-a-custom-model) - Packaging a model with Cog and running it as a dedicated, autoscaling deployment on Replicate.
- [Cog](https://github.com/replicate/cog) - Replicate's open-source tool for packaging models as production-ready containers.
- [E2E Networks TIR Model Endpoints](https://docs.e2enetworks.com/docs/tir/Inference/ModelEndpoints/QuickStartGuide/) - Creating managed endpoints with vLLM, SGLang or Triton runtimes on E2E's TIR platform.
- [Modal: LLM Inference with vLLM](https://modal.com/docs/examples/llm_inference) - Modal's example of serving an OpenAI-compatible vLLM server on serverless GPUs.
- [Baseten Model Deployment](https://docs.baseten.co/deployment/concepts) - Deploying models on Baseten with Truss, from a config file to an autoscaling API.
- [Truss](https://github.com/basetenlabs/truss) - Baseten's open-source framework for packaging models into deployable containers.
- [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) - Deploying Hub models on managed, autoscaling infrastructure across cloud providers.
- [Together AI Dedicated Inference](https://docs.together.ai/docs/dedicated-endpoints/overview) - Running models on dedicated, single-tenant GPU endpoints on Together AI.

<a id="observability-and-reliability"></a>

### 6.5 Observability, security and debugging

- [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) - Prometheus exporter for NVIDIA GPU metrics, built on DCGM.
- [AMD Device Metrics Exporter](https://github.com/ROCm/device-metrics-exporter) - Exports AMD GPU metrics to Prometheus and other collectors.
- [OpenTelemetry Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - Standard span and metric names for model calls, tokens and latency.
- [OpenLLMetry](https://github.com/traceloop/openllmetry) - OpenTelemetry instrumentation for LLM providers, frameworks and vector databases.
- [Helicone](https://github.com/Helicone/helicone) - Open-source LLM observability platform and gateway for request logging, cost and latency.
- Langfuse and Arize Phoenix, listed under [Evaluation](#model-strategy), also trace production LLM traffic.
- [Confidential Compute for AI Inference](https://chutes.ai/news/confidential-compute-for-ai-inference-how-chutes-delivers-verifiable-privacy-with-trusted-execution-environments) - How Chutes uses trusted execution environments for verifiable private inference on third-party GPUs.
- [Token-Level Truth](https://vllm.ai/blog/2025-12-14-halugate) - HaluGate: token-level hallucination detection in vLLM Semantic Router, checked against tool outputs and context.
- [One Token to Corrupt Them All](https://www.ai21.com/blog/vllm-debugging-mamba-bug/) - How AI21 traced a state-corruption bug in vLLM's scheduler and memory management for Mamba models.

<a id="cost-and-efficiency"></a>

### 6.6 Cost and efficiency

- [LLM Inference Benchmarking: How Much Does Your LLM Inference Cost?](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/) - NVIDIA method for turning benchmark results into cost per million tokens and total cost of ownership.
- [The Hidden Economics of LLM Inference](https://www.hebbia.com/blog/the-hidden-economics-of-llm-inference) - Queueing math and GPU utilization, and why pooled multi-tenant serving costs less than single-tenant serving.
- [Energy Use of AI Inference, Efficiency Pathways, and Test-Time Scaling](https://arxiv.org/abs/2509.20241) - Per-query energy estimates and efficiency pathways at scale.
- [GPU Inefficiency in AI Workloads](https://www.anyscale.com/blog/gpu-in-efficiency-in-ai-workloads) - Why production GPUs sit underutilized.

<a id="learning"></a>

## 7. Learning resources

<a id="courses-and-books"></a>

### 7.1 Courses and books

- [Inference Engineering](https://www.baseten.co/inference-engineering/book/) - Philip Kiely's book on building faster, cheaper and more reliable generative AI inference systems.
- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) - TPU/GPU architecture, communication and parallelism for training and inference.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) - Hugging Face's interactive guide to parallelism and communication for training LLMs on GPU clusters.
- [Hands-On LLM Serving and Optimization](https://www.oreilly.com/library/view/hands-on-llm-serving/9798341621480/) - Chi Wang and Peiheng Hu's O'Reilly book on hosting and optimizing LLMs at scale.

<a id="github-repositories"></a>

### 7.2 GitHub repositories

- [100 Days of LLM Inference](https://github.com/elizabetht/100-days-of-inference) - Day-by-day runnable experiments that follow the Inference Engineering book, from KV cache and kernels to quantization, speculative decoding and disaggregated serving.
- [LLM Inference Engineering](https://github.com/amitshekhariitbhu/llm-inference-engineering) - Step-by-step blog series from autoregressive generation, prefill and decode to KV cache, PagedAttention, continuous batching, vLLM, SGLang and GPUs.
- [AI Inference Resources](https://github.com/aerlabsAI/ai-inference-resources) - AER Labs' tiered reading guide covering LLM serving, GPU kernels, attention, quantization, distributed inference and production deployment.

<a id="talks"></a>

### 7.3 Talks

- [The Rise of vLLM](https://www.youtube.com/watch?v=WLl8D1nyaW8) - How vLLM grew from a research project into a widely used engine.
- [SGLang: An Efficient Open-Source Framework for Large-Scale LLM Serving](https://www.youtube.com/watch?v=w9-AYqIhHRo) - Ray Summit 2025 talk on SGLang's architecture and production use.
- [How to Become an Inference Engineer](https://www.youtube.com/watch?v=LUGxx1XNqcM) - Baseten talk on the skills and path to becoming an inference engineer.
- [Inference OSS Ecosystem featuring vLLM](https://www.nvidia.com/en-us/on-demand/session/other25-dynamoday06/) - Disaggregation and wide-EP for MoE on GB200 racks.
- [Inside the MAX Framework](https://www.youtube.com/watch?v=WK5dVQ8vhbU) - Modular meetup walkthrough of the MAX inference stack.
- [Introduction to LLM Serving with SGLang](https://www.youtube.com/watch?v=Ahtaha9fEM0) - AI Engineer workshop by Philip Kiely and Yineng Zhang on serving LLMs with SGLang.

<a id="contributing"></a>

## Contributing

Read [the contribution guide](CONTRIBUTING.md) before submitting. Use the issue templates to propose a topic.

<a id="license"></a>

## License

This repository's original content is licensed under the [MIT License](LICENSE). Linked projects, documentation, papers, images and code retain their respective licenses.

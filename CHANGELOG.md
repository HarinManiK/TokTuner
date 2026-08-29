# Changelog

All notable changes to `TokTuner` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-08-30

### Initial Public Release

#### Features
- **Analytical 0-1 Knapsack Planner**: Mathematically derives optimal GPU residency from token read frequencies ($f_i$).
- **Sub-Layer Offloading**: Automatic emission of `-ncmoe` (for MoE architectures) and `-ncffn` (for dense models to preserve KV cache in VRAM).
- **Exact KV Cache Engine**: Complete layer-by-layer modeling for MHA, GQA, MQA, DeepSeek MLA (latent $+ \text{RoPE}$), Sliding Window (SWA), Recurrent SSM, and Gemma-style heterogeneous masks.
- **Quant-Agnostic Tensor Sizing**: Sizing derived from GGUF data section offset deltas, working across all present and future quantization types without type tables.
- **Multi-Shard GGUF Reader**: Automatic detection and merging of split models (`name-00001-of-00009.gguf`).
- **Hardware Probing**: Real-time detection of NVIDIA GPU VRAM, free RAM, pagefile/swap usage, and vendor-aware CPU core topology (Intel P-cores vs AMD Zen-c).
- **Driver Reserve & Overhead Calibration**: $2.5\%$ driver reserve + dynamic context scratch buffer budgeting to prevent silent KV spilling.
- **Leftover VRAM Absorption**: Automatic micro-batch (`-ub`) tuning to boost prompt prefill throughput using residual slack memory.
- **Interfaces**:
  - Full-featured CLI with `--ctx`, `--all`, `--quiet`, and `--safety` flags.
  - Zero-dependency Tkinter desktop GUI with copyable commands and `.bat` / `.sh` script generation.
- **Cross-Platform Compatibility**: Full platform support for Windows and Linux with host-shell quoting.
- **Validation Suite**: 36 synthetic universality tests, 13 ground-truth measured allocations, and 356 adversarial audit checks.

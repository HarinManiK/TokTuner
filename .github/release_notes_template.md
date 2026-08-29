# TokTuner v0.1.0 - Initial Release 🚀

`TokTuner` computes the mathematically fastest `llama.cpp` flags for your machine in **under 50 milliseconds**, without running the model.

### 🌟 Highlights
- **Analytical 0-1 Knapsack Optimizer**: Derives optimal GPU tensor placement from token read frequencies ($f_i$).
- **Sub-Layer Offloading**: Automatically emits `-ncmoe` (for MoE models) and `-ncffn` (for dense models) to prevent exiling the KV cache from VRAM.
- **Deep Architecture Support**: Exact KV cache modeling for DeepSeek MLA (latent $+ \text{RoPE}$), Sliding Window Attention (SWA), Recurrent SSM (Qwen 3.5), and Gemma-style heterogeneous layer masks.
- **Quant-Agnostic Tensor Sizing**: Sizing computed from GGUF data offset deltas, working across all present and future quantization formats without type tables.
- **Driver Reserve & Buffer Calibration**: Budgets calibrated $2.5\%$ driver reserve and dynamic workspace buffers to completely eliminate silent KV cache spilling.
- **Leftover VRAM Absorption**: Automatically tunes `-ub` (micro-batch size) to accelerate prompt prefill throughput.
- **Zero Runtime Dependencies**: 100% Python standard library.

---

### 📦 Downloads & Installation

#### 🪟 Windows Standalone Binary
Download `toktuner.exe` below, open it, select your `llama-server.exe` folder and model file, and click **Plan**.

#### 🐧 Linux Standalone Binary (x86_64)
Download `toktuner-linux-x86_64` below:
```bash
chmod +x toktuner-linux-x86_64

# Open Desktop GUI
./toktuner-linux-x86_64 --gui

# Or run in terminal (headless servers & desktops)
./toktuner-linux-x86_64 --model your_model.gguf --ctx 4096
```

#### 🐍 Python pip / Local Installation
```bash
git clone https://github.com/HarinManiK/TokTuner.git
cd TokTuner
pip install -e .

# Run
toktuner --model your_model.gguf --ctx 4096
```

---

### 🧪 Validation & Correctness
- **Universality Checks**: 36 synthetic architectures and mathematical invariants passed.
- **Ground-Truth Allocations**: 13 real measured hardware allocations validated (worst-case error: 0.27 GB).
- **Adversarial Audit**: 356 checks passed against degenerate models and `llama-server` flag syntax.

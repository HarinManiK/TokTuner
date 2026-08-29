# Architecture & Engineering Reference: `TokTuner`

This document outlines the internal architecture, mathematical formulations, and engineering principles behind `TokTuner`.

---

## 1. System Architecture Overview

```
                        +--------------------------------+
                        |       Target .gguf File        |
                        +--------------------------------+
                                        |
                                        v
+------------------------+    +--------------------+    +-----------------------+
|   hardware.py          |    |      gguf.py       |    |      tensors.py       |
|  - nvidia-smi GPU info |    | - Header-only read |    | - Offset-delta sizing |
|  - RAM & Swap state    |    | - Shard traversal  |    | - Role classification |
|  - CPU topology/vendor |    | - Metadata dict    |    |   (Always vs Movable) |
+------------------------+    +--------------------+    +-----------------------+
            |                           |                           |
            |                           v                           |
            |                 +--------------------+                |
            |                 |      arch.py       |                |
            |                 | - KV cache profile |                |
            |                 | - MLA / SWA / SSM  |                |
            |                 | - Context limits   |                |
            |                 +--------------------+                |
            |                           |                           |
            +-------------------+       |       +-------------------+
                                |       |       |
                                v       v       v
                        +--------------------------------+
                        |            plan.py             |
                        | - 0-1 Knapsack solver          |
                        | - Driver reserve calibration   |
                        | - Compute workspace buffers    |
                        | - -ub leftover absorption      |
                        | - Shell-safe flag generation   |
                        +--------------------------------+
                                        |
                        +---------------+---------------+
                        |                               |
                        v                               v
            +-----------------------+       +-----------------------+
            |        cli.py         |       |        gui.py         |
            | - Console CLI output  |       | - Pure Tkinter GUI    |
            | - Context ladders     |       | - .bat / .sh exporter |
            +-----------------------+       +-----------------------+
```

---

## 2. Mathematical Formulation: 0-1 Knapsack of Memory Bandwidth

### Memory-Bandwidth Decode Model
During autoregressive generation (token decode), computation is strictly memory-bandwidth bound. For each generated token, every weight tensor that resides in GPU VRAM is streamed at GPU memory bandwidth $\text{BW}_{\text{gpu}}$, while every weight tensor residing in CPU system RAM is streamed over the PCIe bus / host memory channels at $\text{BW}_{\text{cpu}}$.

The time required to generate one token is:

$$t = \sum_i \left[ x_i \frac{b_i}{\text{BW}_{\text{gpu}}} + (1 - x_i) \frac{b_i}{\text{BW}_{\text{cpu}}} \right] + c$$

Where:
* $x_i \in \{0, 1\}$ is the placement indicator ($1$ if tensor $i$ is in VRAM, $0$ if in RAM).
* $b_i$ is the expected number of bytes streamed for tensor $i$ per token.
* $s_i$ is the static memory footprint of tensor $i$ in bytes.
* $f_i \in [0, 1]$ is the **read frequency** (activation probability per token), such that $b_i = s_i \cdot f_i$.
* $c$ represents kernel launch overhead and fixed latency.

### Derivation of Value Density
Minimizing time per token $t$ is equivalent to maximizing the total time saved by GPU residency:

$$\Delta t(x) = \sum_i x_i \cdot b_i \left( \frac{1}{\text{BW}_{\text{cpu}}} - \frac{1}{\text{BW}_{\text{gpu}}} \right) = \left( \frac{1}{\text{BW}_{\text{cpu}}} - \frac{1}{\text{BW}_{\text{gpu}}} \right) \sum_i x_i \cdot s_i \cdot f_i$$

Because $\left( \frac{1}{\text{BW}_{\text{cpu}}} - \frac{1}{\text{BW}_{\text{gpu}}} \right) > 0$ is a constant across all tensors on a given machine, the optimization problem simplifies to:

$$\text{maximize } \sum_i x_i \cdot s_i \cdot f_i \quad \text{subject to } \sum_i x_i \cdot s_i \le M_{\text{usable}}$$

The value density of placing tensor $i$ into VRAM is:

$$\rho_i = \frac{\text{Value}_i}{\text{Weight}_i} = \frac{s_i \cdot f_i}{s_i} = f_i$$

**Theorem:** *The optimal VRAM allocation policy is greedy selection in descending order of token read frequency $f_i$.*

### Practical Implications
1. **Mixture of Experts (MoE)**:
   * Attention weights, Dense FFNs, Shared Experts, and KV Cache are accessed on every token: $f = 1.0$.
   * Routed Experts (e.g. top-$k$ of $E$, such as 8 of 256) are accessed with frequency:
     $$f = \frac{k}{E} = \frac{8}{256} = 0.03125$$
   * Tensors with $f = 1.0$ are **$32\times$ more valuable per byte of VRAM** than routed experts.
   * `TokTuner` prioritizes placing all $f=1.0$ tensors on GPU first, using `-ncmoe` to offload low-frequency routed experts to system RAM.
2. **Dense Models (`-ncffn` vs `-ngl`)**:
   * Whole-layer offloading (`-ngl`) evicts whole layers including their attention weights and KV cache.
   * Because KV cache has $f = 1.0$, exiling it to system RAM introduces continuous PCIe bus streaming on every token.
   * `-ncffn` leaves all attention and KV in VRAM and offloads only FFN weight, maximizing memory bandwidth utilization.

---

## 3. KV Cache Modeling (`toktuner/arch.py`)

Different architectures allocate KV cache memory according to distinct mathematical formulas:

### A. Standard Attention (MHA, GQA, MQA)
$$\text{Bytes/token/layer} = n_{\text{head\_kv}} \times (\text{key\_length} + \text{value\_length}) \times \text{element\_bytes}$$

### B. DeepSeek Multi-Head Latent Attention (MLA)
MLA compresses keys and values into a shared low-rank latent vector:
$$\text{Bytes/token/layer} = (\text{kv\_lora\_rank} + \text{rope\_dimension}) \times \text{element\_bytes}$$
*Standard GQA formulas overestimate MLA cache by $3\text{--}5\times$.*

### C. Sliding Window Attention (SWA)
For layers with window $W$:
$$\text{Total Bytes} = n_{\text{head\_kv}} \times (\text{key\_len} + \text{value\_len}) \times \text{element\_bytes} \times \min(n_{\text{ctx}}, W)$$
*Once context exceeds $W$, SWA memory becomes constant.*

### D. Hybrid Interleaved Architectures (Gemma 4, Qwen 3.5+)
* Gemma 4 uses a heterogeneous pattern (e.g., 5 sliding layers with 8 heads to 1 global layer with 2 heads).
* Qwen 3.5 interleaves recurrent SSM blocks ($d_{\text{state}} \times d_{\text{inner}}$) with full attention layers on a fixed interval (`full_attention_interval`).
* `TokTuner` constructs a per-layer mask and computes the exact piecewise sum.

---

## 4. Tensor Sizing via Offset Deltas (`toktuner/tensors.py`)

Rather than relying on a brittle lookup table of ggml quantization IDs, `TokTuner` computes exact tensor sizes from the GGUF data section offsets:

$$\text{Size}(T_i) = \text{Offset}(T_{i+1}) - \text{Offset}(T_i)$$

For the final tensor $T_N$:
$$\text{Size}(T_N) = \text{DataSectionSize} - \text{Offset}(T_N)$$

This guarantees exact sizing across all quantizations (including custom, private, or newly invented formats like `MXFP4` or experimental IQ variants).

---

## 5. Hardware Budget & Driver Calibration (`toktuner/plan.py`)

### Usable VRAM Formulation
$$M_{\text{usable}} = M_{\text{total}} - M_{\text{driver\_reserve}} - M_{\text{used\_by\_others}} - M_{\text{safety}}$$

Where:
* $M_{\text{driver\_reserve}} = \max(256\text{ MB}, M_{\text{total}} \times 0.025)$ represents OS display compositor, page tables, and driver residency.
* $M_{\text{cuda\_context}} \approx 180\text{ MB}$ represents runtime kernels and CUDA context.

### Compute Scratch Buffers
$$M_{\text{buffers}} = (\text{ubatch} \times n_{\text{embd}} \times 2 \times 12) + \left( \text{ubatch} \times n_{\text{head}} \times \frac{n_{\text{ctx}}}{1024} \times 320 \right) + 128\text{ MB}$$

### Leftover VRAM Absorption (`-ub`)
Any remaining slack after the last whole layer is placed cannot fit another layer. `TokTuner` greedily absorbs this slack into micro-batch size ($\text{ubatch} \in \{768, 1024, 1536, 2048\}$), maximizing prompt prefill throughput without placement risk.

---

## 6. CPU Core Topology & Vendor Rules (`toktuner/hardware.py`)

During offloaded inference, CPU thread contention on the memory bus is critical:
* **Intel Hybrid (P/E Cores)**: E-cores have lower IPC and no SMT, creating memory bus contention. `TokTuner` pins `-t` strictly to physical **P-cores**.
* **AMD Zen 4c/5c**: Compact cores share the identical microarchitecture and IPC with standard cores. `TokTuner` includes **all physical cores**.

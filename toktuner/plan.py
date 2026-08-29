"""The planner: compute the best llama.cpp command, analytically.

No benchmarking. No model execution. Everything here is arithmetic over the
GGUF header and the machine's reported memory.

The optimisation
----------------
Decode is memory-bandwidth bound, so time per token is

    t = SUM_i [ x_i b_i / BW_gpu + (1 - x_i) b_i / BW_cpu ] + c

with x_i in {0,1} for "tensor group i is VRAM-resident" and b_i the bytes read
per token. Minimising t equals maximising the time saved by GPU residency:

    maximise   SUM_i x_i * s_i * f_i        subject to   SUM_i x_i * s_i <= M

    s_i  size in bytes            f_i  reads per token, 0..1
    M    VRAM - KV - buffers - CUDA context

The bandwidth constants cancel, leaving a 0-1 knapsack whose value density is

    (s_i * f_i) / s_i  =  f_i

The value per byte of VRAM is exactly the read frequency. So: sort by reads
per token, fill VRAM in that order. For a top-k-of-E mixture of experts that
puts attention, dense FFN, shared experts and the KV cache (f = 1) ahead of
routed experts (f = k/E), typically by 32x.

Two consequences that are derived rather than assumed:

  * Dense models should use -ncffn, not -ngl. Both keep the same byte count
    on the GPU, but -ngl exiles whole layers including their KV cache, and KV
    is read every single token. -ncffn keeps all attention and KV resident
    and offloads only FFN weight.

  * The CPU block should be contiguous. All expert layers share one read
    frequency, so the knapsack cannot distinguish them; the tie breaks on
    switching cost, and one contiguous block means one boundary crossing.
    That is exactly what -ncmoe and -ncffn encode.
"""

from __future__ import annotations

import platform
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from . import arch, tensors
from .gguf import ModelInfo


def _quote(part: str | Path) -> str:
    """Quote one argument for the shell the user will actually paste into.

    shlex.quote is POSIX and wraps in single quotes. Neither cmd.exe nor
    PowerShell treats a single-quoted string as an executable path, so a
    llama.cpp install under "C:\\Program Files" - the default location -
    produces a command that simply will not run. Windows needs double quotes.

    Only arguments that need quoting get it, so the ordinary case stays a
    clean single line.
    """
    s = str(part)
    if not s or not any(c in s for c in ' \t"\''):
        return s
    if platform.system() == "Windows":
        return '"' + s.replace('"', r'\"') + '"'
    return shlex.quote(s)

GiB = 2 ** 30
MiB = 2 ** 20

# CUDA context, kernels and driver allocations. Calibrated against 13 measured
# configurations spanning a 64x context range: an initial 500 MiB produced a
# constant +0.39 GB overestimate, so the true figure is near 130 MiB. We keep
# ~50 MiB of deliberate conservatism, because the failure modes are not
# symmetric - overestimating rejects a workable config, underestimating
# recommends one that silently spills KV into system RAM at half speed.
CUDA_CONTEXT_MIB = 180

# Compute buffers scale with the micro-batch and mildly with context.
DEFAULT_UBATCH = 512


@dataclass
class Budget:
    vram_total_mib: int
    vram_used_by_others_mib: int
    ram_total_mib: int
    ram_available_mib: int
    safety_mib: int = 0
    physical_cores: int = 0

    @property
    def driver_reserve_mib(self) -> int:
        """VRAM the card reports but never actually hands over.

        A GPU never lets a process allocate its full advertised memory: the
        driver keeps context, page tables and display surfaces outside what
        nvidia-smi attributes to any process. On the validation card - 11.94 GB
        advertised - allocations topped out at 11.64 GB across every measured
        configuration, a 0.30 GB shortfall, and nvidia-smi reported 0 in use at
        the time.

        Budgeting the advertised figure therefore produces plans that fit on
        paper and spill in practice, which is the worst outcome available
        because llama.cpp does not error - it quietly moves the KV cache to
        system RAM and halves throughput.

        2.5% tracks the observed 0.30/11.94 and scales sensibly to larger
        cards, with a floor for small ones.
        """
        return max(256, int(self.vram_total_mib * 0.025))

    @property
    def vram_bytes(self) -> float:
        """VRAM we may actually use.

        Beyond the driver reserve, only what other processes already hold is
        withheld. Idle VRAM buys nothing: with a fixed context the KV cache is
        allocated at load and never grows, so there is no later demand to
        protect against.
        """
        free = (self.vram_total_mib
                - self.driver_reserve_mib
                - self.vram_used_by_others_mib
                - self.safety_mib)
        return max(0, free) * MiB

    @property
    def ram_bytes(self) -> float:
        return max(0, self.ram_available_mib) * MiB


def compute_buffers(model: ModelInfo, n_ctx: int, n_ubatch: int) -> float:
    """Scratch space for one forward pass.

    Scales with the micro-batch, which is the real unit of GPU work, and with
    context because attention scratch tracks sequence length.

    The context coefficient is calibrated against measurement, not derived.
    On the validation model the non-KV footprint grew 0.55 GB between 4k and
    262k; an earlier coefficient of 64 predicted only 0.125 GB and left the
    262k plan 0.63 GB short. That error points the wrong way - underestimating
    buffers produces a plan that fits on paper and spills in practice - so the
    coefficient is set to reproduce the observed growth with a little margin.
    """
    base = n_ubatch * max(1, model.n_embd) * 2 * 12
    attn = n_ubatch * max(1, model.n_head) * (n_ctx / 1024) * 320
    return base + attn + 128 * MiB


@dataclass
class Plan:
    model: ModelInfo
    n_ctx: int
    kv_type: str
    knob: str
    n_offload: int
    layers_on_gpu: int
    layers_total: int
    override: str = ""

    vram_bytes: float = 0.0
    ram_bytes: float = 0.0
    kv_bytes: float = 0.0
    always_bytes: float = 0.0
    offload_gpu_bytes: float = 0.0
    offload_cpu_bytes: float = 0.0
    slack_bytes: float = 0.0

    read_gpu_bytes: float = 0.0    # per generated token
    read_cpu_bytes: float = 0.0

    n_batch: int = 2048
    n_ubatch: int = DEFAULT_UBATCH
    n_threads: int = 0             # 0 = leave llama.cpp's default alone

    feasible: bool = True
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def gpu_read_share(self) -> float:
        total = self.read_gpu_bytes + self.read_cpu_bytes
        return (self.read_gpu_bytes / total) if total else 0.0

    def flags(self) -> list[str]:
        f = ["-ngl", "99", "-c", str(self.n_ctx), "-fa", "on", "--parallel", "1"]
        # -ub is only emitted when it differs from llama.cpp's default, so the
        # command stays as short as it can honestly be.
        if self.n_ubatch != DEFAULT_UBATCH:
            f += ["-b", str(self.n_batch), "-ub", str(self.n_ubatch)]
        if self.n_threads:
            f += ["-t", str(self.n_threads)]
        if self.n_offload > 0:
            f += [self.knob, str(self.n_offload)]
        if self.override:
            f += ["-ot", self.override]
        if self.kv_type != "f16":
            f += ["-ctk", self.kv_type, "-ctv", self.kv_type]
        f += ["--load-mode", self.load_mode]
        if self.model.has_mtp:
            # Built-in multi-token prediction. Measured +68% on a model that
            # ships the heads; n-max 2 beat 3 and 4 in that testing.
            f += ["--spec-type", "draft-mtp,ngram-mod", "--spec-draft-n-max", "2"]
        return f

    @property
    def load_mode(self) -> str:
        """dio avoids a second resident copy of the weights.

        mmap keeps the file mapped after the weights are copied to VRAM, so
        the model occupies RAM and VRAM at once - measured at 16.7 GB of RAM
        versus 1.6 GB for the same configuration. dio reads directly and skips
        that. It is only viable when the CPU-resident share actually fits in
        RAM; past that, mmap's paging is the only thing that makes the model
        run at all.
        """
        return "dio" if self.ram_bytes <= self._ram_budget else "mmap"

    _ram_budget: float = 0.0

    def command(self, server_exe: str | Path) -> str:
        parts = [str(server_exe), "-m", str(self.model.path), *self.flags()]
        return " ".join(_quote(p) for p in parts)


def build(model: ModelInfo, n_ctx: int, budget: Budget,
          *, n_ubatch: int = DEFAULT_UBATCH, prefer_safety: bool = False) -> Plan:
    """Compute the plan. Pure arithmetic."""
    n_ctx, clamp_note = arch.effective_context(model, n_ctx)
    kvp = arch.analyse_kv(model)
    groups = tensors.classify(model)

    warnings: list[str] = []
    notes: list[str] = list(kvp.notes)
    if groups.coverage < 0.90:
        warnings.append(
            f"only {groups.coverage*100:.0f}% of this file's bytes could be "
            f"attributed to known tensor roles - the plan may misjudge how much "
            f"can be offloaded")
    if groups.unknown_bytes and model.file_size > 2 * GiB:
        warnings.append(
            "almost nothing in this model was recognised as offloadable. Its "
            "tensor naming is unfamiliar, so the plan can only place whole "
            "layers with -ngl rather than splitting FFN or experts")
    if clamp_note:
        notes.append(clamp_note)
    if not kvp.confident:
        warnings.append("KV cache size could not be determined confidently for "
                        "this architecture; treat the plan as approximate")

    vram_budget = budget.vram_bytes
    if prefer_safety:
        vram_budget -= 0.4 * GiB

    overhead = compute_buffers(model, n_ctx, n_ubatch) + CUDA_CONTEXT_MIB * MiB

    # KV is read every token, so it outranks every offloadable byte. Step its
    # precision down only when the alternative is not fitting at all.
    kv_type = None
    for cand in arch.KV_LADDER:
        need = kvp.total_bytes(n_ctx, cand) + overhead + groups.always_bytes
        if need <= vram_budget:
            kv_type = cand
            break
    if kv_type is None:
        kv_type = "q4_0"
        kvb = kvp.total_bytes(n_ctx, kv_type)
        p = Plan(model=model, n_ctx=n_ctx, kv_type=kv_type, knob=groups.knob,
                 n_offload=len(groups.layers), layers_on_gpu=0,
                 layers_total=len(groups.layers),
                 kv_bytes=kvb, always_bytes=groups.always_bytes,
                 offload_cpu_bytes=groups.movable_bytes,
                 feasible=False, warnings=warnings, notes=notes)
        # Distinguish "this card is too small" from "this card is busy right
        # now". They call for completely different actions, and conflating
        # them sends the user off to requantise a model when all they needed
        # was to close the server they already had running.
        occupied_gb = budget.vram_used_by_others_mib / 1024
        if occupied_gb > 0.5:
            p.warnings.append(
                f"{occupied_gb:.1f} GB of this GPU's {budget.vram_total_mib/1024:.1f} GB "
                f"is already in use by another process - most likely a model "
                f"server that is still running - leaving only "
                f"{vram_budget/GiB:.1f} GB. Close it and re-run; the card "
                f"itself is not the problem.")
        else:
            p.warnings.append(
                f"even with everything offloaded, the always-resident tensors "
                f"({groups.always_bytes/GiB:.1f} GB) plus KV cache "
                f"({kvb/GiB:.1f} GB) exceed the {vram_budget/GiB:.1f} GB of "
                f"usable VRAM. Reduce the context, or use a smaller "
                f"quantisation of this model.")
        p._ram_budget = budget.ram_bytes
        return p

    if kv_type != "f16":
        notes.append(f"KV cache quantised to {kv_type} - f16 would not fit "
                     f"{n_ctx:,} tokens on this GPU")

    kvb = kvp.total_bytes(n_ctx, kv_type)
    free_for_offload = vram_budget - kvb - overhead - groups.always_bytes

    # Greedy fill. Layers share a read frequency so the knapsack is
    # indifferent between them; keep the CPU block contiguous from layer 0 to
    # minimise CPU<->GPU boundary crossings.
    indices = groups.layers
    on_gpu: list[int] = []
    used = 0
    for idx in reversed(indices):
        size = groups.movable_per_layer[idx]
        if used + size <= free_for_offload:
            on_gpu.append(idx)
            used += size
        else:
            break
    on_gpu.sort()
    n_offload = len(indices) - len(on_gpu)

    # Sub-layer refinement is deliberately NOT emitted.
    #
    # -ncmoe/-ncffn move whole layers, so the last partially-affordable layer
    # leaves 0.2-0.45 GB unused, and an -ot override could rescue its down
    # projection. That was built, and then removed, because the trade is bad:
    #
    #   Several architectures fuse the gate and up projections into one tensor
    #   (Gemma 4 ships ffn_gate_up_exps). Pulling ffn_down_exps onto the GPU
    #   while its fused partner stays in system RAM splits a pair the compute
    #   graph expects to find together. A CUDA allocation failure at load is
    #   one plausible consequence, and exactly what was observed in the field.
    #
    # Whether a given build tolerates that split cannot be established without
    # launching the model, which this tool does not do. Trading a possible
    # hard crash for a fraction of a gigabyte is not worth it - and the
    # leftover is not wasted anyway: it goes to -ub below, buying prompt
    # throughput with no placement risk.
    override = ""

    # ---- spend whatever VRAM is left on prompt-processing speed --------
    #
    # -ub fixes the shape of every intermediate tensor, so it sets the size of
    # the compute workspace. Once no further expert layer fits, the remaining
    # VRAM would otherwise sit idle - raising -ub converts it into prefill
    # throughput at no cost to generation, which -ub does not affect at all.
    #
    # Capped at 2048: measurements across several backends put the knee at
    # 1024-2048, with 4096 adding nothing but allocator pressure.
    n_ubatch = DEFAULT_UBATCH
    leftover_after = vram_budget - (groups.always_bytes + used + kvb
                                    + compute_buffers(model, n_ctx, n_ubatch)
                                    + CUDA_CONTEXT_MIB * MiB)
    for cand in (2048, 1536, 1024, 768):
        extra = (compute_buffers(model, n_ctx, cand)
                 - compute_buffers(model, n_ctx, DEFAULT_UBATCH))
        if extra <= leftover_after:
            n_ubatch = cand
            break
    overhead = compute_buffers(model, n_ctx, n_ubatch) + CUDA_CONTEXT_MIB * MiB

    vram = groups.always_bytes + used + kvb + overhead
    cpu_bytes = groups.movable_bytes - used
    ram = cpu_bytes + 700 * MiB

    read_gpu = groups.always_bytes + used * groups.movable_freq
    read_cpu = cpu_bytes * groups.movable_freq

    plan = Plan(
        model=model, n_ctx=n_ctx, kv_type=kv_type, knob=groups.knob,
        n_offload=n_offload, layers_on_gpu=len(on_gpu),
        layers_total=len(indices), override=override,
        vram_bytes=vram, ram_bytes=ram, kv_bytes=kvb,
        always_bytes=groups.always_bytes, offload_gpu_bytes=used,
        offload_cpu_bytes=cpu_bytes,
        slack_bytes=max(0.0, budget.vram_bytes - vram),
        read_gpu_bytes=read_gpu, read_cpu_bytes=read_cpu,
        n_batch=max(2048, n_ubatch), n_ubatch=n_ubatch,
        # Threads only matter when the CPU actually holds weights. With
        # everything resident on the GPU, pinning them adds nothing and risks
        # overriding a better default.
        n_threads=(budget.physical_cores if (cpu_bytes > 0 and
                                             budget.physical_cores) else 0),
        feasible=True, warnings=warnings, notes=notes)
    plan._ram_budget = budget.ram_bytes

    # --- accountability -----------------------------------------------------
    per_layer_gb = (groups.movable_bytes / max(1, len(indices))) / GiB
    # Filling the budget is the intent, not a hazard: the driver reserve is
    # already excluded from it. What is worth saying is what would break the
    # plan, and what to do if the reserve estimate is wrong on this card.
    notes.append(
        f"uses {plan.vram_bytes/GiB:.2f} GB of the {budget.vram_bytes/GiB:.2f} GB "
        f"usable ({budget.vram_total_mib/1024:.2f} GB card, "
        f"{budget.driver_reserve_mib/1024:.2f} GB held by the driver)")
    if plan.n_ubatch != DEFAULT_UBATCH:
        notes.append(
            f"-ub raised to {plan.n_ubatch} because VRAM remained after the "
            f"last whole layer. This speeds up long-prompt ingestion and does "
            f"not affect generation either way")
    if plan.n_threads:
        notes.append(
            f"-t {plan.n_threads} is the physical core count. CPU work here is "
            f"memory-bandwidth bound, so hyperthreaded siblings contend for the "
            f"same channels instead of adding throughput")
    if plan.slack_bytes / GiB < 0.03:
        notes.append(
            f"packed to the limit. If throughput looks about half what you "
            f"expect, the KV cache has spilled to system RAM - llama.cpp does "
            f"not warn about this. Raising {groups.knob} by 1 fixes it.")
    if ram > budget.ram_bytes:
        short = (ram - budget.ram_bytes) / GiB
        plan.warnings.append(
            f"needs {ram/GiB:.1f} GB of system RAM against "
            f"{budget.ram_bytes/GiB:.1f} GB free - short by {short:.1f} GB. It "
            f"will still run, paging from disk, but slower. Freeing about "
            f"{short:.1f} GB removes that.")
    if model.is_moe and groups.movable_freq < 1.0:
        notes.append(
            f"routed experts are read {groups.movable_freq*100:.1f}% of the "
            f"time ({model.n_expert_used} of {model.n_expert}), so always-"
            f"resident tensors are worth {1/groups.movable_freq:.0f}x more per "
            f"byte of VRAM - they are placed first")
    _ = per_layer_gb
    return plan

"""Architecture analysis: what a model is, and exactly what its KV cache costs.

This is the correctness core. Every downstream decision - how much VRAM is
left, how many expert layers fit, whether a context length is reachable -
rests on the KV figure being right. Getting it wrong by 4x, which is easy,
produces confident advice that silently spills.

llama.cpp supports 50+ architectures and they do not all cache the same way.
Five mechanisms matter:

  MHA / GQA / MQA   n_head_kv x (key_len + value_len) per token per layer.
                    MQA is GQA with one KV head; MHA is GQA with as many KV
                    heads as query heads. One formula covers all three.

  MLA               DeepSeek's latent attention stores only a compressed
                    vector plus its RoPE part: (kv_lora_rank + rope_dim).
                    DeepSeek-V3 reaches ~70 KB/token where a GQA model of
                    similar size needs 192-328 KB. Applying the GQA formula
                    to an MLA model overestimates by 3-5x.

  SWA               Sliding-window layers only retain `n_swa` tokens, so
                    their contribution is constant once context exceeds the
                    window instead of growing with it.

  Recurrent         Mamba/SSM and RWKV keep a fixed-size state. It does not
                    grow with context at all.

  Hybrid            Modern Qwen and gpt-oss interleave: every nth layer is
                    full attention, the rest are SWA or recurrent. Only the
                    full-attention layers scale with context. On a 40-layer
                    1-in-4 model that is a 4x difference.

Where metadata is missing or an architecture is unrecognised, the analysis
says so rather than guessing quietly. A stated uncertainty can be worked
around; a silent one cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .gguf import ModelInfo

# Bytes per element for each KV cache type, including block scale overhead.
# llama.cpp quantises the cache in blocks of 32.
KV_ELEM_BYTES: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
}

# Quality order, best first. We only step down when the alternative is a
# context length that does not fit at all.
KV_LADDER = ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]


class Attention(str, Enum):
    STANDARD = "standard"      # MHA / GQA / MQA
    MLA = "mla"                # DeepSeek latent attention
    RECURRENT = "recurrent"    # Mamba / RWKV - no growing cache
    UNKNOWN = "unknown"


@dataclass
class KVProfile:
    """How this model's cache behaves as context grows."""
    mechanism: Attention
    growing_layers: int          # layers whose cache scales with context
    bytes_per_token_per_layer: float   # at f16, before quantisation scaling
    constant_bytes: float        # SWA windows + recurrent state, context-free
    sliding_window: int = 0
    recurrent_layers: int = 0
    swa_layers: int = 0
    notes: list[str] = field(default_factory=list)
    confident: bool = True

    def bytes_per_token(self, kv_type: str = "f16") -> float:
        scale = KV_ELEM_BYTES.get(kv_type.lower(), 2.0) / 2.0
        return self.bytes_per_token_per_layer * self.growing_layers * scale

    def total_bytes(self, n_ctx: int, kv_type: str = "f16") -> float:
        scale = KV_ELEM_BYTES.get(kv_type.lower(), 2.0) / 2.0
        return self.bytes_per_token(kv_type) * n_ctx + self.constant_bytes * scale


def _kv_int(m: ModelInfo, *suffixes: str, default: int = 0) -> int:
    """First arch-prefixed key that exists, as an int."""
    for s in suffixes:
        v = m.kv.get(f"{m.arch}.{s}")
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            vals = [int(x) for x in v if isinstance(x, (int, float))]
            if vals:
                return max(vals)
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return default


def detect_attention(m: ModelInfo) -> Attention:
    if _kv_int(m, "attention.kv_lora_rank") > 0:
        return Attention.MLA
    if _kv_int(m, "ssm.state_size") > 0 or m.arch.startswith(("mamba", "rwkv", "jamba")):
        # May still have attention layers interleaved; the hybrid path handles
        # that. Pure recurrent models fall through to a constant-size state.
        if m.n_head_kv == 0 or _kv_int(m, "attention.head_count_kv") == 0:
            return Attention.RECURRENT
    if m.n_head_kv > 0 and m.key_length > 0:
        return Attention.STANDARD
    return Attention.UNKNOWN


def _as_list(v) -> list[int] | None:
    """Per-layer metadata arrives as a list on heterogeneous architectures."""
    if isinstance(v, (list, tuple)) and v:
        try:
            return [int(x) for x in v]
        except (TypeError, ValueError):
            return None
    return None


def _sliding_mask(m: ModelInfo, n_layer: int) -> list[bool] | None:
    """Which layers use sliding-window attention, per layer.

    Three encodings exist in the wild and all must be honoured:

      sliding_window_pattern   an explicit per-layer bool array (Gemma).
                               True marks a *local* (sliding) layer.
      full_attention_interval  every nth layer is global (Qwen 3.5+).
      attention.sliding_window alone, meaning every layer slides.
    """
    pat = m.kv.get(f"{m.arch}.attention.sliding_window_pattern")
    if isinstance(pat, (list, tuple)) and pat:
        mask = [bool(x) for x in pat]
        if len(mask) < n_layer:
            mask = (mask * (n_layer // len(mask) + 1))[:n_layer]
        return mask[:n_layer]

    interval = max(1, m.full_attention_interval)
    if interval > 1:
        # Global layers are the ones on the interval; the rest are cheap.
        return [((i + 1) % interval) != 0 for i in range(n_layer)]

    if m.sliding_window > 0:
        return [True] * n_layer
    return None


def analyse_kv(m: ModelInfo) -> KVProfile:
    """Exact per-token KV cost, computed layer by layer.

    Layer-by-layer rather than with a single multiplier, because real
    architectures are not uniform. Gemma 4, for instance, alternates five
    sliding layers to one global layer, and the two kinds differ in *both*
    their KV head count and their key/value width:

        head_count_kv = [8,8,8,8,8,2, ...]   8 on sliding, 2 on global
        key_length    = 512, key_length_swa = 256

    Collapsing that to max(head_count_kv) and one width, then treating every
    layer as growing, overestimates the cache by roughly 20x - which forces a
    needless drop to q4_0, evicts every expert to system RAM, and still runs
    out of VRAM. The general form below reduces to the simple one whenever
    the metadata is scalar.
    """
    mech = detect_attention(m)
    n_layer = max(1, m.n_layer)
    notes: list[str] = []
    confident = True

    if mech is Attention.RECURRENT:
        return KVProfile(mechanism=mech, growing_layers=0,
                         bytes_per_token_per_layer=0.0, constant_bytes=0.0,
                         notes=["recurrent architecture: fixed state, no "
                                "growth with context"], confident=True)

    if mech is Attention.MLA:
        lora = _kv_int(m, "attention.kv_lora_rank")
        rope = _kv_int(m, "rope.dimension_count", "attention.rope_dimension_count")
        if rope == 0:
            rope, confident = 64, False
            notes.append("RoPE dimension not in metadata; assuming 64")
        per_layer = float(lora + rope) * 2.0
        notes.append(f"MLA: caches a {lora}-dim latent plus {rope} RoPE dims "
                     f"per layer, not {m.n_head_kv} x "
                     f"{m.key_length + m.value_length}")
        return KVProfile(mechanism=mech, growing_layers=n_layer,
                         bytes_per_token_per_layer=per_layer,
                         constant_bytes=0.0, notes=notes, confident=confident)

    # --- standard attention, resolved per layer -----------------------------
    heads_kv = _as_list(m.kv.get(f"{m.arch}.attention.head_count_kv"))
    k_glob = m.key_length
    v_glob = m.value_length
    k_swa = _kv_int(m, "attention.key_length_swa") or k_glob
    v_swa = _kv_int(m, "attention.value_length_swa") or v_glob
    window = m.sliding_window
    mask = _sliding_mask(m, n_layer)

    # A layer that is not global is cheap in one of two different ways, and
    # which one depends on the architecture rather than on the mask:
    #
    #   sliding   Gemma, Mistral. Capped at the window, so its cost is a
    #             constant once context exceeds it.
    #   recurrent Qwen 3.5+, gpt-oss. An SSM state of fixed size.
    #
    # An earlier version handled only the sliding case, so on hybrids whose
    # secondary layers are recurrent every layer fell through to "growing" and
    # the cache was overstated four-fold.
    d_state = _kv_int(m, "ssm.state_size")
    d_inner = _kv_int(m, "ssm.inner_size")
    conv = _kv_int(m, "ssm.conv_kernel")
    has_ssm = bool(d_state and d_inner)

    growing_bytes = 0.0        # bytes per token, summed over global layers
    constant = 0.0
    n_growing = 0
    n_sliding = 0
    n_recurrent = 0

    for i in range(n_layer):
        h = heads_kv[i] if (heads_kv and i < len(heads_kv)) else m.n_head_kv
        secondary = bool(mask[i]) if mask else False
        if secondary and window > 0:
            constant += float(h) * (k_swa + v_swa) * 2.0 * window
            n_sliding += 1
        elif secondary and has_ssm:
            constant += d_inner * (d_state + max(0, conv - 1)) * 4.0
            n_recurrent += 1
        else:
            # Global, or a secondary layer we cannot characterise - in which
            # case counting it as growing is the conservative error.
            growing_bytes += float(h) * (k_glob + v_glob) * 2.0
            n_growing += 1

    if n_growing == 0:
        # Everything slides: the cache stops growing once past the window.
        n_growing, per_layer = 0, 0.0
    else:
        per_layer = growing_bytes / n_growing

    if heads_kv and len(set(heads_kv)) > 1:
        notes.append(f"per-layer KV heads {sorted(set(heads_kv))} - resolved "
                     f"individually rather than collapsed to a maximum")
    if n_sliding:
        notes.append(f"{n_sliding} of {n_layer} layers slide over a "
                     f"{window:,}-token window and cost a constant "
                     f"{constant/2**20:,.0f} MiB; only {n_growing} grow with "
                     f"context")
    if k_swa != k_glob or v_swa != v_glob:
        notes.append(f"sliding layers use {k_swa}/{v_swa} key/value dims "
                     f"against {k_glob}/{v_glob} on global layers")
    if not n_sliding:
        if m.n_head_kv == 1:
            notes.append("multi-query attention (1 KV head)")
        elif m.n_head_kv < m.n_head:
            notes.append(f"grouped-query attention ({m.n_head}:{m.n_head_kv})")

    if mech is Attention.UNKNOWN:
        per_layer = max(per_layer,
                        float(max(1, m.n_head_kv)) * max(64, k_glob + v_glob) * 2.0)
        n_growing = n_growing or n_layer
        notes.append("unrecognised attention metadata; the KV figure is a "
                     "conservative guess and may be wrong")
        confident = False

    if n_recurrent:
        notes.append(f"{n_recurrent} recurrent layers hold a fixed "
                     f"{d_inner}x{d_state} state; only {n_growing} of "
                     f"{n_layer} grow with context")

    # Cross-layer attention: some architectures let adjacent layers share one
    # K/V cache, so fewer caches exist than there are layers. The field is
    # rare and its exact semantics are not documented in the GGUF spec, so
    # rather than guess a divisor and risk understating the cache - which
    # ends in an out-of-memory failure - the count is left alone and the
    # user is told the figure is conservative.
    shared = _kv_int(m, "attention.shared_kv_layers")
    if shared > 0:
        notes.append(f"declares {shared} cross-layer-shared KV layers, which "
                     f"is not modelled - the cache figure here is an "
                     f"overestimate and the plan will be a little conservative")
        confident = False

    return KVProfile(
        mechanism=mech, growing_layers=n_growing,
        bytes_per_token_per_layer=per_layer, constant_bytes=constant,
        sliding_window=window, recurrent_layers=n_recurrent,
        swa_layers=n_sliding, notes=notes, confident=confident)


def effective_context(m: ModelInfo, requested: int) -> tuple[int, str]:
    """Clamp a request to what the model was trained for.

    Beyond the trained context llama.cpp needs RoPE scaling, which changes
    quality and is a decision for the user rather than something to apply
    silently.
    """
    trained = m.n_ctx_train
    if trained and requested > trained:
        return trained, (f"requested {requested:,} exceeds the model's trained "
                         f"context of {trained:,}; clamped (going beyond needs "
                         f"RoPE scaling and costs quality)")
    return requested, ""


@dataclass
class Capabilities:
    """Model features that change which flags are worth emitting."""
    has_mtp: bool
    reasoning_levels: list[str]
    supports_thinking: bool
    is_moe: bool
    n_expert: int
    n_expert_used: int
    is_vision: bool
    notes: list[str] = field(default_factory=list)


def capabilities(m: ModelInfo) -> Capabilities:
    vision = any(k.startswith(("clip.", "vision.")) for k in m.kv) or \
        any(".vision." in t.name or t.name.startswith("v.") for t in m.tensors)
    notes: list[str] = []
    if vision:
        notes.append("multimodal: the vision projector is usually a separate "
                     "mmproj file and is not counted here")
    return Capabilities(
        has_mtp=m.has_mtp,
        reasoning_levels=m.reasoning_levels,
        supports_thinking=m.supports_thinking,
        is_moe=m.is_moe,
        n_expert=m.n_expert,
        n_expert_used=m.n_expert_used,
        is_vision=vision,
        notes=notes,
    )

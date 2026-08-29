"""Tensor sizing and classification, architecture-agnostic.

Two problems have to be solved for this to work on a model nobody has seen:

Sizing
------
The obvious approach is a table mapping ggml type ids to bytes per block. That
works until someone ships a quantisation the table does not know - a new IQ
variant, MXFP4, whatever lands next - and then every size is silently wrong,
which corrupts the whole plan.

The GGUF tensor table already records each tensor's byte offset into the data
section. Sorting by offset and differencing gives exact sizes for every tensor
except the last, whatever its type, with no table at all. The final tensor is
recovered from the file length. The type table stays only as a cross-check and
as a fallback for malformed files.

Classification
--------------
Which tensors may be moved to the CPU depends on the architecture, and naming
is not consistent across the 50+ that llama.cpp supports. Rather than match
exact names, classify by the substrings that have remained stable across
every architecture ggml has adopted, and treat anything unrecognised as
always-resident - the conservative choice, since misclassifying a hot tensor
as offloadable is far more damaging than the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gguf import ModelInfo, TensorInfo

# Fallback only. Sizes come from offsets when the file permits.
GGML_BLOCK: dict[int, tuple[int, int]] = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 50), 20: (32, 20), 21: (256, 110), 22: (256, 82),
    23: (256, 98), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 31: (256, 56), 32: (256, 56), 33: (256, 56),
    34: (256, 56), 35: (256, 56), 36: (256, 56), 37: (1, 1), 38: (1, 1),
    39: (32, 18),   # MXFP4
}

DEFAULT_BITS_PER_WEIGHT = 4.5  # only reached for a type we cannot identify


def _elements(t: TensorInfo) -> int:
    n = 1
    for d in t.dims:
        n *= d
    return n


def _size_from_type(t: TensorInfo) -> int:
    n = _elements(t)
    blk = GGML_BLOCK.get(t.dtype)
    if blk is None:
        return int(n * DEFAULT_BITS_PER_WEIGHT / 8)
    per_block, block_bytes = blk
    if per_block <= 1:
        return n * block_bytes
    return (n // per_block) * block_bytes


def sizes(model: ModelInfo) -> dict[str, int]:
    """Byte size of every tensor.

    Derived from offset deltas, which is exact regardless of quantisation
    type. Falls back to the type table per-tensor if the offsets look wrong
    (unsorted, overlapping, or absurd), which can happen with hand-edited or
    truncated files.
    """
    out: dict[str, int] = {}
    if not model.tensors:
        return out

    # The reader already sized each tensor within its own file, which is the
    # only place offset differencing is valid - a split model's shards each
    # restart their offsets from zero. Trust that when it is present.
    if all(getattr(t, "nbytes", 0) > 0 for t in model.tensors):
        return {t.name: t.nbytes for t in model.tensors}

    ordered = sorted(model.tensors, key=lambda t: t.offset)
    total_from_type = sum(_size_from_type(t) for t in model.tensors)

    # Offsets are relative to the start of the data section, whose absolute
    # position we never learn. That is fine for every tensor but the last,
    # which has no successor to difference against - so anchor it on the
    # summed type-derived total. Note this makes the last tensor the one place
    # the type table still matters; everything else is offset-exact.
    data_span = total_from_type

    ok = True
    for i, t in enumerate(ordered):
        if i + 1 < len(ordered):
            size = ordered[i + 1].offset - t.offset
        else:
            size = max(0, data_span - t.offset)
        if size <= 0 or size > model.file_size:
            ok = False
            break
        out[t.name] = size

    if not ok:
        return {t.name: _size_from_type(t) for t in model.tensors}

    # Offsets are padded for alignment, so the offset-derived total runs a
    # little above the type-derived one. A large disagreement means something
    # is off and the type table is the safer source.
    derived = sum(out.values())
    if total_from_type and not (0.90 <= derived / total_from_type <= 1.15):
        return {t.name: _size_from_type(t) for t in model.tensors}
    return out


# --- classification --------------------------------------------------------

# Substrings that have been stable across every architecture ggml supports.
_EXPERT_MARKERS = ("_exps", "ffn_gate_exp", "ffn_up_exp", "ffn_down_exp",
                   "experts.", "_exp.")
_SHARED_EXPERT_MARKERS = ("shexp", "shared_exp", "_shared")
_FFN_MARKERS = ("ffn_gate", "ffn_up", "ffn_down", "mlp.", "feed_forward")
_DOWN_MARKERS = ("ffn_down", "down_proj", "w2")


@dataclass
class Groups:
    """Tensors split by read frequency and by which flag can move them."""
    always_bytes: int = 0
    movable_bytes: int = 0
    movable_per_layer: dict[int, int] = field(default_factory=dict)
    down_per_layer: dict[int, int] = field(default_factory=dict)
    movable_freq: float = 1.0
    knob: str = "-ncffn"
    unknown_bytes: int = 0
    coverage: float = 1.0     # fraction of file bytes we accounted for

    @property
    def layers(self) -> list[int]:
        return sorted(self.movable_per_layer)


def _layer_of(name: str) -> int:
    if not name.startswith("blk."):
        return -1
    try:
        return int(name.split(".")[1])
    except (IndexError, ValueError):
        return -1


def classify(model: ModelInfo) -> Groups:
    """Decide, per tensor, whether it is always read or can be offloaded.

    MoE  routed experts are read k/E of the time, so they are the cheap ones
         to exile. Shared experts fire every token and stay.
    Dense every weight is read every token, so nothing is cheap - but FFN can
         still be offloaded while attention and the KV cache stay resident,
         which is strictly better than exiling whole layers.
    """
    g = Groups()
    sz = sizes(model)

    if model.is_moe:
        g.knob = "-ncmoe"
        k, E = model.n_expert_used, model.n_expert
        g.movable_freq = (k / E) if (k and E) else 1.0
    else:
        g.knob = "-ncffn"
        g.movable_freq = 1.0

    for t in model.tensors:
        b = sz.get(t.name) or _size_from_type(t)
        name = t.name.lower()
        layer = _layer_of(t.name)

        is_shared = any(m in name for m in _SHARED_EXPERT_MARKERS)
        is_expert = any(m in name for m in _EXPERT_MARKERS) and not is_shared
        is_ffn = any(m in name for m in _FFN_MARKERS)

        if model.is_moe:
            movable = is_expert
        else:
            movable = is_ffn and layer >= 0 and not is_shared

        if movable:
            g.movable_bytes += b
            if layer >= 0:
                g.movable_per_layer[layer] = g.movable_per_layer.get(layer, 0) + b
                if any(m in name for m in _DOWN_MARKERS):
                    g.down_per_layer[layer] = g.down_per_layer.get(layer, 0) + b
        else:
            g.always_bytes += b

    accounted = g.always_bytes + g.movable_bytes
    g.coverage = (accounted / model.file_size) if model.file_size else 1.0

    # If almost nothing was classified as movable on a model far larger than
    # VRAM, the naming is unfamiliar. Say so upstream rather than silently
    # producing a plan that cannot offload anything.
    if g.movable_bytes < model.file_size * 0.05:
        g.unknown_bytes = g.movable_bytes
    return g

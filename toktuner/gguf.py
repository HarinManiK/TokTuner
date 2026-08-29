"""GGUF metadata reader.

Pure stdlib. Reads only the header and metadata block - never the tensor data -
so opening a 60 GB model costs a few milliseconds and no meaningful memory.

The metadata is where every number we need to reason analytically lives:
layer count, attention geometry, expert counts, sliding-window intervals.
Getting these right is what lets us rule out impossible configurations
without ever launching llama.cpp.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# GGUF value type enum
(
    T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32,
    T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64,
) = range(13)

_SCALAR = {
    T_UINT8: ("<B", 1), T_INT8: ("<b", 1),
    T_UINT16: ("<H", 2), T_INT16: ("<h", 2),
    T_UINT32: ("<I", 4), T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4), T_BOOL: ("<?", 1),
    T_UINT64: ("<Q", 8), T_INT64: ("<q", 8), T_FLOAT64: ("<d", 8),
}

# How many bytes one element of a quantised tensor occupies, on average.
# Used only for sanity checks; real sizes come from the file itself.
BITS_PER_WEIGHT = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.67, "Q5_K_S": 5.52, "Q5_0": 5.5,
    "Q4_K_M": 4.85, "Q4_K_S": 4.58, "Q4_0": 4.55, "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q3_K_M": 3.91, "Q3_K_S": 3.5, "IQ3_M": 3.66, "IQ3_XXS": 3.06,
    "Q2_K": 2.63, "IQ2_M": 2.7, "MXFP4": 4.25,
}


class GGUFError(Exception):
    pass


class _Reader:
    """Buffered sequential reader over a file object."""

    def __init__(self, fh):
        self.fh = fh

    def raw(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise GGUFError(f"unexpected end of file (wanted {n} bytes, got {len(b)})")
        return b

    def scalar(self, vtype: int):
        fmt, size = _SCALAR[vtype]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        n = struct.unpack("<Q", self.raw(8))[0]
        if n > 64 * 1024 * 1024:
            raise GGUFError(f"implausible string length {n} - file likely not GGUF")
        return self.raw(n).decode("utf-8", errors="replace")

    def value(self, vtype: int) -> Any:
        if vtype in _SCALAR:
            return self.scalar(vtype)
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            elem_type = struct.unpack("<I", self.raw(4))[0]
            count = struct.unpack("<Q", self.raw(8))[0]
            if count > 64 * 1024 * 1024:
                raise GGUFError(f"implausible array length {count}")
            # Long arrays are almost always the tokenizer vocab. We skip their
            # contents rather than materialising 150k python strings.
            if count > 4096:
                if elem_type in _SCALAR:
                    _, size = _SCALAR[elem_type]
                    self.raw(size * count)
                elif elem_type == T_STRING:
                    for _ in range(count):
                        self.string()
                else:
                    raise GGUFError(f"cannot skip nested array type {elem_type}")
                return _Elided(count)
            return [self.value(elem_type) for _ in range(count)]
        raise GGUFError(f"unknown GGUF value type {vtype}")


@dataclass(frozen=True)
class _Elided:
    """Placeholder for a large array we deliberately did not read."""
    count: int

    def __repr__(self) -> str:
        return f"<{self.count} values elided>"


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    dtype: int
    offset: int
    # Byte size, resolved at read time. Offsets are relative to the start of
    # the file's own data section, so differencing them is only valid within
    # a single shard - which is why sizing happens here rather than later.
    nbytes: int = 0
    shard: int = 0


@dataclass
class ModelInfo:
    """Everything we can learn about a model without reading its weights."""

    path: Path
    file_size: int
    arch: str
    name: str
    kv: dict[str, Any] = field(repr=False, default_factory=dict)
    tensors: list[TensorInfo] = field(repr=False, default_factory=list)
    shards: int = 1

    # ---- architecture, resolved with fallbacks -------------------------------

    def _a(self, suffix: str, default=None):
        """Read an arch-prefixed key, e.g. 'qwen35.block_count'."""
        return self.kv.get(f"{self.arch}.{suffix}", default)

    @property
    def n_layer(self) -> int:
        return int(self._a("block_count", 0) or 0)

    @property
    def n_embd(self) -> int:
        return int(self._a("embedding_length", 0) or 0)

    @property
    def n_head(self) -> int:
        return int(self._a("attention.head_count", 0) or 0)

    @property
    def n_head_kv(self) -> int:
        v = self._a("attention.head_count_kv")
        if v is None:
            return self.n_head
        if isinstance(v, _Elided):
            return self.n_head
        if isinstance(v, list):
            return max(int(x) for x in v) if v else self.n_head
        return int(v)

    @property
    def key_length(self) -> int:
        v = self._a("attention.key_length")
        if v:
            return int(v)
        return self.n_embd // self.n_head if self.n_head else 0

    @property
    def value_length(self) -> int:
        v = self._a("attention.value_length")
        if v:
            return int(v)
        return self.key_length

    @property
    def n_ctx_train(self) -> int:
        return int(self._a("context_length", 0) or 0)

    @property
    def n_expert(self) -> int:
        return int(self._a("expert_count", 0) or 0)

    @property
    def n_expert_used(self) -> int:
        return int(self._a("expert_used_count", 0) or 0)

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 1

    @property
    def full_attention_interval(self) -> int:
        """1 = every layer does full attention.

        n = every nth layer is full attention, the rest are cheap (SSM or
        sliding-window). This single number changes KV cache size by 4x on
        the hybrid Qwen3.5/3.6/3.8 architectures and is the most commonly
        mis-modelled value in third-party memory calculators.
        """
        v = self._a("full_attention_interval")
        return int(v) if v else 1

    @property
    def sliding_window(self) -> int:
        for key in ("attention.sliding_window", "attention.n_swa"):
            v = self._a(key)
            if v:
                return int(v)
        return 0

    @property
    def has_mtp(self) -> bool:
        """Model ships multi-token-prediction heads (built-in speculation)."""
        if self._a("nextn_predict_layers"):
            return True
        return any(".nextn." in t.name for t in self.tensors)

    @property
    def n_attention_layers(self) -> int:
        """Layers that actually hold a growing KV cache."""
        if self.full_attention_interval > 1:
            return max(1, self.n_layer // self.full_attention_interval)
        return self.n_layer

    @property
    def quant(self) -> str:
        """Best-effort quantisation label, from filename then tensor types."""
        stem = self.path.stem.upper()
        # longest first so Q4_K_M wins over Q4_0
        for label in sorted(BITS_PER_WEIGHT, key=len, reverse=True):
            if label in stem:
                return label
        return "unknown"

    @property
    def chat_template(self) -> str:
        v = self.kv.get("tokenizer.chat_template", "")
        return v if isinstance(v, str) else ""

    @property
    def reasoning_levels(self) -> list[str]:
        """Effort levels this model's chat template actually branches on.

        llama.cpp forwards --reasoning-effort verbatim to the template and
        does not validate it, so a level the template ignores silently does
        nothing. Reading them out of the template is the only honest source.
        """
        t = self.chat_template
        if not t:
            return []
        found = []
        for lvl in ("minimal", "low", "medium", "high", "xhigh", "max"):
            if f"'{lvl}'" in t or f'"{lvl}"' in t:
                found.append(lvl)
        return found

    @property
    def supports_thinking(self) -> bool:
        t = self.chat_template
        return "enable_thinking" in t or "reasoning_effort" in t

    def summary(self) -> str:
        bits = [
            f"{self.name or self.path.stem}",
            f"arch={self.arch}",
            f"{self.n_layer}L",
            f"embd={self.n_embd}",
            f"heads={self.n_head}/{self.n_head_kv}kv",
            f"kv_len={self.key_length}",
            f"quant={self.quant}",
            f"{self.file_size / 2**30:.2f}GB" + (f" in {self.shards} shards" if self.shards > 1 else ""),
        ]
        if self.is_moe:
            bits.append(f"MoE {self.n_expert}x{self.n_expert_used}")
        if self.full_attention_interval > 1:
            bits.append(f"hybrid 1:{self.full_attention_interval}")
        if self.sliding_window:
            bits.append(f"swa={self.sliding_window}")
        if self.has_mtp:
            bits.append("MTP")
        return "  ".join(bits)


_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<total>\d{5})\.gguf$",
                       re.IGNORECASE)


def shard_siblings(path: Path) -> list[Path]:
    """Every file belonging to a split model, in order.

    Models much past ~50 GB are routinely published as
    `name-00001-of-00009.gguf`. Each shard carries its own tensor table and
    its own data section, so reading only the one the user happened to point
    at yields a fraction of the tensors and a fraction of the size - and a
    confidently wrong plan.

    Returns [path] for an ordinary single-file model.
    """
    m = _SHARD_RE.match(path.name)
    if not m:
        return [path]
    stem, total = m.group("stem"), int(m.group("total"))
    found: list[Path] = []
    for i in range(1, total + 1):
        cand = path.with_name(f"{stem}-{i:05d}-of-{total:05d}.gguf")
        if cand.is_file():
            found.append(cand)
    return found or [path]


def _size_tensors(tensors: list[TensorInfo], data_span: int) -> None:
    """Fill in nbytes from offset deltas, within one file."""
    if not tensors:
        return
    order = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(order):
        if i + 1 < len(order):
            t.nbytes = max(0, order[i + 1].offset - t.offset)
        else:
            t.nbytes = max(0, data_span - t.offset)


def read(path: str | Path, *, with_tensors: bool = True) -> ModelInfo:
    """Parse a GGUF model's metadata.

    Reads headers only - never tensor data - so an arbitrarily large model is
    inspected in milliseconds. Split models are followed to all their shards
    automatically.
    """
    first = Path(path)
    if not first.is_file():
        raise GGUFError(f"not a file: {first}")

    parts = shard_siblings(first)
    base = _read_one(parts[0], with_tensors=with_tensors, shard=0)
    if len(parts) == 1:
        return base

    total_size = base.file_size
    for idx, extra in enumerate(parts[1:], start=1):
        more = _read_one(extra, with_tensors=with_tensors, shard=idx)
        base.tensors.extend(more.tensors)
        total_size += more.file_size
    base.file_size = total_size
    base.shards = len(parts)
    return base


def _read_one(p: Path, *, with_tensors: bool, shard: int) -> ModelInfo:
    with p.open("rb") as fh:
        r = _Reader(fh)
        if r.raw(4) != GGUF_MAGIC:
            raise GGUFError(f"{p.name} is not a GGUF file (bad magic)")
        version = struct.unpack("<I", r.raw(4))[0]
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors = struct.unpack("<Q", r.raw(8))[0]
        n_kv = struct.unpack("<Q", r.raw(8))[0]
        if n_kv > 100_000 or n_tensors > 1_000_000:
            raise GGUFError("implausible header counts - file may be corrupt")

        kv: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = struct.unpack("<I", r.raw(4))[0]
            kv[key] = r.value(vtype)

        tensors: list[TensorInfo] = []
        data_start = fh.tell()
        if with_tensors:
            for _ in range(n_tensors):
                name = r.string()
                n_dims = struct.unpack("<I", r.raw(4))[0]
                dims = struct.unpack(f"<{n_dims}Q", r.raw(8 * n_dims))
                dtype = struct.unpack("<I", r.raw(4))[0]
                offset = struct.unpack("<Q", r.raw(8))[0]
                tensors.append(TensorInfo(name, dims, dtype, offset,
                                          shard=shard))
        data_start = fh.tell()

    # Offsets are relative to this file's data section, which begins right
    # after the header. data_start was captured while the file was still open -
    # reading it afterwards yields 0 and makes the final tensor of every shard
    # swallow the header bytes.
    file_size = p.stat().st_size
    _size_tensors(tensors, max(0, file_size - data_start))

    arch = kv.get("general.architecture", "unknown")
    return ModelInfo(
        path=p,
        file_size=file_size,
        arch=arch if isinstance(arch, str) else "unknown",
        name=kv.get("general.name", "") if isinstance(kv.get("general.name", ""), str) else "",
        kv=kv,
        tensors=tensors,
    )

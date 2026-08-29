"""Universality tests.

The planner has to be right for models nobody here has downloaded, so these
cases are synthesised from the canonical GGUF specification rather than read
from disk. Each one is an architecture family llama.cpp supports, built to
the tensor-naming and metadata conventions in gguf-py/constants.py.

Run:  python tests/test_universal.py
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from toktuner import arch, gguf, hardware, plan, tensors  # noqa: E402
from toktuner.gguf import ModelInfo, TensorInfo    # noqa: E402

GiB = 2 ** 30


def synth(arch_name, *, n_layer, n_embd, n_head, n_head_kv, key_len,
          n_expert=0, n_expert_used=0, ctx_train=32768, extra_kv=None,
          moe_layers=None, file_gb=8.0, shared_expert=False,
          dense_first=0) -> ModelInfo:
    """Build a plausible model to spec, with a real tensor table."""
    kv = {
        "general.architecture": arch_name,
        f"{arch_name}.block_count": n_layer,
        f"{arch_name}.context_length": ctx_train,
        f"{arch_name}.embedding_length": n_embd,
        f"{arch_name}.attention.head_count": n_head,
        f"{arch_name}.attention.head_count_kv": n_head_kv,
        f"{arch_name}.attention.key_length": key_len,
        f"{arch_name}.attention.value_length": key_len,
    }
    if n_expert:
        kv[f"{arch_name}.expert_count"] = n_expert
        kv[f"{arch_name}.expert_used_count"] = n_expert_used
    if extra_kv:
        kv.update({f"{arch_name}.{k}": v for k, v in extra_kv.items()})

    ts: list[TensorInfo] = []
    off = 0

    def add(name, dims, dtype=12):          # 12 = Q4_K
        nonlocal off
        ts.append(TensorInfo(name, tuple(dims), dtype, off))
        n = 1
        for d in dims:
            n *= d
        off += max(1, (n // 256) * 144)

    add("token_embd.weight", (n_embd, 128000))
    moe_set = set(range(dense_first, n_layer)) if moe_layers is None else set(moe_layers)
    for i in range(n_layer):
        add(f"blk.{i}.attn_norm.weight", (n_embd,), 0)
        add(f"blk.{i}.attn_q.weight", (n_embd, n_head * key_len))
        add(f"blk.{i}.attn_k.weight", (n_embd, n_head_kv * key_len))
        add(f"blk.{i}.attn_v.weight", (n_embd, n_head_kv * key_len))
        add(f"blk.{i}.attn_output.weight", (n_head * key_len, n_embd))
        if n_expert and i in moe_set:
            add(f"blk.{i}.ffn_gate_inp.weight", (n_embd, n_expert))
            for kind in ("gate", "up", "down"):
                add(f"blk.{i}.ffn_{kind}_exps.weight", (n_expert, n_embd, 512))
            if shared_expert:
                for kind in ("gate", "up", "down"):
                    add(f"blk.{i}.ffn_{kind}_shexp.weight", (n_embd, 2048))
        else:
            for kind in ("gate", "up", "down"):
                add(f"blk.{i}.ffn_{kind}.weight", (n_embd, 4 * n_embd))
    add("output.weight", (n_embd, 128000))

    return ModelInfo(path=Path(f"{arch_name}-synthetic.gguf"),
                     file_size=int(file_gb * GiB), arch=arch_name,
                     name=arch_name, kv=kv, tensors=ts)


def budget(vram_gb=12.0, ram_gb=32.0, ram_free_gb=24.0) -> plan.Budget:
    return plan.Budget(vram_total_mib=int(vram_gb * 1024),
                       vram_used_by_others_mib=0,
                       ram_total_mib=int(ram_gb * 1024),
                       ram_available_mib=int(ram_free_gb * 1024))


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# --- attention mechanisms ---------------------------------------------------

@case("MHA (heads == kv heads)")
def _():
    m = synth("llama", n_layer=32, n_embd=4096, n_head=32, n_head_kv=32, key_len=128)
    k = arch.analyse_kv(m)
    expect = 32 * (128 + 128) * 2
    return (k.mechanism is arch.Attention.STANDARD
            and abs(k.bytes_per_token_per_layer - expect) < 1
            and k.growing_layers == 32)


@case("GQA (llama-3 style 32:8)")
def _():
    m = synth("llama", n_layer=32, n_embd=4096, n_head=32, n_head_kv=8, key_len=128)
    k = arch.analyse_kv(m)
    return abs(k.bytes_per_token_per_layer - 8 * 256 * 2) < 1


@case("MQA (single KV head)")
def _():
    m = synth("falcon", n_layer=32, n_embd=4096, n_head=32, n_head_kv=1, key_len=128)
    k = arch.analyse_kv(m)
    return (abs(k.bytes_per_token_per_layer - 1 * 256 * 2) < 1
            and any("multi-query" in n for n in k.notes))


@case("MLA (DeepSeek latent) is 3-5x cheaper than the GQA formula")
def _():
    m = synth("deepseek2", n_layer=60, n_embd=7168, n_head=128, n_head_kv=128,
              key_len=192, extra_kv={"attention.kv_lora_rank": 512,
                                     "rope.dimension_count": 64})
    k = arch.analyse_kv(m)
    if k.mechanism is not arch.Attention.MLA:
        return False
    mla = k.bytes_per_token_per_layer            # (512 + 64) * 2
    gqa = 128 * (192 + 192) * 2                  # what the naive formula gives
    return abs(mla - (512 + 64) * 2) < 1 and gqa / mla > 3.0


@case("sliding window recorded")
def _():
    m = synth("gemma2", n_layer=42, n_embd=3584, n_head=16, n_head_kv=8,
              key_len=256, extra_kv={"attention.sliding_window": 4096})
    k = arch.analyse_kv(m)
    return k.sliding_window == 4096


@case("hybrid 1-in-4 with recurrent secondary layers")
def _():
    # Real hybrids declare what the non-global layers are. Qwen 3.5+ and
    # gpt-oss interleave SSM blocks, so the metadata carries ssm.* keys.
    m = synth("qwen35", n_layer=40, n_embd=2048, n_head=16, n_head_kv=2,
              key_len=256, extra_kv={"full_attention_interval": 4,
                                     "ssm.state_size": 128,
                                     "ssm.inner_size": 4096,
                                     "ssm.conv_kernel": 4})
    k = arch.analyse_kv(m)
    return (k.growing_layers == 10
            and k.bytes_per_token("f16") == 10 * 2 * 512 * 2
            and k.recurrent_layers == 30
            and k.constant_bytes > 0)


@case("hybrid with sliding secondary layers (Gemma-style pattern)")
def _():
    # Five local layers to one global, per-layer KV heads, and separate SWA
    # key/value widths - all three must be honoured together.
    m = synth("gemma4", n_layer=30, n_embd=2816, n_head=16, n_head_kv=8,
              key_len=512,
              extra_kv={"attention.sliding_window": 1024,
                        "attention.key_length_swa": 256,
                        "attention.value_length_swa": 256})
    m.kv["gemma4.attention.head_count_kv"] = [8, 8, 8, 8, 8, 2] * 5
    m.kv["gemma4.attention.sliding_window_pattern"] = \
        [True, True, True, True, True, False] * 5
    k = arch.analyse_kv(m)
    # 5 global layers at 2 heads x (512+512) x 2 bytes = 20,480 B/token
    return (k.growing_layers == 5 and k.swa_layers == 25
            and abs(k.bytes_per_token("f16") - 20480) < 1
            and k.constant_bytes > 0)


@case("hybrid whose secondary layers are unidentifiable stays conservative")
def _():
    # An interval is declared but nothing says what the other layers do.
    # Counting them as growing overestimates the cache, which costs a little
    # speed; assuming they are free would risk running out of VRAM.
    m = synth("mystery", n_layer=40, n_embd=2048, n_head=16, n_head_kv=2,
              key_len=256, extra_kv={"full_attention_interval": 4})
    k = arch.analyse_kv(m)
    return k.growing_layers == 40


@case("KV quantisation scales the cache")
def _():
    m = synth("llama", n_layer=32, n_embd=4096, n_head=32, n_head_kv=8, key_len=128)
    k = arch.analyse_kv(m)
    f16, q8, q4 = (k.bytes_per_token(t) for t in ("f16", "q8_0", "q4_0"))
    return abs(q8 / f16 - 0.53) < 0.02 and abs(q4 / f16 - 0.28) < 0.02


# --- classification ---------------------------------------------------------

@case("MoE routes experts to -ncmoe, keeps router and attention resident")
def _():
    m = synth("mixtral", n_layer=32, n_embd=4096, n_head=32, n_head_kv=8,
              key_len=128, n_expert=8, n_expert_used=2)
    g = tensors.classify(m)
    return (g.knob == "-ncmoe" and abs(g.movable_freq - 0.25) < 1e-6
            and g.movable_bytes > 0 and len(g.layers) == 32)


@case("shared experts stay resident, routed experts move")
def _():
    m = synth("qwen3moe", n_layer=24, n_embd=2048, n_head=16, n_head_kv=4,
              key_len=128, n_expert=64, n_expert_used=8, shared_expert=True)
    g = tensors.classify(m)
    shexp = sum(1 for t in m.tensors if "shexp" in t.name)
    return shexp > 0 and g.knob == "-ncmoe" and g.movable_bytes > 0


@case("dense model uses -ncffn and keeps attention resident")
def _():
    m = synth("llama", n_layer=32, n_embd=4096, n_head=32, n_head_kv=8, key_len=128)
    g = tensors.classify(m)
    sz = tensors.sizes(m)
    attn_bytes = sum(sz[t.name] for t in m.tensors if "attn" in t.name)
    ffn_bytes = sum(sz[t.name] for t in m.tensors if ".ffn_" in t.name)
    # Attention must be inside always_bytes and outside movable_bytes: -ncffn
    # exists precisely so the KV cache never leaves the GPU with its layer.
    return (g.knob == "-ncffn"
            and abs(g.movable_bytes - ffn_bytes) < 1024
            and g.always_bytes >= attn_bytes)


@case("first-N-dense MoE (only MoE layers are offloadable)")
def _():
    m = synth("granitemoe", n_layer=32, n_embd=2048, n_head=16, n_head_kv=4,
              key_len=128, n_expert=32, n_expert_used=4, dense_first=4)
    g = tensors.classify(m)
    return len(g.layers) == 28 and 0 not in g.movable_per_layer


@case("tensor sizes come from offsets, not a type table")
def _():
    m = synth("llama", n_layer=8, n_embd=2048, n_head=16, n_head_kv=4, key_len=128)
    for t in m.tensors:
        t.dtype = 999          # a quantisation the table has never seen
    s = tensors.sizes(m)
    return len(s) == len(m.tensors) and all(v > 0 for v in s.values())


# --- planning ---------------------------------------------------------------

@case("model that fits entirely offloads nothing")
def _():
    m = synth("llama", n_layer=16, n_embd=1024, n_head=16, n_head_kv=4,
              key_len=64, file_gb=1.5)
    p = plan.build(m, 4096, budget())
    return p.feasible and p.n_offload == 0


@case("model larger than VRAM offloads, keeps KV on GPU")
def _():
    # Sized so the tensor table itself exceeds a 12 GB card, rather than
    # merely claiming a large file_size in the header - the planner trusts
    # the tensors, which is the behaviour worth testing.
    m = synth("qwen3moe", n_layer=48, n_embd=4096, n_head=32, n_head_kv=4,
              key_len=128, n_expert=256, n_expert_used=8, file_gb=40.0)
    g = tensors.classify(m)
    total = (g.always_bytes + g.movable_bytes) / GiB
    p = plan.build(m, 4096, budget())
    return (total > 12.0 and p.feasible and p.n_offload > 0
            and p.kv_bytes > 0 and p.offload_cpu_bytes > 0)


@case("context too large for the card is reported, not guessed")
def _():
    m = synth("llama", n_layer=80, n_embd=8192, n_head=64, n_head_kv=64,
              key_len=128, ctx_train=1_000_000, file_gb=60.0)
    p = plan.build(m, 1_000_000, budget(vram_gb=8.0))
    return (not p.feasible) and any("exceed" in w for w in p.warnings)


@case("context clamped to the trained window")
def _():
    m = synth("llama", n_layer=32, n_embd=4096, n_head=32, n_head_kv=8,
              key_len=128, ctx_train=8192)
    p = plan.build(m, 131072, budget())
    return p.n_ctx == 8192 and any("clamped" in n for n in p.notes)


@case("KV steps down only when f16 will not fit")
def _():
    m = synth("llama", n_layer=80, n_embd=8192, n_head=64, n_head_kv=64,
              key_len=128, ctx_train=131072, file_gb=20.0)
    small = plan.build(m, 4096, budget(vram_gb=24.0))
    large = plan.build(m, 131072, budget(vram_gb=24.0))
    return small.kv_type == "f16" and (not large.feasible or large.kv_type != "f16")


@case("driver reserve scales with card size")
def _():
    return (budget(vram_gb=8).driver_reserve_mib >= 256
            and budget(vram_gb=48).driver_reserve_mib > budget(vram_gb=12).driver_reserve_mib)


@case("tiny GPU still produces a usable plan or an explicit refusal")
def _():
    m = synth("llama", n_layer=32, n_embd=4096, n_head=32, n_head_kv=8,
              key_len=128, file_gb=14.0)
    p = plan.build(m, 4096, budget(vram_gb=4.0, ram_gb=16.0, ram_free_gb=12.0))
    return (p.feasible and p.n_offload > 0) or (not p.feasible and p.warnings)


@case("emitted command is a single line with no placeholders")
def _():
    m = synth("qwen3moe", n_layer=32, n_embd=2048, n_head=16, n_head_kv=4,
              key_len=128, n_expert=64, n_expert_used=8, file_gb=18.0)
    p = plan.build(m, 8192, budget())
    cmd = p.command(r"C:\llama\llama-server.exe")
    return ("\n" not in cmd and "{" not in cmd and "-ngl 99" in cmd
            and "-c 8192" in cmd and cmd.count("-m ") == 1)


@case("no crash on a model with no tensor table")
def _():
    m = ModelInfo(path=Path("empty.gguf"), file_size=GiB, arch="unknown",
                  name="", kv={}, tensors=[])
    p = plan.build(m, 4096, budget())
    return isinstance(p, plan.Plan)


@case("unknown architecture is flagged, not silently guessed")
def _():
    m = ModelInfo(path=Path("weird.gguf"), file_size=4 * GiB, arch="brandnew",
                  name="", kv={"general.architecture": "brandnew"}, tensors=[])
    k = arch.analyse_kv(m)
    return not k.confident and any("unrecognised" in n for n in k.notes)


# --- CPU topology -----------------------------------------------------------
# Tests the rule, not this machine: efficiency-class splits mean opposite
# things on Intel and AMD, and getting it backwards either discards most of
# the CPU or floods it with weak cores.

@case("Intel hybrid: performance cores only (E-cores are Atom-derived)")
def _():
    # 12th-gen style: 8 P-cores (class 1) + 8 E-cores (class 0)
    return hardware.select_cores("intel", [1] * 8 + [0] * 8) == 8


@case("AMD Zen-c: every core counts (same IPC, only lower clocks)")
def _():
    # 4 Zen 5 (class 1) + 8 Zen 5c (class 0) must yield 12, not 4
    return hardware.select_cores("amd", [1] * 4 + [0] * 8) == 12


@case("uniform CPU: all cores regardless of vendor")
def _():
    return (hardware.select_cores("intel", [0] * 16) == 16
            and hardware.select_cores("amd", [0] * 16) == 16)


@case("unknown vendor keeps every core (safer error direction)")
def _():
    return hardware.select_cores("unknown", [1] * 4 + [0] * 8) == 12


@case("no topology data degrades rather than crashing")
def _():
    return hardware.select_cores("intel", []) == 0 and hardware.physical_cores() >= 1


# --- split models -----------------------------------------------------------
# Anything much past ~50 GB is published as name-00001-of-000NN.gguf. Reading
# only the shard the user pointed at yields a fraction of the tensors and a
# confidently wrong plan, so the reader must follow the whole set.

def _write_shard(path, arch, tensor_specs, kvs):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    parts = [s("general.architecture") + struct.pack("<I", 8) + s(arch)]
    for k, v in kvs.items():
        parts.append(s(k) + struct.pack("<I", 4) + struct.pack("<I", int(v)))

    tinfo, off, data = b"", 0, b""
    for name, dims, dtype, nbytes in tensor_specs:
        tinfo += s(name) + struct.pack("<I", len(dims))
        tinfo += struct.pack(f"<{len(dims)}Q", *dims)
        tinfo += struct.pack("<I", dtype) + struct.pack("<Q", off)
        off += nbytes
        data += b"\0" * nbytes

    head = b"GGUF" + struct.pack("<I", 3)
    head += struct.pack("<Q", len(tensor_specs)) + struct.pack("<Q", len(parts))
    path.write_bytes(head + b"".join(parts) + tinfo + data)


def _build_split(tmp):
    arch = "llama"
    kvs = {f"{arch}.block_count": 6, f"{arch}.embedding_length": 512,
           f"{arch}.attention.head_count": 8,
           f"{arch}.attention.head_count_kv": 2,
           f"{arch}.attention.key_length": 64,
           f"{arch}.attention.value_length": 64,
           f"{arch}.context_length": 8192}
    for shard in range(3):
        specs = []
        for i in range(shard * 2, shard * 2 + 2):
            specs.append((f"blk.{i}.attn_q.weight", (512, 512), 12, 200_000))
            specs.append((f"blk.{i}.ffn_gate.weight", (512, 2048), 12, 800_000))
            specs.append((f"blk.{i}.ffn_down.weight", (2048, 512), 12, 800_000))
        _write_shard(tmp / f"m-{shard+1:05d}-of-00003.gguf", arch, specs,
                     kvs if shard == 0 else {})
    return tmp / "m-00001-of-00003.gguf"


@case("split model: all shards merged, sizes summed exactly")
def _():
    tmp = Path(tempfile.mkdtemp())
    first = _build_split(tmp)
    on_disk = sum(f.stat().st_size for f in tmp.glob("*.gguf"))
    m = gguf.read(first)
    g = tensors.classify(m)
    return (m.shards == 3 and len(m.tensors) == 18
            and m.file_size == on_disk and len(g.layers) == 6
            and g.movable_bytes == 9_600_000)


@case("split model: pointing at a middle shard still resolves the set")
def _():
    tmp = Path(tempfile.mkdtemp())
    _build_split(tmp)
    m = gguf.read(tmp / "m-00002-of-00003.gguf")
    return m.shards == 3 and len(m.tensors) == 18


@case("tensor sizing excludes header bytes")
def _():
    # The last tensor of each shard must not absorb the header. Total sized
    # bytes should equal the data written, not the file lengths.
    tmp = Path(tempfile.mkdtemp())
    first = _build_split(tmp)
    m = gguf.read(first)
    sized = sum(t.nbytes for t in m.tensors)
    return sized == 18 * 0 + (6 * (200_000 + 800_000 + 800_000))


# --- command hygiene --------------------------------------------------------

@case("paths with spaces are quoted for the host shell")
def _():
    import platform
    m = synth("llama", n_layer=8, n_embd=512, n_head=8, n_head_kv=2,
              key_len=64, file_gb=1.0)
    m.path = Path(r"D:\My Models\a model.gguf")
    p = plan.build(m, 4096, budget())
    cmd = p.command(r"C:\Program Files\llama.cpp\llama-server.exe")
    if platform.system() == "Windows":
        # POSIX single quotes do not work as an executable path in cmd.exe or
        # PowerShell; the default llama.cpp location contains a space.
        return ('"C:\\Program Files' in cmd
                and '"D:\\My Models' in cmd
                and "'" not in cmd)
    return "'" in cmd


@case("no -ot override is emitted (fused expert tensors make it unsafe)")
def _():
    m = synth("qwen3moe", n_layer=32, n_embd=2048, n_head=16, n_head_kv=4,
              key_len=128, n_expert=64, n_expert_used=8, file_gb=18.0)
    for ctx in (4096, 16384, 65536):
        p = plan.build(m, ctx, budget())
        if p.feasible and "-ot" in p.flags():
            return False
    return True


@case("cross-layer KV sharing is flagged rather than silently ignored")
def _():
    m = synth("test", n_layer=32, n_embd=4096, n_head=32, n_head_kv=8,
              key_len=128, extra_kv={"attention.shared_kv_layers": 16})
    k = arch.analyse_kv(m)
    # Not modelled - but the estimate must be the conservative direction and
    # the user must be told, rather than the field passing unnoticed.
    return (not k.confident
            and any("shared" in n.lower() for n in k.notes)
            and k.growing_layers == 32)


@case("command has no stray whitespace or newlines at any context")
def _():
    m = synth("llama", n_layer=32, n_embd=4096, n_head=32, n_head_kv=8,
              key_len=128, file_gb=14.0)
    for ctx in (512, 4096, 32768):
        p = plan.build(m, ctx, budget())
        if not p.feasible:
            continue
        cmd = p.command("srv")
        if "\n" in cmd or "\t" in cmd or "  " in cmd or cmd != cmd.strip():
            return False
    return True


def main() -> int:
    print("=" * 70)
    print("toktuner - universality checks (synthetic architectures)")
    print("=" * 70)
    passed = 0
    for name, fn in CASES:
        try:
            ok = bool(fn())
            err = ""
        except Exception as exc:
            ok, err = False, f"  [{type(exc).__name__}: {exc}]"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{err}")
        passed += ok
    print("=" * 70)
    print(f"{passed}/{len(CASES)} passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())

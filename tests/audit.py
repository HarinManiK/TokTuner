"""Adversarial audit.

Three questions this answers that the other suites do not:

  1. Do the flags toktuner emits actually parse? llama.cpp validates arguments
     before it opens the model, so pointing it at a nonexistent file exercises
     the whole argument parser without loading anything. An unparseable flag is
     the one failure a user sees immediately, and it is entirely preventable.

  2. Does any real model on this machine produce a plan that overruns its own
     budget, or contains nonsense - negative sizes, impossible layer counts,
     KV larger than the card?

  3. Do adversarial synthetic models break the planner? Degenerate shapes,
     absurd contexts, missing metadata, contradictory metadata.

Run:  python tests/audit.py [--server PATH]
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from toktuner import arch, gguf, hardware, plan, tensors   # noqa: E402
from toktuner.gguf import ModelInfo, TensorInfo            # noqa: E402

GiB = 2 ** 30
CONTEXTS = [512, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

problems: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> bool:
    global checks
    checks += 1
    if not ok:
        problems.append(f"{label}{('  -> ' + detail) if detail else ''}")
    return ok


# --------------------------------------------------------------------------
# 1. flag syntax
# --------------------------------------------------------------------------

def find_server(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        for n in ("llama-server.exe", "llama-server"):
            if (p / n).is_file():
                return p / n
    for guess in (Path(r"E:\LLMs\Llama_cpp"), Path.cwd()):
        for n in ("llama-server.exe", "llama-server"):
            if (guess / n).is_file():
                return guess / n
    return None


def flags_parse(server: Path, flags: list[str]) -> tuple[bool, str]:
    """True if llama.cpp accepts every flag.

    A deliberately absent model path means it fails at load, after the parser
    has already validated the arguments - so no model is ever read.
    """
    args = [str(server), "-m", "___toktuner_audit_no_such_model.gguf", *flags]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=90,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:
        return False, f"could not run llama-server: {exc}"
    out = (r.stdout + r.stderr).lower()
    for marker in ("error while handling argument", "invalid argument",
                   "unknown argument", "unrecognized", "unknown buffer type",
                   "the argument has been removed"):
        if marker in out:
            line = next((l for l in (r.stdout + r.stderr).splitlines()
                         if marker in l.lower()), marker)
            return False, line.strip()[:160]
    return True, ""


def audit_flags(server: Path | None) -> None:
    print("\n[1] flag syntax (parser only - no model is loaded)")
    if server is None:
        print("    SKIP - llama-server not found")
        return

    models = sorted(glob.glob(r"E:\LLMs\*.gguf"))
    if not models:
        print("    SKIP - no models to derive flag sets from")
        return

    mach = hardware.detect()
    budget = plan.Budget(mach.gpu.total_mib, 0, mach.memory.total_mib,
                         mach.memory.available_mib,
                         physical_cores=hardware.physical_cores())

    seen: set[tuple] = set()
    tested = 0
    for path in models:
        m = gguf.read(path)
        for ctx in CONTEXTS:
            if m.n_ctx_train and ctx > m.n_ctx_train:
                continue
            p = plan.build(m, ctx, budget)
            if not p.feasible:
                continue
            key = tuple(p.flags())
            if key in seen:
                continue
            seen.add(key)
            ok, why = flags_parse(server, list(key))
            tested += 1
            check(ok, f"flags rejected for {Path(path).name} @ {ctx}", why)
    print(f"    {tested} distinct flag combinations parsed")


# --------------------------------------------------------------------------
# 2. real models, every context
# --------------------------------------------------------------------------

def audit_models() -> None:
    print("\n[2] real models on this machine")
    models = sorted(glob.glob(r"E:\LLMs\*.gguf"))
    if not models:
        print("    SKIP - none present")
        return

    mach = hardware.detect()
    budget = plan.Budget(mach.gpu.total_mib, 0, mach.memory.total_mib,
                         mach.memory.available_mib,
                         physical_cores=hardware.physical_cores())
    cap = budget.vram_bytes

    for path in models:
        m = gguf.read(path)
        name = Path(path).name
        kvp = arch.analyse_kv(m)
        g = tensors.classify(m)

        check(kvp.bytes_per_token("f16") >= 0, f"{name}: negative KV rate")
        check(0 <= kvp.growing_layers <= max(1, m.n_layer),
              f"{name}: growing layers {kvp.growing_layers} vs {m.n_layer} total")
        check(g.always_bytes + g.movable_bytes > 0, f"{name}: no bytes classified")
        check(g.coverage > 0.85, f"{name}: only {g.coverage*100:.0f}% of bytes classified")

        for ctx in CONTEXTS:
            if m.n_ctx_train and ctx > m.n_ctx_train:
                continue
            p = plan.build(m, ctx, budget)
            tag = f"{name} @ {ctx:,}"
            if not p.feasible:
                check(bool(p.warnings), f"{tag}: infeasible with no explanation")
                continue
            check(p.vram_bytes <= cap * 1.001, f"{tag}: plan overruns VRAM budget",
                  f"{p.vram_bytes/GiB:.2f} GB vs {cap/GiB:.2f} GB")
            check(p.vram_bytes > 0 and p.ram_bytes >= 0, f"{tag}: nonsense footprint")
            check(0 <= p.n_offload <= max(1, p.layers_total),
                  f"{tag}: offload {p.n_offload} of {p.layers_total}")
            check(p.n_ctx <= (m.n_ctx_train or p.n_ctx), f"{tag}: context not clamped")
            check(p.kv_bytes < cap, f"{tag}: KV alone exceeds the card")
            check(p.n_ubatch >= 1 and p.n_batch >= p.n_ubatch,
                  f"{tag}: batch {p.n_batch} < ubatch {p.n_ubatch}")
            cmd = p.command("srv")
            check("\n" not in cmd and "  " not in cmd.replace("  ", " "),
                  f"{tag}: malformed command string")
    print(f"    {len(models)} models x {len(CONTEXTS)} contexts")


# --------------------------------------------------------------------------
# 3. adversarial synthetic models
# --------------------------------------------------------------------------

def _m(arch_name, kv, tensors_=None, size=8 * GiB) -> ModelInfo:
    kv = {"general.architecture": arch_name, **{f"{arch_name}.{k}": v
                                                for k, v in kv.items()}}
    return ModelInfo(path=Path("x.gguf"), file_size=size, arch=arch_name,
                     name="x", kv=kv, tensors=tensors_ or [])


def audit_adversarial() -> None:
    print("\n[3] adversarial and degenerate models")
    b = plan.Budget(12 * 1024, 0, 32 * 1024, 24 * 1024, physical_cores=8)

    cases = {
        "zero layers": _m("z", {"block_count": 0, "embedding_length": 512,
                                "attention.head_count": 8,
                                "attention.head_count_kv": 8,
                                "attention.key_length": 64,
                                "context_length": 4096}),
        "zero heads": _m("z", {"block_count": 8, "embedding_length": 512,
                               "attention.head_count": 0,
                               "attention.head_count_kv": 0,
                               "attention.key_length": 0,
                               "context_length": 4096}),
        "no metadata at all": _m("z", {}),
        "context_length zero": _m("z", {"block_count": 8, "embedding_length": 512,
                                        "attention.head_count": 8,
                                        "attention.head_count_kv": 2,
                                        "attention.key_length": 64,
                                        "context_length": 0}),
        "expert_used > expert_count": _m("z", {
            "block_count": 8, "embedding_length": 512, "attention.head_count": 8,
            "attention.head_count_kv": 2, "attention.key_length": 64,
            "context_length": 4096, "expert_count": 4, "expert_used_count": 99}),
        "sliding window larger than context": _m("z", {
            "block_count": 8, "embedding_length": 512, "attention.head_count": 8,
            "attention.head_count_kv": 2, "attention.key_length": 64,
            "context_length": 4096, "attention.sliding_window": 1_000_000}),
        "per-layer heads shorter than layer count": _m("z", {
            "block_count": 30, "embedding_length": 512, "attention.head_count": 8,
            "attention.head_count_kv": [8, 2], "attention.key_length": 64,
            "context_length": 4096}),
        "absurd layer count": _m("z", {
            "block_count": 100000, "embedding_length": 512,
            "attention.head_count": 8, "attention.head_count_kv": 2,
            "attention.key_length": 64, "context_length": 4096}),
        "MLA without rope dims": _m("deepseek2", {
            "block_count": 60, "embedding_length": 7168, "attention.head_count": 128,
            "attention.head_count_kv": 128, "attention.key_length": 192,
            "context_length": 4096, "attention.kv_lora_rank": 512}),
    }

    for label, model in cases.items():
        try:
            kvp = arch.analyse_kv(model)
            g = tensors.classify(model)
            for ctx in (512, 4096, 262144):
                p = plan.build(model, ctx, b)
                if p.feasible:
                    check(p.vram_bytes <= b.vram_bytes * 1.001,
                          f"adversarial '{label}' @ {ctx} overruns budget")
                    check(p.n_offload >= 0, f"adversarial '{label}': negative offload")
                    p.command("srv")
            check(kvp.bytes_per_token("f16") >= 0,
                  f"adversarial '{label}': negative KV rate")
            _ = g.coverage
        except Exception as exc:
            check(False, f"adversarial '{label}' raised",
                  f"{type(exc).__name__}: {exc}")
    print(f"    {len(cases)} degenerate models")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server")
    args = ap.parse_args()

    print("=" * 72)
    print("toktuner audit")
    print("=" * 72)

    audit_adversarial()
    audit_models()
    audit_flags(find_server(args.server))

    print()
    print("=" * 72)
    if problems:
        print(f"{len(problems)} PROBLEM(S) out of {checks} checks:")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print(f"clean - {checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

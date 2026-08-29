"""Validate the planner's memory model against real measurements.

The synthetic tests in test_universal.py prove the code handles every
architecture family correctly. They cannot prove the *numbers* are right,
because they check the code against itself.

This does the other half: it replays thirteen configurations that were
actually launched and measured - VRAM read from nvidia-smi, throughput from
llama.cpp's own timings - and checks the analytical model reproduces them.

A memory model that has never been confronted with a real allocation is a
guess with extra steps. This is what turns the arithmetic into a claim.

Run:  python tests/test_ground_truth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from toktuner import arch, gguf, plan, tensors    # noqa: E402

GiB = 2 ** 30
MODEL_DIR = Path(r"E:\LLMs")
GT = json.loads((ROOT / "tests" / "ground_truth.json").read_text(encoding="utf-8"))

# The model is used as a feasibility gate, so what matters is not mistaking a
# configuration that fits for one that spills. On a 12 GB card the gap between
# those outcomes is a few hundred MB.
TOLERANCE_GB = 0.40


def _from_spec(name: str):
    """Rebuild the measured model from its recorded properties.

    The measurements are permanent; the file they came from need not be. The
    spec was read off the real file with toktuner's own readers, so the
    validation reproduces on any machine rather than only where that model
    happens to sit.
    """
    spec = GT.get("model_specs", {}).get(name)
    if spec is None:
        return None
    a = spec["arch"]
    kv = {
        "general.architecture": a,
        f"{a}.block_count": spec["n_layer"],
        f"{a}.context_length": spec["n_ctx_train"],
        f"{a}.attention.head_count": spec["n_head"],
        f"{a}.attention.head_count_kv": spec["n_head_kv"],
        f"{a}.attention.key_length": spec["key_length"],
        f"{a}.attention.value_length": spec["value_length"],
        f"{a}.expert_count": spec["n_expert"],
        f"{a}.expert_used_count": spec["n_expert_used"],
        f"{a}.full_attention_interval": spec["full_attention_interval"],
        f"{a}.ssm.state_size": spec["ssm_state_size"],
        f"{a}.ssm.inner_size": spec["ssm_inner_size"],
        f"{a}.ssm.conv_kernel": spec["ssm_conv_kernel"],
        f"{a}.embedding_length": 2048,
    }
    model = gguf.ModelInfo(path=Path(f"{name}.gguf"),
                           file_size=spec["file_bytes"], arch=a, name=name,
                           kv=kv, tensors=[])
    return model, spec


def _predict_from_spec(model, spec, run) -> float:
    kvp = arch.analyse_kv(model)
    kv_gb = kvp.total_bytes(run["n_ctx"], run["kv"]) / GiB
    n_layers = spec["movable_layers"]
    on_cpu = min(run["ncmoe"], n_layers)
    per_layer = spec["movable_bytes"] / max(1, n_layers)
    movable_on_gpu = spec["movable_bytes"] - on_cpu * per_layer
    overhead = plan.compute_buffers(model, run["n_ctx"], 512) \
        + plan.CUDA_CONTEXT_MIB * plan.MiB
    return (spec["always_bytes"] + movable_on_gpu) / GiB + kv_gb + overhead / GiB


def _predict(model, run, machine) -> float:
    """VRAM this configuration should occupy, per the planner's own model."""
    kvp = arch.analyse_kv(model)
    groups = tensors.classify(model)
    kv_gb = kvp.total_bytes(run["n_ctx"], run["kv"]) / GiB

    n_layers = len(groups.layers)
    on_cpu = min(run["ncmoe"], n_layers)
    per_layer = groups.movable_bytes / max(1, n_layers)
    movable_on_gpu = groups.movable_bytes - on_cpu * per_layer

    overhead = plan.compute_buffers(model, run["n_ctx"], 512) \
        + plan.CUDA_CONTEXT_MIB * plan.MiB
    return (groups.always_bytes + movable_on_gpu) / GiB + kv_gb + overhead / GiB


def main() -> int:
    print("=" * 72)
    print("toktuner - analytical model vs measured allocations")
    print("=" * 72)

    machine = GT["machine"]
    results: list[bool] = []
    cache: dict[str, object] = {}

    print(f"  {'ctx':>8} {'kv':>5} {'ncmoe':>6} {'measured':>9} {'predicted':>10} {'delta':>7}")
    print("  " + "-" * 54)

    for run in GT["runs"]:
        name = run["model"]
        if name not in cache:
            path = MODEL_DIR / f"{name}.gguf"
            if path.exists():
                cache[name] = ("file", gguf.read(path))
            else:
                built = _from_spec(name)
                cache[name] = ("spec", built) if built else ("none", None)
        kind, payload = cache[name]
        if kind == "none":
            continue
        if kind == "file":
            got = _predict(payload, run, machine)
        else:
            model, spec = payload
            got = _predict_from_spec(model, spec, run)
        delta = got - run["vram_gb"]
        ok = abs(delta) <= TOLERANCE_GB
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {run['n_ctx']:>8,} {run['kv']:>5} "
              f"{run['ncmoe']:>6} {run['vram_gb']:>8.2f}G {got:>9.2f}G {delta:>+6.2f}G")

    if not results:
        print("\n  SKIPPED - none of the measured models are present on this "
              "machine.\n  The synthetic suite (test_universal.py) still applies.")
        return 0

    print()
    worst = 0.0
    for run in GT["runs"]:
        kind, payload = cache.get(run["model"], ("none", None))
        if kind == "file":
            got = _predict(payload, run, machine)
        elif kind == "spec":
            model, spec = payload
            got = _predict_from_spec(model, spec, run)
        else:
            continue
        worst = max(worst, abs(got - run["vram_gb"]))
    print(f"  worst-case error: {worst:.2f} GB across {len(results)} measured "
          f"configurations")
    src = {k for k, _ in cache.values()}
    if "spec" in src:
        print("  (replayed from embedded model properties - the measured file "
              "is no longer on this machine)")
    print("=" * 72)
    print(f"{sum(results)}/{len(results)} within {TOLERANCE_GB:.2f} GB")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

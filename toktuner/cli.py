r"""Command line entry point.

    toktuner --model model.gguf                     4096 tokens, default
    toktuner --model model.gguf --ctx 32768
    toktuner --model model.gguf --all               every context length
    toktuner --gui                                  open the window

Computes the flags. Runs nothing, loads nothing, measures nothing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import arch, gguf, hardware, plan

CTX_LADDER = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


def find_server(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        for name in ("llama-server.exe", "llama-server"):
            for cand in (p / name, p / "bin" / name, p / "build" / "bin" / name):
                if cand.is_file():
                    return cand
        return None
    for name in ("llama-server.exe", "llama-server"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _machine_budget(safety_gb: float = 0.0) -> tuple[hardware.Machine, plan.Budget]:
    m = hardware.detect()
    if m.gpu is None:
        raise SystemExit(
            "No NVIDIA GPU detected (nvidia-smi unavailable).\n"
            "toktuner plans for single-GPU NVIDIA systems.")
    b = plan.Budget(
        vram_total_mib=m.gpu.total_mib,
        vram_used_by_others_mib=m.gpu.used_mib,
        ram_total_mib=m.memory.total_mib,
        ram_available_mib=m.memory.available_mib,
        safety_mib=int(safety_gb * 1024),
        physical_cores=hardware.physical_cores(),
    )
    return m, b


def describe(model: gguf.ModelInfo, machine: hardware.Machine,
             budget: plan.Budget) -> None:
    kvp = arch.analyse_kv(model)
    caps = arch.capabilities(model)
    print("=" * 78)
    print(f"  {model.path.name}")
    print(f"  {model.summary()}")
    print("=" * 78)
    print(f"  {machine.gpu.name}")
    print(f"  VRAM {machine.gpu.total_gb:.2f} GB total, "
          f"{budget.vram_bytes/plan.GiB:.2f} GB usable "
          f"({budget.driver_reserve_mib/1024:.2f} GB driver reserve"
          + (f", {machine.gpu.used_mib/1024:.2f} GB in use by other apps"
             if machine.gpu.used_mib > 64 else "") + ")")
    print(f"  RAM  {machine.memory.total_gb:.1f} GB total, "
          f"{machine.memory.available_gb:.1f} GB available")
    print()
    print(f"  attention      {kvp.mechanism.value}")
    print(f"  KV per token   {kvp.bytes_per_token('f16'):,.0f} bytes at f16")
    print(f"  cache layers   {kvp.growing_layers} of {model.n_layer} grow with context")
    if caps.has_mtp:
        print("  speculation    built-in MTP heads detected")
    if caps.reasoning_levels:
        print(f"  reasoning      {', '.join(caps.reasoning_levels)} "
              f"(--reasoning-effort <level>)")
    elif caps.supports_thinking:
        print("  reasoning      on/off only (--reasoning off)")
    for n in kvp.notes:
        print(f"     - {n}")
    for n in caps.notes:
        print(f"     - {n}")
    print()


def show(p: plan.Plan, server: Path, verbose: bool) -> None:
    if not p.feasible:
        print("  NOT POSSIBLE at this context length.")
        for w in p.warnings:
            print(f"  ! {w}")
        return
    print(p.command(server))
    if not verbose:
        return
    print()
    print(f"  always resident  {p.always_bytes/plan.GiB:6.2f} GB")
    print(f"  KV cache         {p.kv_bytes/plan.GiB:6.2f} GB  ({p.kv_type})")
    print(f"  offload -> GPU   {p.offload_gpu_bytes/plan.GiB:6.2f} GB  "
          f"{p.layers_on_gpu} of {p.layers_total} layers")
    print(f"  offload -> CPU   {p.offload_cpu_bytes/plan.GiB:6.2f} GB  "
          f"{p.n_offload} layers")
    print(f"  VRAM             {p.vram_bytes/plan.GiB:6.2f} GB")
    print(f"  system RAM       {p.ram_bytes/plan.GiB:6.2f} GB")
    print(f"  {p.gpu_read_share*100:.1f}% of per-token reads come from VRAM")
    for w in p.warnings:
        print(f"  ! {w}")
    for n in p.notes:
        print(f"  - {n}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="toktuner",
        description="Compute the fastest llama.cpp flags for this machine. "
                    "No benchmarking - it reads the model and solves for the "
                    "best memory layout.")
    ap.add_argument("--model", "-m", help="path to a .gguf file")
    ap.add_argument("--ctx", "-c", type=int, default=4096,
                    help="context length to plan for (default 4096)")
    ap.add_argument("--server", "-s",
                    help="llama-server path, or the folder containing it")
    ap.add_argument("--all", action="store_true",
                    help="show a plan for every common context length")
    ap.add_argument("--safety", type=float, default=0.0, metavar="GB",
                    help="hold back extra VRAM beyond the driver reserve")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="print only the command")
    ap.add_argument("--gui", action="store_true", help="open the desktop app")
    args = ap.parse_args(argv)

    if args.gui or not args.model:
        from .gui import main as gui_main
        return gui_main()

    path = Path(args.model)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2
    try:
        model = gguf.read(path)
    except gguf.GGUFError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    machine, budget = _machine_budget(args.safety)
    server = find_server(args.server) or Path("llama-server")

    if not args.quiet:
        describe(model, machine, budget)

    if args.all:
        for ctx in CTX_LADDER:
            if model.n_ctx_train and ctx > model.n_ctx_train:
                continue
            p = plan.build(model, ctx, budget)
            print(f"--- {ctx:,} tokens " + "-" * max(0, 58 - len(f"{ctx:,}")))
            show(p, server, not args.quiet)
            print()
        return 0

    p = plan.build(model, args.ctx, budget)
    show(p, server, not args.quiet)
    return 0 if p.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal desktop front end.

Pick the llama.cpp folder, pick a .gguf, choose a context length, press Plan.
It prints one command.

tkinter is used deliberately: it ships with Python, so the packaged binary
needs no runtime and the source needs no install step.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import arch, gguf, hardware, plan

APP = "TokTuner"
CONFIG = Path(os.environ.get("APPDATA", Path.home())) / "toktuner" / "gui.json"

CTX_CHOICES = ["2048", "4096", "8192", "16384", "32768",
               "49152", "65536", "98304", "131072", "196608", "262144"]

BG = "#1b1d21"
FG = "#e6e6e6"
DIM = "#9aa0a6"
ACCENT = "#4c8dff"
FIELD = "#25282d"
OK = "#5fd68a"
WARN = "#f0b132"
BAD = "#ff6b6b"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config(d: dict) -> None:
    try:
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP} - llama.cpp flag planner")
        self.configure(bg=BG)
        self.geometry("980x680")
        self.minsize(860, 600)

        cfg = _load_config()
        self.var_llama = tk.StringVar(value=cfg.get("llama_dir", ""))
        self.var_model = tk.StringVar(value=cfg.get("model", ""))
        self.var_ctx = tk.StringVar(value=cfg.get("ctx", "4096"))
        self.var_status = tk.StringVar(value="")
        self._command = ""
        self._icon_img = None

        self._set_icon()
        self._style()
        self._build()
        self._probe_hardware()

    def _set_icon(self) -> None:
        candidates = [
            Path(getattr(sys, "_MEIPASS", "")) / "assets" / "icon.ico",
            Path(getattr(sys, "_MEIPASS", "")) / "assets" / "logo.png",
            Path(__file__).resolve().parent.parent / "assets" / "icon.ico",
            Path(__file__).resolve().parent.parent / "assets" / "logo.png",
        ]
        for cand in candidates:
            if cand.is_file():
                try:
                    if cand.suffix == ".ico":
                        self.iconbitmap(str(cand))
                        break
                    elif cand.suffix == ".png":
                        img = tk.PhotoImage(file=str(cand))
                        self.iconphoto(True, img)
                        self._icon_img = img
                        break
                except Exception:
                    pass

    # -- chrome --------------------------------------------------------------

    def _style(self) -> None:
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("Dim.TLabel", background=BG, foreground=DIM, font=("Segoe UI", 9))
        s.configure("Head.TLabel", background=BG, foreground=FG,
                    font=("Segoe UI Semibold", 12))
        s.configure("TButton", font=("Segoe UI", 10), padding=6)
        s.configure("Go.TButton", font=("Segoe UI Semibold", 11), padding=9)
        s.configure("TEntry", fieldbackground=FIELD, foreground=FG,
                    insertcolor=FG, borderwidth=0)
        s.configure("TCombobox", fieldbackground=FIELD, foreground=FG,
                    background=FIELD, borderwidth=0)

    def _row(self, parent, label, var, browse, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 2))
        e = ttk.Entry(parent, textvariable=var)
        e.grid(row=row, column=1, sticky="ew", padx=(10, 8), pady=(0, 2))
        ttk.Button(parent, text="Browse", command=browse, width=10).grid(
            row=row, column=2, pady=(0, 2))
        return e

    def _build(self) -> None:
        pad = {"padx": 18}
        head = ttk.Frame(self)
        head.pack(fill="x", pady=(16, 6), **pad)
        ttk.Label(head, text="TokTuner", style="Head.TLabel").pack(anchor="w")
        ttk.Label(head, text="Computes the fastest llama.cpp flags for your "
                             "machine. No benchmarking - it reads the model and "
                             "solves for the best memory layout.",
                  style="Dim.TLabel", wraplength=900, justify="left").pack(anchor="w")

        self.hw = ttk.Label(self, text="detecting hardware...", style="Dim.TLabel")
        self.hw.pack(anchor="w", pady=(8, 10), **pad)

        form = ttk.Frame(self)
        form.pack(fill="x", **pad)
        form.columnconfigure(1, weight=1)
        self._row(form, "llama.cpp folder", self.var_llama, self._pick_llama, 0)
        self._row(form, "Model (.gguf)", self.var_model, self._pick_model, 1)

        ttk.Label(form, text="Context").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ctxbox = ttk.Combobox(form, textvariable=self.var_ctx, values=CTX_CHOICES,
                              width=14)
        ctxbox.grid(row=2, column=1, sticky="w", padx=(10, 8), pady=(8, 0))

        self.go = ttk.Button(self, text="Plan", style="Go.TButton",
                             command=self._start)
        self.go.pack(anchor="w", pady=(14, 6), **pad)

        ttk.Label(self, textvariable=self.var_status, style="Dim.TLabel").pack(
            anchor="w", **pad)

        ttk.Label(self, text="Command", style="Head.TLabel").pack(
            anchor="w", pady=(12, 4), **pad)
        wrap = tk.Frame(self, bg=BG)
        wrap.pack(fill="x", **pad)
        self.out = tk.Text(wrap, height=4, wrap="word", bg=FIELD, fg=OK,
                           insertbackground=FG, relief="flat",
                           font=("Consolas", 10), padx=10, pady=8)
        self.out.pack(fill="x")
        self.out.configure(state="disabled")

        btns = ttk.Frame(self)
        btns.pack(anchor="w", pady=(8, 4), **pad)
        self.copy_btn = ttk.Button(btns, text="Copy command", command=self._copy,
                                   state="disabled")
        self.copy_btn.pack(side="left")
        self.save_btn = ttk.Button(btns, text="Save as .bat", command=self._save_bat,
                                   state="disabled")
        self.save_btn.pack(side="left", padx=(8, 0))

        ttk.Label(self, text="Why", style="Head.TLabel").pack(
            anchor="w", pady=(12, 4), **pad)
        dwrap = tk.Frame(self, bg=BG)
        dwrap.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        self.detail = tk.Text(dwrap, wrap="word", bg=FIELD, fg=FG, relief="flat",
                              font=("Consolas", 9), padx=10, pady=8)
        sb = ttk.Scrollbar(dwrap, command=self.detail.yview)
        self.detail.configure(yscrollcommand=sb.set, state="disabled")
        sb.pack(side="right", fill="y")
        self.detail.pack(side="left", fill="both", expand=True)
        self.detail.tag_configure("warn", foreground=WARN)
        self.detail.tag_configure("bad", foreground=BAD)
        self.detail.tag_configure("dim", foreground=DIM)

    # -- actions -------------------------------------------------------------

    def _pick_llama(self) -> None:
        d = filedialog.askdirectory(title="Folder containing llama-server.exe")
        if d:
            self.var_llama.set(d)

    def _pick_model(self) -> None:
        start = self.var_model.get() or self.var_llama.get() or ""
        f = filedialog.askopenfilename(
            title="Select a GGUF model",
            initialdir=str(Path(start).parent) if start else None,
            filetypes=[("GGUF model", "*.gguf"), ("All files", "*.*")])
        if f:
            self.var_model.set(f)

    def _probe_hardware(self) -> None:
        def work():
            try:
                m = hardware.detect()
            except Exception as exc:
                self.after(0, lambda: self.hw.configure(
                    text=f"hardware detection failed: {exc}"))
                return
            if m.gpu is None:
                txt = ("No NVIDIA GPU detected. TokTuner targets single-GPU "
                       "NVIDIA systems; results will be CPU-only.")
            else:
                b = plan.Budget(m.gpu.total_mib, m.gpu.used_mib,
                                m.memory.total_mib, m.memory.available_mib)
                txt = (f"{m.gpu.name} - {m.gpu.total_gb:.2f} GB VRAM "
                       f"({b.vram_bytes/plan.GiB:.2f} GB usable after driver "
                       f"reserve)   |   RAM {m.memory.total_gb:.0f} GB, "
                       f"{m.memory.available_gb:.1f} GB free")
            self.after(0, lambda: self.hw.configure(text=txt))
        threading.Thread(target=work, daemon=True).start()

    def _start(self) -> None:
        llama = self.var_llama.get().strip()
        model = self.var_model.get().strip()
        try:
            ctx = int(self.var_ctx.get().strip())
            if ctx < 256:
                raise ValueError
        except ValueError:
            messagebox.showerror(APP, "Context must be a whole number of tokens.")
            return
        if not model or not Path(model).is_file():
            messagebox.showerror(APP, "Select a .gguf model file.")
            return

        exe = self._find_server(llama)
        if exe is None:
            messagebox.showerror(
                APP, "Could not find llama-server.exe in that folder.\n\n"
                     "Pick the folder that contains llama-server.exe.")
            return

        _save_config({"llama_dir": llama, "model": model, "ctx": str(ctx)})
        self.go.configure(state="disabled")
        self.var_status.set("reading model and solving...")
        threading.Thread(target=self._work, args=(exe, model, ctx),
                         daemon=True).start()

    @staticmethod
    def _find_server(folder: str) -> Path | None:
        if not folder:
            return None
        base = Path(folder)
        for name in ("llama-server.exe", "llama-server"):
            for cand in (base / name, base / "bin" / name, base / "build" / "bin" / name):
                if cand.is_file():
                    return cand
        return None

    def _work(self, exe: Path, model_path: str, ctx: int) -> None:
        try:
            model = gguf.read(model_path)
            mach = hardware.detect()
            if mach.gpu is None:
                raise RuntimeError("no NVIDIA GPU detected (nvidia-smi unavailable)")
            budget = plan.Budget(mach.gpu.total_mib, mach.gpu.used_mib,
                                 mach.memory.total_mib, mach.memory.available_mib,
                                 physical_cores=hardware.physical_cores())
            p = plan.build(model, ctx, budget)
            kvp = arch.analyse_kv(model)
            caps = arch.capabilities(model)
        except Exception as exc:
            self.after(0, lambda: self._failed(str(exc)))
            return
        self.after(0, lambda: self._done(p, kvp, caps, exe))

    def _failed(self, msg: str) -> None:
        self.go.configure(state="normal")
        self.var_status.set("")
        messagebox.showerror(APP, msg)

    def _done(self, p, kvp, caps, exe) -> None:
        self.go.configure(state="normal")
        self.var_status.set("")

        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        if p.feasible:
            self._command = p.command(exe)
            self.out.insert("1.0", self._command)
            self.out.configure(fg=OK)
            self.copy_btn.configure(state="normal")
            self.save_btn.configure(state="normal")
        else:
            self._command = ""
            self.out.insert("1.0", "No configuration fits at this context length.")
            self.out.configure(fg=BAD)
            self.copy_btn.configure(state="disabled")
            self.save_btn.configure(state="disabled")
        self.out.configure(state="disabled")

        d = self.detail
        d.configure(state="normal")
        d.delete("1.0", "end")

        m = p.model
        d.insert("end", f"{m.path.name}\n{m.summary()}\n\n")
        d.insert("end", f"attention        {kvp.mechanism.value}\n")
        d.insert("end", f"KV per token     {kvp.bytes_per_token(p.kv_type):,.0f} bytes "
                        f"({p.kv_type})\n")
        d.insert("end", f"growing layers   {kvp.growing_layers} of {m.n_layer}\n")
        if caps.has_mtp:
            d.insert("end", "speculation      built-in MTP heads (enabled)\n")
        if caps.reasoning_levels:
            d.insert("end", f"reasoning levels {', '.join(caps.reasoning_levels)} "
                            f"(add --reasoning-effort <level>)\n")
        elif caps.supports_thinking:
            d.insert("end", "reasoning        on/off only (--reasoning off)\n")
        d.insert("end", "\n")

        if p.feasible:
            d.insert("end", "MEMORY PLAN\n")
            d.insert("end", f"  always resident   {p.always_bytes/plan.GiB:6.2f} GB   "
                            f"read every token\n")
            d.insert("end", f"  KV cache          {p.kv_bytes/plan.GiB:6.2f} GB\n")
            d.insert("end", f"  offloadable/GPU   {p.offload_gpu_bytes/plan.GiB:6.2f} GB   "
                            f"{p.layers_on_gpu} of {p.layers_total} layers\n")
            d.insert("end", f"  offloadable/CPU   {p.offload_cpu_bytes/plan.GiB:6.2f} GB   "
                            f"{p.n_offload} layers\n")
            d.insert("end", f"  VRAM total        {p.vram_bytes/plan.GiB:6.2f} GB\n")
            d.insert("end", f"  system RAM        {p.ram_bytes/plan.GiB:6.2f} GB\n\n")
            d.insert("end", f"  {p.gpu_read_share*100:.1f}% of the bytes read per "
                            f"generated token come from VRAM\n\n")

        for w in p.warnings:
            d.insert("end", f"! {w}\n\n", "warn")
        for n in p.notes:
            d.insert("end", f"- {n}\n", "dim")
        d.configure(state="disabled")

    def _copy(self) -> None:
        if not self._command:
            return
        self.clipboard_clear()
        self.clipboard_append(self._command)
        self.var_status.set("copied to clipboard")
        self.after(2200, lambda: self.var_status.set(""))

    def _save_bat(self) -> None:
        if not self._command:
            return
        f = filedialog.asksaveasfilename(
            defaultextension=".bat", initialfile="run-model.bat",
            filetypes=[("Batch file", "*.bat")])
        if not f:
            return
        Path(f).write_text(
            "@echo off\r\n"
            "title llama.cpp\r\n"
            "echo Starting. The web UI will be at http://localhost:8080\r\n"
            "echo.\r\n"
            f"{self._command}\r\n"
            "pause\r\n", encoding="utf-8")
        self.var_status.set(f"saved {Path(f).name}")
        self.after(2600, lambda: self.var_status.set(""))


def main() -> int:
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

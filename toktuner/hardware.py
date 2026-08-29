"""Hardware discovery.

Single discrete NVIDIA GPU plus system RAM. Deliberately narrow: multi-GPU
and RPC introduce interactions that cannot be reasoned about analytically,
and tuning advice that ignores them is worse than no advice.

An integrated GPU alongside the dGPU is expected and ignored - llama.cpp's
CUDA backend does not use it, but it does mean the dGPU may be free of
desktop compositing, which materially changes usable VRAM.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass


class HardwareError(Exception):
    pass


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return out.stdout.strip()
    except Exception:
        return ""


@dataclass
class GPU:
    name: str
    total_mib: int
    free_mib: int
    used_mib: int
    driver: str
    bandwidth_gbps: float | None = None

    @property
    def total_gb(self) -> float:
        return self.total_mib / 1024

    @property
    def free_gb(self) -> float:
        return self.free_mib / 1024


@dataclass
class Memory:
    total_mib: int
    available_mib: int

    @property
    def total_gb(self) -> float:
        return self.total_mib / 1024

    @property
    def available_gb(self) -> float:
        return self.available_mib / 1024

    @property
    def used_by_others_gb(self) -> float:
        return (self.total_mib - self.available_mib) / 1024


@dataclass
class Machine:
    gpu: GPU | None
    memory: Memory
    cpu_name: str
    cpu_threads: int
    os_name: str

    def fingerprint(self) -> str:
        """Stable key for caching results across runs.

        Deliberately excludes free memory and driver patch level: those change
        constantly and would defeat the cache without changing the optimum.
        """
        gpu = f"{self.gpu.name}:{self.gpu.total_mib}" if self.gpu else "nogpu"
        return f"{gpu}|ram{self.memory.total_mib}|{self.cpu_threads}t"

    def summary(self) -> str:
        g = (f"{self.gpu.name} ({self.gpu.total_gb:.1f} GB, "
             f"{self.gpu.free_gb:.1f} free)") if self.gpu else "no CUDA GPU"
        return (f"{g}\n"
                f"{self.cpu_name} ({self.cpu_threads} threads)\n"
                f"RAM {self.memory.total_gb:.1f} GB "
                f"({self.memory.available_gb:.1f} free, "
                f"{self.memory.used_by_others_gb:.1f} in use)")


# --- GPU -----------------------------------------------------------------

# Peak theoretical bandwidth, GB/s. Used only for the roofline sanity check:
# if measured throughput is a small fraction of the ceiling, something is
# misconfigured and we say so rather than reporting it as optimal.
_BANDWIDTH = {
    "5090": 1792, "5080": 960, "5070 TI": 896, "5070": 672, "5060": 448,
    "4090": 1008, "4080": 717, "4070 TI": 504, "4070": 504, "4060": 272,
    "3090": 936, "3080": 760, "3070": 448, "3060": 360,
    "A6000": 768, "A5000": 768, "L40": 864, "H100": 3350, "A100": 1935,
}


def _bandwidth_for(name: str) -> float | None:
    up = name.upper()
    for key in sorted(_BANDWIDTH, key=len, reverse=True):
        if key in up:
            bw = _BANDWIDTH[key]
            # Laptop parts run materially narrower memory than desktop.
            if "LAPTOP" in up or "MOBILE" in up:
                bw *= 0.75
            return bw
    return None


def detect_gpu() -> GPU | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 5:
        return None
    try:
        name = parts[0]
        gpu = GPU(
            name=name,
            total_mib=int(float(parts[1])),
            free_mib=int(float(parts[2])),
            used_mib=int(float(parts[3])),
            driver=parts[4],
            bandwidth_gbps=_bandwidth_for(name),
        )
    except ValueError:
        return None
    return gpu


def gpu_free_mib() -> int:
    """Live VRAM reading. Called during benchmarking to detect spill."""
    out = _run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], timeout=8)
    try:
        return int(float(out.splitlines()[0].strip()))
    except Exception:
        return -1


def gpu_used_mib() -> int:
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=8)
    try:
        return int(float(out.splitlines()[0].strip()))
    except Exception:
        return -1


# --- system memory -------------------------------------------------------

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def detect_memory() -> Memory:
    if platform.system() == "Windows":
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return Memory(
            total_mib=int(st.ullTotalPhys / 2**20),
            available_mib=int(st.ullAvailPhys / 2**20),
        )
    # Linux / macOS
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for line in fh:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0])  # kB
        return Memory(
            total_mib=info["MemTotal"] // 1024,
            available_mib=info.get("MemAvailable", info["MemFree"]) // 1024,
        )
    except Exception:
        pass
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return Memory(total_mib=int(total / 2**20), available_mib=int(total / 2**20 * 0.5))
    except Exception:
        raise HardwareError("cannot determine system memory on this platform")


def memory_available_mib() -> int:
    return detect_memory().available_mib


def pagefile_used_mib() -> int:
    """Bytes currently committed to the page file.

    This, not free RAM, is the signal that a machine is genuinely thrashing.

    Under mmap the OS deliberately fills all spare RAM with file-backed page
    cache, so "free memory" falls close to zero on any healthy run with a
    large model - measurements taken at 51 tok/s sat at 1-2 GB free
    throughout. Treating that as danger rejects exactly the configurations
    the tool exists to find.

    Page file growth is different: it means anonymous memory is being evicted
    to disk, which is slow, wears the drive, and degrades everything else on
    the machine. That is worth aborting for.
    """
    if platform.system() != "Windows":
        try:
            with open("/proc/meminfo") as fh:
                total = free = 0
                for line in fh:
                    if line.startswith("SwapTotal"):
                        total = int(line.split()[1]) // 1024
                    elif line.startswith("SwapFree"):
                        free = int(line.split()[1]) // 1024
                return max(0, total - free)
        except Exception:
            return 0
    out = _run(["powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_PageFileUsage | "
                "Measure-Object -Property CurrentUsage -Sum).Sum"], timeout=15)
    try:
        return int(float(out.strip() or 0))
    except Exception:
        return 0


# --- CPU -----------------------------------------------------------------

def _cpu_vendor() -> str:
    """'intel', 'amd', 'apple' or 'unknown'.

    Needed because efficiency-class splits mean opposite things per vendor.
    """
    name = ""
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as k:
                name = str(winreg.QueryValueEx(k, "VendorIdentifier")[0])
                if not name:
                    name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0])
        except Exception:
            name = ""
    if not name:
        name = platform.processor() or ""
    low = name.lower()
    if "genuineintel" in low or "intel" in low:
        return "intel"
    if "authenticamd" in low or "amd" in low or "ryzen" in low:
        return "amd"
    if "apple" in low:
        return "apple"
    return "unknown"


def select_cores(vendor: str, efficiency_classes: list[int]) -> int:
    """How many cores to hand llama.cpp, given a CPU's core tiers.

    Pure decision logic, separated from hardware probing so it can be tested
    against CPUs this machine does not have.

    A hybrid split means opposite things per vendor:

      Intel P/E   E-cores are Atom-derived - different microarchitecture,
                  much lower IPC, no SMT. Pinning to P-cores only is reported
                  at up to 3x on 12th-gen and later.

      AMD Zen c   Zen 4c/5c are the same architecture as their full-size
                  siblings with identical IPC, only lower clocks and less
                  cache. They are fully useful; on a 4+8 part, keeping just
                  the four "performance" cores discards two thirds of the
                  compute.

    Anything unrecognised keeps every core, which is the safer error: too
    many threads costs a little contention, too few costs most of the CPU.
    """
    if not efficiency_classes:
        return 0
    if vendor == "intel" and len(set(efficiency_classes)) > 1:
        top = max(efficiency_classes)
        return sum(1 for c in efficiency_classes if c == top)
    return len(efficiency_classes)


class _PROCESSOR_RELATIONSHIP_HEAD(ctypes.Structure):
    """Leading fields of PROCESSOR_RELATIONSHIP.

    Only the first few members are needed, and declaring the trailing
    variable-length GROUP_AFFINITY array is unnecessary for counting.
    """
    _fields_ = [
        ("Flags", ctypes.c_ubyte),
        ("EfficiencyClass", ctypes.c_ubyte),
        ("Reserved", ctypes.c_ubyte * 20),
        ("GroupCount", ctypes.c_ushort),
    ]


def _windows_core_classes() -> list[int]:
    """EfficiencyClass of every physical core, via GetLogicalProcessorInformationEx.

    On hybrid designs (Intel 12th gen onward, Core Ultra) performance cores
    report a higher EfficiencyClass than efficiency cores. On uniform CPUs
    every core reports 0. Returns an empty list if the API is unavailable.
    """
    RelationProcessorCore = 0
    kernel32 = ctypes.windll.kernel32
    length = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformationEx(
        RelationProcessorCore, None, ctypes.byref(length))
    if length.value == 0:
        return []
    buf = (ctypes.c_ubyte * length.value)()
    if not kernel32.GetLogicalProcessorInformationEx(
            RelationProcessorCore, buf, ctypes.byref(length)):
        return []

    classes: list[int] = []
    offset = 0
    while offset < length.value:
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: DWORD Relationship,
        # DWORD Size, then the union.
        rel = int.from_bytes(bytes(buf[offset:offset + 4]), "little")
        size = int.from_bytes(bytes(buf[offset + 4:offset + 8]), "little")
        if size <= 0:
            break
        if rel == RelationProcessorCore:
            head = _PROCESSOR_RELATIONSHIP_HEAD.from_buffer(buf, offset + 8)
            classes.append(int(head.EfficiencyClass))
        offset += size
    return classes


def physical_cores() -> int:
    """Cores worth giving llama.cpp, on any CPU.

    Three facts drive this:

      * The CPU side of inference is memory-bandwidth bound, so hyperthreaded
        siblings contend for the same channels rather than adding throughput.
        Benchmarks peak at the physical core count and regress past it.

      * On hybrid CPUs - Intel 12th gen onward, Core Ultra, Apple Silicon -
        the efficiency cores are markedly slower and share the same memory
        path. Counting them in raises thread count without raising bandwidth,
        and community testing reports up to 3x from pinning to performance
        cores only. So hybrid parts return the performance-core count, not the
        total.

      * Everything degrades to something sane. If the platform APIs are
        unavailable, half the logical count is correct for every
        hyperthreaded x86 CPU and harmless where it is not.
    """
    system = platform.system()
    n = 0

    if system == "Windows":
        try:
            classes = _windows_core_classes()
            if classes:
                # Both Intel and AMD report multiple efficiency classes, but
                # they mean very different things and the distinction decides
                # whether discarding the lower tier helps or throws away most
                # of the CPU.
                #
                #   Intel P/E   E-cores are Atom-derived: different micro-
                #               architecture, much lower IPC, no SMT. Pinning
                #               to P-cores only is reported at up to 3x.
                #
                #   AMD Zen c   Zen 5c/4c are the SAME architecture as their
                #               full-size siblings with identical IPC, merely
                #               lower clocks and less cache. They are fully
                #               useful. On a 4+8 part, keeping only the four
                #               "performance" cores would discard two thirds
                #               of the available compute.
                #
                # So: honour the split on Intel, ignore it everywhere else.
                n = select_cores(_cpu_vendor(), classes)
        except Exception:
            n = 0
        if n <= 0:
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor | "
                        "Measure-Object -Property NumberOfCores -Sum).Sum"],
                       timeout=20)
            try:
                n = int(float(out.strip() or 0))
            except ValueError:
                n = 0

    elif system == "Darwin":
        # Apple Silicon exposes performance cores as perflevel0.
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            out = _run(["sysctl", "-n", key], timeout=10)
            try:
                n = int(out.strip())
                if n > 0:
                    break
            except ValueError:
                n = 0

    else:
        # Linux. Prefer the topology view, which handles multi-socket and
        # shows every core exactly once.
        try:
            seen = set()
            with open("/proc/cpuinfo") as fh:
                phys = core = None
                for line in fh:
                    line = line.strip()
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                        if phys is not None:
                            seen.add((phys, core))
                    elif not line:
                        phys = core = None
            n = len(seen)
        except Exception:
            n = 0

    if n <= 0:
        logical = os.cpu_count() or 1
        n = max(1, logical // 2) if logical > 1 else 1
    return max(1, n)


def detect_cpu() -> tuple[str, int]:
    threads = os.cpu_count() or 1
    name = platform.processor() or "unknown CPU"
    if platform.system() == "Windows":
        # wmic was removed from recent Windows builds, so read the registry
        # directly and fall back to PowerShell only if that fails.
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as k:
                name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        except Exception:
            got = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor).Name"], timeout=20)
            if got.strip():
                name = got.strip().splitlines()[0]
    else:
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    return re.sub(r"\s+", " ", name).strip(), threads


def detect() -> Machine:
    cpu_name, threads = detect_cpu()
    return Machine(
        gpu=detect_gpu(),
        memory=detect_memory(),
        cpu_name=cpu_name,
        cpu_threads=threads,
        os_name=f"{platform.system()} {platform.release()}",
    )

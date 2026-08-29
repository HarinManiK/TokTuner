"""Build standalone toktuner binary (Windows .exe or Linux ELF).

    python build_exe.py

Produces dist/toktuner.exe (on Windows) or dist/toktuner (on Linux).
A single self-contained binary, no Python install needed on the target machine.
PyInstaller is the only build-time dependency; the tool itself has none at runtime.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Installing it now...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
        if r.returncode != 0:
            print("Could not install PyInstaller.", file=sys.stderr)
            return 1

    entry = ROOT / "toktuner_app.py"
    entry.write_text(
        "from toktuner.gui import main\n"
        "import sys\n"
        "sys.exit(main())\n", encoding="utf-8")

    ico = ROOT / "assets" / "icon.ico"
    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--onefile",
        "--windowed",                 # clean GUI / CLI launch
        "--name", "toktuner",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
        # tkinter is stdlib but PyInstaller needs telling for hidden imports
        "--hidden-import", "tkinter",
        "--hidden-import", "tkinter.ttk",
        "--hidden-import", "tkinter.filedialog",
        "--hidden-import", "tkinter.messagebox",
    ]
    if ico.is_file() and platform.system() == "Windows":
        args += ["--icon", str(ico)]
    if (ROOT / "assets").is_dir():
        args += ["--add-data", f"{ROOT / 'assets'}{os.pathsep}assets"]
    args.append(str(entry))

    print(" ".join(args))
    r = subprocess.run(args)
    entry.unlink(missing_ok=True)

    if r.returncode != 0:
        return r.returncode

    out_name = "toktuner.exe" if platform.system() == "Windows" else "toktuner"
    exe = ROOT / "dist" / out_name
    if exe.is_file():
        print(f"\nBuilt {exe}  ({exe.stat().st_size / 2**20:.1f} MB)")
        return 0
    print("build finished but the executable was not found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

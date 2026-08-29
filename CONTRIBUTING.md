# Contributing to `TokTuner`

Thank you for your interest in contributing to `TokTuner`!

`TokTuner` is built to be a fast, zero-dependency, mathematically rigorous flag compiler for `llama.cpp`. To preserve this, all contributions must follow a few core engineering principles.

---

## Core Development Rules

1. **Zero External Runtime Dependencies**:
   * The `toktuner` runtime and CLI must depend **only on the Python Standard Library**.
   * Do not add PyPI dependencies to `dependencies = []` in `pyproject.toml`.
   * PyInstaller is permitted solely as a build-time tool for `build_exe.py`.

2. **Deterministic Arithmetic**:
   * No heuristics or random sampling.
   * Given the same GGUF header and hardware specification, `plan.py` must return the exact same mathematical optimum every time.

3. **No Model Loading**:
   * Never read tensor payload data during planning.
   * All analysis must operate exclusively on GGUF metadata, header offsets, and hardware metrics in milliseconds.

4. **Preserve Conservative Failure Modes**:
   * Underestimating memory causes silent KV cache spilling or OOM crashes.
   * If metadata is unknown, uncertain, or ambiguous, state the assumption clearly in `notes` or `warnings` rather than guessing quietly.

---

## Development Setup

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/HarinManiK/TokTuner.git
cd TokTuner
pip install -e .
```

---

## Running the Test Suite

Before submitting a Pull Request, all three test suites must pass:

```bash
# 1. Synthetic architecture suite (36 checks)
python tests/test_universal.py

# 2. Ground-truth measured allocations (13 checks)
python tests/test_ground_truth.py

# 3. Adversarial audit suite (356 checks)
python tests/audit.py
```

---

## Adding Support for a New Architecture

When `llama.cpp` adds support for a new model architecture:

1. **Update `toktuner/arch.py`**:
   * Add any new metadata keys to `detect_attention()` or `analyse_kv()`.
   * Add exact formulas for per-token KV growth and state sizes.
2. **Update `toktuner/tensors.py`**:
   * Ensure new tensor names are properly recognized by `classify()` (distinguishing always-resident weights vs movable FFN/expert weights).
3. **Add Synthetic Tests in `tests/test_universal.py`**:
   * Add a synthetic test case matching the canonical GGUF specification for the new architecture.

---

## Pull Request Guidelines

1. Ensure all tests pass cleanly.
2. Keep code clean, type-annotated, and properly formatted.
3. Include an explanation in your PR description of any new architectural metadata or mathematical derivation changes.

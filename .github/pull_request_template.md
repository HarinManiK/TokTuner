## Description
Briefly describe the changes, bug fixes, or architectural support introduced in this PR.

## Checklist
- [ ] No external runtime dependencies added (Pure Python standard library preserved).
- [ ] Ran `python tests/test_universal.py` (All synthetic architecture tests pass).
- [ ] Ran `python tests/test_ground_truth.py` (All ground-truth allocation tests pass).
- [ ] Ran `python tests/audit.py` (Adversarial and flag syntax tests pass).
- [ ] Added new test cases for any newly introduced GGUF architectures or metadata keys.
- [ ] Maintained conservative memory limits and failure modes (no silent assumptions).

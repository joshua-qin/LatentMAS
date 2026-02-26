#!/usr/bin/env python3
"""Minimal test for no-vllm path. Safe on login node: no model download, no GPU, ~1 second."""
import sys
sys.path.insert(0, "/n/home03/jqin/LatentMAS")

# 1) models.py: vllm import is try/except; must not crash
from models import _HAS_VLLM
print("models import OK, _HAS_VLLM =", _HAS_VLLM)

# 2) Method and run entrypoints import without requiring vllm
from methods.latent_mas import LatentMASMethod
print("methods.latent_mas import OK")

# 3) If vllm were required unconditionally, we would have failed above
if _HAS_VLLM:
    print("(vllm is installed; you can use --use_vllm)")
else:
    print("(vllm not installed; runs without --use_vllm will use HF backend)")

print("PASS: no-vllm path is usable.")

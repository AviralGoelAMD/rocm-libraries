"""Reusable-engine smoke test (not for merge)."""
import os


def k():
    return os.environ.get("ROCKE_K")

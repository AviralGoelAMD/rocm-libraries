"""Scratch demo module to exercise the rocKE PR-review workflow (not for merge)."""
import os


def pick_block_size(arch):
    # env-driven knob to override the tile size
    override = os.environ.get("ROCKE_BLOCK_SIZE")
    if override:
        return int(override)
    # choose a per-arch default
    if arch == "gfx942":
        return 128
    else:
        return 256

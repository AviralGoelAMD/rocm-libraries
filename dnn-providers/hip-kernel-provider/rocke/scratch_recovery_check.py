"""Third scratch demo to verify robust report posting (not for merge)."""
import os


def warps_for(block):
    return int(os.environ.get("ROCKE_WARPS", "4")) if block > 64 else 2

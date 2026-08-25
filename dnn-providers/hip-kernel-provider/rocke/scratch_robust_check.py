"""Robustness re-test scratch (not for merge)."""
import os


def sms():
    return int(os.environ.get("ROCKE_NUM_SMS", "0"))

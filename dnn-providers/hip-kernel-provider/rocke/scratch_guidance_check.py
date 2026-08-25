"""Second scratch demo to verify private-guidance staging (not for merge)."""


def tile_for(heads):
    return 64 if heads <= 8 else 128

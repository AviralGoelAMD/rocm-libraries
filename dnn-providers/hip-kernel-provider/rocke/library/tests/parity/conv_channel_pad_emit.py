#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_channel_pad_emit.py -- Python reference emitter for the
# conv_channel_pad parity harness. Selects one of N sampled ChannelPadSpec
# configs by argv[1], builds the kernel via build_conv_channel_pad(arch='gfx950')
# and prints the .ll so it can be byte-compared with conv_channel_pad_emit.c.
#
# The kernel is shape-generic (rows are runtime args), so the dispatcher's stem
# convs collapse onto the (c_in, c_pad, dtype) configs below.
from kernels.common.conv_channel_pad import ChannelPadSpec, build_conv_channel_pad
from _emit_common import run_emit


def _spec(idx: int) -> ChannelPadSpec:
    if idx == 0:  # C=3 stem (R50 7x7 s2 C3->64, MBv2 3x3 C3->32, VGG 3x3 C3->64), fp16
        return ChannelPadSpec(c_in=3, c_pad=8, dtype="fp16")
    if idx == 1:  # C=3 stem, bf16
        return ChannelPadSpec(c_in=3, c_pad=8, dtype="bf16")
    if idx == 2:  # gray C=1 stem (7x7 C1->64), fp16
        return ChannelPadSpec(c_in=1, c_pad=8, dtype="fp16")
    if idx == 3:  # gray C=1 stem, bf16
        return ChannelPadSpec(c_in=1, c_pad=8, dtype="bf16")
    if idx == 4:  # c_pad=4, block 128
        return ChannelPadSpec(c_in=3, c_pad=4, dtype="bf16", block_size=128)
    if idx == 5:  # c_pad=2, block 64
        return ChannelPadSpec(c_in=1, c_pad=2, dtype="fp16", block_size=64)
    if idx == 6:  # must-reject: c_pad=16
        return ChannelPadSpec(c_in=3, c_pad=16, dtype="bf16")
    if idx == 7:  # must-reject: c_in == c_pad
        return ChannelPadSpec(c_in=8, c_pad=8, dtype="fp16")
    if idx == 8:  # must-reject: dtype f32
        return ChannelPadSpec(c_in=3, c_pad=8, dtype="f32")
    if idx == 9:  # must-reject: block_size 96
        return ChannelPadSpec(c_in=3, c_pad=8, dtype="bf16", block_size=96)
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_conv_channel_pad,
        usage="usage: conv_channel_pad_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())

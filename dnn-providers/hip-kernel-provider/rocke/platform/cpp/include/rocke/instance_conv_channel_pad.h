/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_channel_pad.h -- C99 port of the channel zero-pad
 * pre-pass kernel library/kernels/common/conv_channel_pad.py.
 *
 *   Python (conv_channel_pad.py)          C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   ChannelPadSpec (frozen dataclass)     rocke_conv_channel_pad_spec_t
 *   ChannelPadSpec.kernel_name()          rocke_conv_channel_pad_kernel_name(...)
 *   is_valid_channel_pad_spec(spec)       rocke_conv_channel_pad_is_valid_spec(...)
 *   build_conv_channel_pad(spec, arch)    rocke_build_conv_channel_pad(...)
 *   channel_pad_grid(spec, x, w)          rocke_conv_channel_pad_grid(...)
 *   channel_pad_signature(spec)           rocke_conv_channel_pad_signature(...)
 *   (+ convenience: build -> lower .ll)   rocke_conv_channel_pad_lower_to_llvm(...)
 */
#ifndef ROCKE_INSTANCE_CONV_CHANNEL_PAD_H
#define ROCKE_INSTANCE_CONV_CHANNEL_PAD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ChannelPadSpec:
 *   c_in: int
 *   c_pad: int
 *   dtype: str = "bf16"
 *   block_size: int = 256
 *   name: str = "rocke_conv_channel_pad" */
typedef struct rocke_conv_channel_pad_spec
{
    int c_in;
    int c_pad;
    const char* dtype; /* "fp16" | "bf16"; default "bf16" */
    int block_size; /* default 256 */
    const char* name; /* default "rocke_conv_channel_pad" */
} rocke_conv_channel_pad_spec_t;

/* Defaulted record (c_in = c_pad = 0; caller fills them). */
rocke_conv_channel_pad_spec_t rocke_conv_channel_pad_spec_default(void);

/* kernel_name_join(name, f"c{c_in}to{c_pad}", dtype, f"b{block_size}"). */
rocke_status_t rocke_conv_channel_pad_kernel_name(const rocke_conv_channel_pad_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap);

/* is_valid_channel_pad_spec(spec) -> (ok, reason). Reason strings are
 * byte-identical to the Python; "ok" on success. Safe with reason == NULL. */
bool rocke_conv_channel_pad_is_valid_spec(const rocke_conv_channel_pad_spec_t* spec,
                                          char* reason,
                                          size_t reason_cap);

/* build_conv_channel_pad(spec, arch) into an already-initialised builder `b`.
 * Returns b->kernel, or NULL with b's sticky error set ("invalid channel-pad
 * spec: <reason>" on a rejected spec). `arch` is unused (arch-neutral). */
rocke_kernel_def_t* rocke_build_conv_channel_pad(rocke_ir_builder_t* b,
                                                 const rocke_conv_channel_pad_spec_t* spec,
                                                 const char* arch);

/* Init `b` with spec.kernel_name(), then build. Caller frees `b`. */
rocke_kernel_def_t* rocke_build_conv_channel_pad_new(rocke_ir_builder_t* b,
                                                     const rocke_conv_channel_pad_spec_t* spec,
                                                     const char* arch);

/* channel_pad_grid: ((x_rows + w_rows + block_size - 1) // block_size, 1, 1).
 * ROCKE_ERR_VALUE on NULL args or block_size <= 0. */
rocke_status_t rocke_conv_channel_pad_grid(const rocke_conv_channel_pad_spec_t* spec,
                                           int x_rows,
                                           int w_rows,
                                           int out[3]);

/* channel_pad_signature: ptr X, W, Xp, Wp (dtype) + scalar x_rows, w_rows (i32).
 * Writes 6 entries (strings in `arena`). ROCKE_ERR_VALUE on NULL / out_cap < 6. */
rocke_status_t rocke_conv_channel_pad_signature(rocke_arena_t* arena,
                                                const rocke_conv_channel_pad_spec_t* spec,
                                                rocke_sig_entry_t* out_items,
                                                size_t out_cap,
                                                size_t* out_count);

/* Build + lower to LLVM .ll. `arch` NULL => "gfx950". *out_ll is malloc'd on
 * ROCKE_OK (caller free()s); on failure a diagnostic goes to err. */
rocke_status_t rocke_conv_channel_pad_lower_to_llvm(const rocke_conv_channel_pad_spec_t* spec,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_CHANNEL_PAD_H */

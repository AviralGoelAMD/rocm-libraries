/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_fmha_bwd.h -- C99 port of the FMHA backward-pass kernel instance
 * builder rocke/instances/common/fmha_bwd.py.
 *
 * Computes the Flash-Attention backward gradients dQ / dK / dV with a
 * warp-distributed scalar body (one wave64 warp per CTA; lane t owns head-dim
 * slot t*EPT+k). The recompute-from-(M,L) path issues NO MFMA atoms and
 * accumulates the three gradients through f32 global atomics, so the emitted IR
 * is arch-portable; `arch` only feeds is_valid_spec's per-WG thread / wave-size
 * validation against rocke_arch_target.
 *
 *   Python (fmha_bwd.py)                   C99 (this header)
 *   ------------------------------------   --------------------------------------
 *   @dataclass(frozen=True) FmhaBwdSpec    rocke_fmha_bwd_spec_t
 *     .kernel_name()                       rocke_fmha_bwd_kernel_name(...)
 *   is_valid_spec(spec, arch)              rocke_fmha_bwd_is_valid_spec(...)
 *   build_fmha_bwd(spec, arch)             rocke_build_fmha_bwd(...)
 *   fmha_bwd_grid(spec)                    rocke_fmha_bwd_grid(...)
 *   fmha_bwd_signature(spec)               rocke_fmha_bwd_signature(...)
 *   (+ convenience: build -> lower .ll)    rocke_fmha_bwd_lower_to_llvm(...)
 *
 * The build-entry mirrors the gemm_universal.c pattern: the FmhaKernelBuilder
 * owns its own rocke_ir_builder_t, so rocke_build_fmha_bwd takes a caller-supplied
 * (uninitialised) rocke_fmha_kernel_builder_t whose lifetime the caller controls
 * (free with rocke_fmha_kernel_builder_free). On any invalid spec the build
 * returns NULL with the builder's sticky error set (read with
 * rocke_ir_builder_error on kb->b).
 *
 * Error model mirrors the rest of the C port: build routes through the
 * sticky-error IRBuilder; the validity gate returns a bool + a reason string;
 * the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_FMHA_BWD_H
#define ROCKE_INSTANCE_FMHA_BWD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/helper_rocke.instances.common._fmha_common.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * FmhaBwdSpec
 * ------------------------------------------------------------------ *
 *
 * @dataclass(frozen=True)
 * class FmhaBwdSpec:
 *     common: FmhaCommonSpec
 *     seqlen_q: int
 *     seqlen_k: int
 *     name: str = "rocke_fmha_bwd"
 *     use_mfma_body: bool = False
 *     output_grad_dtype: str = "f32"
 *     bwd_mode: str = "atomic"            ("atomic" | "mfma")
 *     mfma_dkdv_waves: int = 4
 *     mfma_dkdv_ktiles: int = 1
 *     mfma_dq_waves: int = 4
 *     mfma_dq_qtiles: int = 2
 *     mfma_lds_pad: int = 16
 *     mfma_waves_per_eu: int = 2           (0 = no attribute)
 *     mfma_exp2_fast: bool = True
 *     mfma_natural_rows: bool = True
 *     mfma_causal_order: str = "lpt"      ("none" | "lpt" | "pair")
 *     mfma_causal_split: bool = True
 *     mfma_fused_dq: bool = False
 *
 * String fields are referenced as-is (not copied); keep them alive for the
 * spec's use. rocke_fmha_bwd_spec_default fills every dataclass default. */
typedef struct rocke_fmha_bwd_spec
{
    rocke_fmha_common_spec_t common;
    int seqlen_q;
    int seqlen_k;
    const char* name; /* default "rocke_fmha_bwd" */
    bool use_mfma_body; /* default false             */
    const char* output_grad_dtype; /* default "f32"; mfma: "f32" or common.dtype */
    const char* bwd_mode; /* default "atomic"          */
    int mfma_dkdv_waves;
    int mfma_dkdv_ktiles;
    int mfma_dq_waves;
    int mfma_dq_qtiles;
    int mfma_lds_pad;
    int mfma_waves_per_eu;
    bool mfma_exp2_fast;
    bool mfma_natural_rows;
    const char* mfma_causal_order;
    bool mfma_causal_split;
    bool mfma_fused_dq;
} rocke_fmha_bwd_spec_t;

/* FmhaBwdSpec(common, seqlen_q, seqlen_k): construct from the required fields
 * with every dataclass default. */
rocke_fmha_bwd_spec_t
    rocke_fmha_bwd_spec_default(rocke_fmha_common_spec_t common, int seqlen_q, int seqlen_k);

/* FmhaBwdSpec.kernel_name(): join name + H{head_size} + HQ{nq} + HK{nk} +
 * dtype + Q{seqlen_q} + K{seqlen_k} + mask_mode via kernel_name_join. Writes
 * NUL-terminated into `out` (capacity out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE
 * (buffer too small). */
rocke_status_t
    rocke_fmha_bwd_kernel_name(const rocke_fmha_bwd_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". On a
 * reject, `reason` (if non-NULL, capacity reason_cap) receives the structured
 * message; returns false. On accept returns true and writes "ok". */
bool rocke_fmha_bwd_is_valid_spec(const rocke_fmha_bwd_spec_t* spec,
                                  const char* arch,
                                  char* reason,
                                  size_t reason_cap);

/* build_fmha_bwd(spec, arch). Builds the IR into the supplied (uninitialised)
 * FmhaKernelBuilder `kb`, exactly as the Python build does, and returns the
 * kernel (kb->b.kernel) on success or NULL with kb->b's sticky error set.
 * `arch` NULL => "gfx950". The caller owns `kb` and frees it with
 * rocke_fmha_kernel_builder_free. On an invalid spec returns NULL (the reason is
 * recorded on kb->b's sticky error). */
rocke_kernel_def_t* rocke_build_fmha_bwd(rocke_fmha_kernel_builder_t* kb,
                                         const rocke_fmha_bwd_spec_t* spec,
                                         const char* arch);

/* fmha_bwd_grid(spec) -> (seqlen_q, num_query_heads, 1). */
void rocke_fmha_bwd_grid(const rocke_fmha_bwd_spec_t* spec, int* gx, int* gy, int* gz);

/* fmha_bwd_signature(spec): declare the ABI on a throwaway FmhaKernelBuilder
 * and emit the SignatureBuilder shape. The caller supplies an uninitialised
 * `kb` (owns + frees it) and an `arena` backing the signature storage. On
 * ROCKE_OK *out_items / *out_count hold the arena-owned array. */
rocke_status_t rocke_fmha_bwd_signature(rocke_fmha_kernel_builder_t* kb,
                                        const rocke_fmha_bwd_spec_t* spec,
                                        rocke_arena_t* arena,
                                        const rocke_sig_entry_t** out_items,
                                        size_t* out_count);

/* Convenience: given a spec, build the FMHA backward kernel and lower to LLVM
 * .ll text. `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is left
 * NULL and (if err!=NULL, capacity err_cap) a diagnostic is written. Internally
 * owns and frees its FmhaKernelBuilder. */
rocke_status_t rocke_fmha_bwd_lower_to_llvm(const rocke_fmha_bwd_spec_t* spec,
                                            const char* arch,
                                            rocke_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap);

/* ------------------------------------------------------------------ *
 * Launch plan: build_fmha_bwd_kernels(spec, arch)
 * ------------------------------------------------------------------ *
 *
 * bwd_mode="atomic": one stage (build_fmha_bwd). bwd_mode="mfma": the
 * build_fmha_bwd_mfma stages "pre", "dkdv" and (unless mfma_fused_dq) "dq",
 * launched in order on one stream. */
#define ROCKE_FMHA_BWD_MAX_STAGES 3

typedef struct rocke_fmha_bwd_stage_info
{
    const char* stage; /* "atomic" | "pre" | "dkdv" | "dq" (static) */
    int grid[3];
    int block[3];
} rocke_fmha_bwd_stage_info_t;

/* Fill `out` (capacity ROCKE_FMHA_BWD_MAX_STAGES) with the launch plan and
 * write the stage count to *out_count. On a spec the Python plan rejects
 * returns ROCKE_ERR_VALUE with the message in `err` (capacity err_cap). */
rocke_status_t rocke_fmha_bwd_launch_plan(const rocke_fmha_bwd_spec_t* spec,
                                          const char* arch,
                                          rocke_fmha_bwd_stage_info_t* out,
                                          int* out_count,
                                          char* err,
                                          size_t err_cap);

/* build_fmha_bwd_kernels(spec, arch)[stage][0]: build stage `stage` of the
 * launch plan into the uninitialised `kb` (caller frees). NULL + sticky
 * error on a rejected spec or an out-of-range stage. */
rocke_kernel_def_t* rocke_build_fmha_bwd_kernels_stage(rocke_fmha_kernel_builder_t* kb,
                                                       const rocke_fmha_bwd_spec_t* spec,
                                                       const char* arch,
                                                       int stage);

/* build_fmha_bwd_kernels(spec, arch)[stage][1]: the stage's signature
 * (declared on the uninitialised throwaway `kb`; caller frees). */
rocke_status_t rocke_fmha_bwd_stage_signature(rocke_fmha_kernel_builder_t* kb,
                                              const rocke_fmha_bwd_spec_t* spec,
                                              const char* arch,
                                              int stage,
                                              rocke_arena_t* arena,
                                              const rocke_sig_entry_t** out_items,
                                              size_t* out_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_FMHA_BWD_H */

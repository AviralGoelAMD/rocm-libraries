/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/fmha_bwd_emit.c -- C-side emitter for the FMHA-backward parity
 * harness. Selects one sampled config by argv[1], mirroring fmha_bwd_emit.py:
 * 0..5 are bwd_mode="atomic" specs built via rocke_build_fmha_bwd; 6.. are
 * (bwd_mode="mfma" spec, launch-plan stage) pairs built via
 * rocke_build_fmha_bwd_kernels_stage. Lowers via rocke_lower_kernel_to_llvm_ex
 * (arch gfx950, flavor AUTO), printing the .ll to stdout so the two outputs
 * can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcmp (C++ build) */

#include "rocke/instance_fmha_bwd.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* bwd_mode="mfma" rows (fmha_bwd_emit.py _MFMA): every listed stage is one
 * config index, in row order. */
typedef struct
{
    int h, hq, hk;
    const char* dtype;
    rocke_fmha_mask_mode_t mask;
    int seqlen;
    int knobs; /* 0 defaults; 1 pair/old knobs/f16 grads; 2 fused k4; 3 fused k2;
                * 4 bf16 grads; 5 order none */
    int nstages;
    int stages[3];
} mfma_row_t;

static const mfma_row_t k_mfma[] = {
    {128, 2, 2, "bf16", ROCKE_FMHA_MASK_NONE, 256, 0, 3, {0, 1, 2}},
    {128, 4, 2, "bf16", ROCKE_FMHA_MASK_CAUSAL, 256, 0, 3, {0, 1, 2}},
    {128, 2, 2, "f16", ROCKE_FMHA_MASK_CAUSAL, 256, 1, 2, {1, 2, 0}},
    {128, 2, 2, "bf16", ROCKE_FMHA_MASK_NONE, 256, 2, 2, {0, 1, 0}},
    {128, 4, 2, "bf16", ROCKE_FMHA_MASK_CAUSAL, 256, 3, 1, {1, 0, 0}},
    {128, 2, 2, "bf16", ROCKE_FMHA_MASK_NONE, 256, 4, 2, {1, 2, 0}},
    {64, 1, 1, "bf16", ROCKE_FMHA_MASK_CAUSAL, 128, 5, 2, {1, 2, 0}},
};

/* Fill `spec` / `stage` for mfma config `idx` (0-based past the atomic ones). */
static int make_mfma_spec(int idx, rocke_fmha_bwd_spec_t* spec, int* stage)
{
    for(size_t r = 0; r < sizeof(k_mfma) / sizeof(k_mfma[0]); ++r)
    {
        const mfma_row_t* m = &k_mfma[r];
        if(idx >= m->nstages)
        {
            idx -= m->nstages;
            continue;
        }
        rocke_fmha_shape_t shape = rocke_fmha_shape_default(m->h, m->hq, m->hk);
        rocke_fmha_common_spec_t common = rocke_fmha_common_spec_default(shape);
        common.dtype = m->dtype;
        common.mask_mode = m->mask;
        *spec = rocke_fmha_bwd_spec_default(common, m->seqlen, m->seqlen);
        spec->bwd_mode = "mfma";
        switch(m->knobs)
        {
        case 1:
            spec->mfma_causal_order = "pair";
            spec->mfma_causal_split = false;
            spec->output_grad_dtype = "f16";
            spec->mfma_natural_rows = false;
            spec->mfma_exp2_fast = false;
            spec->mfma_waves_per_eu = 0;
            spec->mfma_dq_qtiles = 1;
            spec->mfma_lds_pad = 8;
            break;
        case 2:
            spec->mfma_fused_dq = true;
            spec->mfma_dkdv_ktiles = 4;
            break;
        case 3:
            spec->mfma_fused_dq = true;
            spec->mfma_dkdv_ktiles = 2;
            break;
        case 4:
            spec->output_grad_dtype = "bf16";
            break;
        case 5:
            spec->mfma_causal_order = "none";
            break;
        default:
            break;
        }
        *stage = m->stages[idx];
        return 0;
    }
    return -1;
}

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown.
 * `*stage` is -1 for the atomic configs. */
static int make_spec(int idx, rocke_fmha_bwd_spec_t* spec, int* stage)
{
    rocke_fmha_shape_t shape;
    rocke_fmha_common_spec_t common;

    *stage = -1;
    if(idx >= 6)
    {
        return make_mfma_spec(idx - 6, spec, stage);
    }
    switch(idx)
    {
    case 0:
        shape = rocke_fmha_shape_default(64, 4, 4);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_bwd_spec_default(common, 16, 16);
        break;
    case 1:
        shape = rocke_fmha_shape_default(128, 8, 4);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_bwd_spec_default(common, 32, 64);
        break;
    case 2:
        shape = rocke_fmha_shape_default(256, 16, 8);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        *spec = rocke_fmha_bwd_spec_default(common, 64, 64);
        break;
    case 3:
        shape = rocke_fmha_shape_default(64, 2, 2);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_SLIDING_WINDOW;
        common.sliding_window = 16;
        *spec = rocke_fmha_bwd_spec_default(common, 128, 128);
        break;
    case 4:
        shape = rocke_fmha_shape_default(192, 8, 2);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "bf16";
        common.mask_mode = ROCKE_FMHA_MASK_NONE;
        *spec = rocke_fmha_bwd_spec_default(common, 256, 256);
        break;
    case 5:
        shape = rocke_fmha_shape_default(128, 1, 1);
        common = rocke_fmha_common_spec_default(shape);
        common.dtype = "f16";
        common.mask_mode = ROCKE_FMHA_MASK_CAUSAL;
        *spec = rocke_fmha_bwd_spec_default(common, 128, 256);
        break;
    default:
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_fmha_bwd_spec_t spec;
    int stage = -1;
    if(make_spec(idx, &spec, &stage) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_fmha_kernel_builder_t kb;
    rocke_kernel_def_t* kernel = stage < 0
                                     ? rocke_build_fmha_bwd(&kb, &spec, "gfx950")
                                     : rocke_build_fmha_bwd_kernels_stage(&kb, &spec, "gfx950", stage);
    if(!kernel)
    {
        fprintf(stderr,
                "build failed: err=%s\n",
                rocke_ir_builder_error(rocke_fmha_kernel_builder_builder(&kb)));
        rocke_fmha_kernel_builder_free(&kb);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char lerr[ROCKE_ERR_MSG_CAP];
        lerr[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text, lerr, sizeof lerr);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, lerr);
            rocke_fmha_kernel_builder_free(&kb);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            rocke_fmha_kernel_builder_free(&kb);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        rocke_fmha_kernel_builder_free(&kb);
        return 2;
    }
    rocke_fmha_kernel_builder_free(&kb);
    return ret;
}

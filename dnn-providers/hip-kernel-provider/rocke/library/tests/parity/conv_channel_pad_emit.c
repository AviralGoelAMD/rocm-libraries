/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_channel_pad_emit.c -- C-side emitter for the conv_channel_pad
 * parity harness. Selects one of N sampled spec configs by argv[1], builds
 * rocke_conv_channel_pad_spec_t identically to conv_channel_pad_emit.py, builds
 * via rocke_build_conv_channel_pad_new + lowers via rocke_lower_kernel_to_llvm
 * (arch gfx950, flavor AUTO) and prints the .ll to stdout.
 *
 * Optional argv[2] selects the output mode:
 *   "ll"     (default) - lower to LLVM and print
 *   "ir"               - print ck.dsl.ir/v1 serialization
 *   "verify"           - run verifier; print each diagnostic on its own line
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_channel_pad.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill `spec` for config index `idx`. Returns 0 on success, -1 if unknown. */
static int make_spec(int idx, rocke_conv_channel_pad_spec_t* spec)
{
    *spec = rocke_conv_channel_pad_spec_default();
    switch(idx)
    {
    case 0: /* C=3 stem (R50 7x7 s2 C3->64, MBv2 3x3 C3->32, VGG 3x3 C3->64), fp16 */
        spec->c_in = 3;
        spec->c_pad = 8;
        spec->dtype = "fp16";
        break;
    case 1: /* C=3 stem, bf16 */
        spec->c_in = 3;
        spec->c_pad = 8;
        spec->dtype = "bf16";
        break;
    case 2: /* gray C=1 stem (7x7 C1->64), fp16 */
        spec->c_in = 1;
        spec->c_pad = 8;
        spec->dtype = "fp16";
        break;
    case 3: /* gray C=1 stem, bf16 */
        spec->c_in = 1;
        spec->c_pad = 8;
        spec->dtype = "bf16";
        break;
    case 4: /* c_pad=4, block 128 */
        spec->c_in = 3;
        spec->c_pad = 4;
        spec->dtype = "bf16";
        spec->block_size = 128;
        break;
    case 5: /* c_pad=2, block 64 */
        spec->c_in = 1;
        spec->c_pad = 2;
        spec->dtype = "fp16";
        spec->block_size = 64;
        break;
    case 6: /* must-reject: c_pad=16 */
        spec->c_in = 3;
        spec->c_pad = 16;
        spec->dtype = "bf16";
        break;
    case 7: /* must-reject: c_in == c_pad */
        spec->c_in = 8;
        spec->c_pad = 8;
        spec->dtype = "fp16";
        break;
    case 8: /* must-reject: dtype f32 */
        spec->c_in = 3;
        spec->c_pad = 8;
        spec->dtype = "f32";
        break;
    case 9: /* must-reject: block_size 96 */
        spec->c_in = 3;
        spec->c_pad = 8;
        spec->dtype = "bf16";
        spec->block_size = 96;
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
        fprintf(stderr, "usage: %s <config_index> [ll|ir|verify]\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }

    rocke_conv_channel_pad_spec_t spec;
    if(make_spec(idx, &spec) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_conv_channel_pad_new(&b, &spec, "gfx950");
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, "gfx950", &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* text = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &text);
        if(st != ROCKE_OK || !text)
        {
            fprintf(stderr, "serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(text, stdout);
        free(text);
    }
    else
    { /* verify */
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

    rocke_ir_builder_free(&b);
    return 0;
}

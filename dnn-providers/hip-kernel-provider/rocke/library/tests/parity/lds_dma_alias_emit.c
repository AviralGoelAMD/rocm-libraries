/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/lds_dma_alias_emit.c -- C-side emitter for LDS-DMA alias rings
 * (rocke_b_lds_alias_ring + the rocke_lds_alias_t tag of the *_alias builder
 * calls) and the counted barrier rocke_b_sync_keep_lds_dma. Builds each kernel
 * identically to lds_dma_alias_emit.py so run_diff.py can byte-compare the two
 * engines' .ll; see that file for the config rationale.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

typedef struct alias_config
{
    int tagged;
    int b_slots;
    int any_slot;
    int keep_lds_dma;
    int wide_tr; /* step-3 B read: ds_read_tr16_b128 instead of _b64 */
    const char* arch;
} alias_config_t;

static const alias_config_t CONFIGS[] = {
    {0, 2, 0, 0, 0, "gfx950"},
    {1, 2, 0, 1, 0, "gfx950"},
    {1, 3, 1, 0, 0, "gfx950"},
    {1, 2, 0, 63, 1, "gfx950"},
};

static const int NUM_CONFIGS = (int)(sizeof(CONFIGS) / sizeof(CONFIGS[0]));

enum
{
    ROWS = 32,
    COLS = 64,
    SLOT_BYTES = ROWS * COLS * 2,
    DWORDS = 4
};

static rocke_value_t*
    ptr_param(rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem, bool readonly)
{
    rocke_param_opts_t o;
    memset(&o, 0, sizeof(o));
    o.noalias = true;
    o.noalias_set = true;
    if(readonly)
    {
        o.readonly = true;
        o.readonly_set = true;
    }
    o.align = 16;
    o.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &o);
}

/* Python tag(ring, slot): NULL when untagged; slot < 0 = any_slot. */
static const rocke_lds_alias_t* tag(rocke_lds_alias_t* out, int ring, int slot)
{
    if(ring < 0)
        return NULL;
    out->ring = ring;
    out->slot = slot;
    return out;
}

static rocke_value_t* f32_lane(rocke_ir_builder_t* b, rocke_value_t* v, int i)
{
    return rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, v, i));
}

/* One builder call per statement, in the Python emitter's order, so both
 * engines assign the same SSA ids. */
static void build(rocke_ir_builder_t* b, const alias_config_t* c)
{
    rocke_lds_alias_t t;
    const rocke_type_t* bf16 = rocke_bf16();
    rocke_value_t* src = ptr_param(b, "S", bf16, true);
    rocke_value_t* dst = ptr_param(b, "D", rocke_f32(), false);
    int a_shape[3] = {2, ROWS, COLS};
    int b_shape[3] = {c->b_slots, ROWS, COLS};
    rocke_value_t* a_lds = rocke_b_smem_alloc(b, bf16, a_shape, 3, "A_lds");
    rocke_value_t* b_lds = rocke_b_smem_alloc(b, bf16, b_shape, 3, "B_lds");
    int ring_a = c->tagged ? rocke_b_lds_alias_ring(b, a_lds, 2) : -1;
    int ring_b = c->tagged ? rocke_b_lds_alias_ring(b, b_lds, c->b_slots) : -1;
    int rs0 = c->any_slot ? ROCKE_LDS_ALIAS_ANY_SLOT : 0;
    int rs1 = c->any_slot ? ROCKE_LDS_ALIAS_ANY_SLOT : 1;

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* rsrc = rocke_b_buffer_rsrc(b, src, rocke_b_const_i32(b, 1 << 20));
    rocke_value_t* voff = rocke_b_mul(b, tid, rocke_b_const_i32(b, DWORDS * 4));
    rocke_value_t* soff = rocke_b_const_i32(b, 0);
    rocke_value_t* a_base = rocke_b_smem_addr_of(b, a_lds);
    rocke_value_t* b_base = rocke_b_smem_addr_of(b, b_lds);
    rocke_value_t* slot_off = rocke_b_zext(b, rocke_b_const_i32(b, SLOT_BYTES), rocke_i64());
    rocke_value_t* a_slot1 = rocke_b_smem_ptr_add(b, a_base, slot_off);
    rocke_value_t* b_slot1 = rocke_b_smem_ptr_add(b, b_base, slot_off);

    /* 1. Fill slot 0 of both rings, then a full barrier. */
    rocke_b_async_buffer_load_lds_addr_alias(
        b, rsrc, a_base, voff, soff, DWORDS, 0, tag(&t, ring_a, 0));
    rocke_b_global_load_lds_alias(b, src, voff, b_base, DWORDS * 4, 0, tag(&t, ring_b, 0));
    rocke_b_sync(b);

    /* 2. Fill slot 1 (an ordinary load in between) while slot 0 is read. */
    rocke_b_async_buffer_load_lds_alias(
        b, rsrc, a_slot1, voff, soff, DWORDS, 0, tag(&t, ring_a, 1));
    rocke_value_t* gidx = rocke_b_add(b, tid, rocke_b_const_i32(b, 4096));
    rocke_value_t* g = rocke_b_global_load(b, src, gidx, bf16, 2);
    rocke_b_async_buffer_load_lds_addr_alias(
        b, rsrc, b_slot1, voff, soff, DWORDS, 0, tag(&t, ring_b, 1));
    rocke_value_t* row = rocke_b_div(b, tid, rocke_b_const_i32(b, 8));
    rocke_value_t* lane8 = rocke_b_mod(b, tid, rocke_b_const_i32(b, 8));
    rocke_value_t* col = rocke_b_mul(b, lane8, rocke_b_const_i32(b, 8));
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* idx0[3] = {c0, row, col};
    rocke_value_t* idx0c1[3] = {c0, row, c1};
    rocke_value_t* idx1[3] = {c1, row, col};
    rocke_value_t* x0 = rocke_b_smem_load_vN_alias(b, a_lds, idx0, 3, bf16, 8, tag(&t, ring_a, rs0));
    rocke_value_t* x1
        = rocke_b_smem_load_vN_alias(b, a_lds, idx0c1, 3, bf16, 1, tag(&t, ring_a, rs0));
    rocke_value_t* y0 = rocke_b_ds_read_tr16_b64_alias(b, b_lds, idx0, 3, bf16, tag(&t, ring_b, rs0));
    rocke_value_t* x0f = f32_lane(b, x0, 7);
    rocke_value_t* acc = rocke_b_fadd(b, x0f, f32_lane(b, x1, 0));
    acc = rocke_b_fadd(b, acc, f32_lane(b, y0, 3));
    acc = rocke_b_fadd(b, acc, rocke_b_cast_to_f32(b, g));

    /* 3. Publish A slot 1 (B slot 1 may stay in flight), then read it. */
    rocke_b_sync_keep_lds_dma(b, c->keep_lds_dma);
    rocke_value_t* z0
        = c->wide_tr
              ? rocke_b_ds_read_tr16_b128_alias(b, b_lds, idx0, 3, bf16, tag(&t, ring_b, rs0))
              : rocke_b_ds_read_tr16_b64_alias(b, b_lds, idx0, 3, bf16, tag(&t, ring_b, rs0));
    rocke_value_t* z1 = rocke_b_smem_load_vN_alias(b, a_lds, idx1, 3, bf16, 8, tag(&t, ring_a, rs1));
    /* b64 returns 4 lanes, b128 8: lane 5 only exists in the wide form. */
    acc = rocke_b_fadd(b, acc, f32_lane(b, z0, c->wide_tr ? 5 : 1));
    acc = rocke_b_fadd(b, acc, f32_lane(b, z1, 2));
    rocke_b_global_store(b, dst, tid, acc, 4);
    rocke_b_ret(b);
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(
            stderr, "usage: %s <config_index 0..%d> [ll|ir|verify]\n", argv[0], NUM_CONFIGS - 1);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    if(strcmp(mode, "ll") != 0 && strcmp(mode, "ir") != 0 && strcmp(mode, "verify") != 0)
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    if(idx < 0 || idx >= NUM_CONFIGS)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "lds_dma_alias") != ROCKE_OK)
    {
        fprintf(stderr, "builder init failed\n");
        return 1;
    }
    /* Python: b.kernel.attrs["max_workgroup_size"] = 256 */
    rocke_attr_set_int(&b, &b.kernel->attrs, "max_workgroup_size", 256);
    build(&b, &CONFIGS[idx]);

    if(!rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "builder error: %s\n", rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return 1;
    }

    rocke_kernel_def_t* kernel = rocke_ir_builder_kernel(&b);
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = 0;
        rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
            kernel, ROCKE_LLVM_FLAVOR_AUTO, CONFIGS[idx].arch, &llvm_text, err, sizeof err);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d err=%s\n", (int)st, err);
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

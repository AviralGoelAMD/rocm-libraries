/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C99 port of the channel zero-pad pre-pass kernel
 * library/kernels/common/conv_channel_pad.py. See
 * rocke/instance_conv_channel_pad.h for the public symbol map.
 *
 * rocke_build_conv_channel_pad mirrors build_conv_channel_pad op-for-op (same
 * param order, same const order) so the lowered .ll is byte-identical to the
 * Python reference.
 */
#include "rocke/instance_conv_channel_pad.h"

#include <stdio.h> /* snprintf */
#include <string.h> /* memset, memcpy, strcmp, strlen */

#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.io.h" /* rocke_b_io_ir_type, rocke_b_load_scalar */
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#define ROCKE_CHANNEL_PAD_MAX_C 8

rocke_conv_channel_pad_spec_t rocke_conv_channel_pad_spec_default(void)
{
    rocke_conv_channel_pad_spec_t s;
    s.c_in = 0;
    s.c_pad = 0;
    s.dtype = "bf16";
    s.block_size = 256;
    s.name = "rocke_conv_channel_pad";
    return s;
}

rocke_status_t rocke_conv_channel_pad_kernel_name(const rocke_conv_channel_pad_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap)
{
    char cstr[48];
    char bstr[24];
    const char* parts[3];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(cstr, sizeof(cstr), "c%dto%d", spec->c_in, spec->c_pad);
    snprintf(bstr, sizeof(bstr), "b%d", spec->block_size);
    parts[0] = cstr;
    parts[1] = spec->dtype;
    parts[2] = bstr;
    return rocke_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_channel_pad_spec(spec) -> (ok, reason)
 * ===================================================================== */

bool rocke_conv_channel_pad_is_valid_spec(const rocke_conv_channel_pad_spec_t* spec,
                                          char* reason,
                                          size_t reason_cap)
{
    char buf[160];
    const char* dt;

    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    dt = spec->dtype != NULL ? spec->dtype : "";
    if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
    {
        snprintf(buf, sizeof(buf), "unsupported dtype '%s'; fp16 or bf16 only", dt);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->c_pad != 2 && spec->c_pad != 4 && spec->c_pad != 8)
    {
        snprintf(buf,
                 sizeof(buf),
                 "c_pad must be 2, 4 or 8 (one vector store), got %d",
                 spec->c_pad);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(!(1 <= spec->c_in && spec->c_in < spec->c_pad))
    {
        snprintf(buf,
                 sizeof(buf),
                 "need 1 <= c_in < c_pad, got c_in=%d c_pad=%d",
                 spec->c_in,
                 spec->c_pad);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->block_size <= 0 || spec->block_size % 64 != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_size must be a positive multiple of 64, got %d",
                 spec->block_size);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  build_conv_channel_pad(spec, arch)
 * ===================================================================== */

/* copy_row(src, dst, r): c_in scalar loads, zero fill, one c_pad-wide store. */
static void rocke_channel_pad_copy_row(rocke_ir_builder_t* b,
                                       const rocke_conv_channel_pad_spec_t* spec,
                                       const rocke_type_t* elem,
                                       rocke_value_t* src,
                                       rocke_value_t* dst,
                                       rocke_value_t* r,
                                       rocke_value_t* c_in,
                                       rocke_value_t* c_pad,
                                       rocke_value_t* zero)
{
    rocke_value_t* vals[ROCKE_CHANNEL_PAD_MAX_C];
    rocke_value_t* base = rocke_b_mul(b, r, c_in);
    rocke_value_t* off;
    rocke_value_t* packed;
    int c;

    for(c = 0; c < spec->c_in; ++c)
    {
        rocke_value_t* idx = c == 0 ? base : rocke_b_add(b, base, rocke_b_const_i32(b, c));
        vals[c] = rocke_b_load_scalar(b, src, idx, spec->dtype);
    }
    for(c = spec->c_in; c < spec->c_pad; ++c)
    {
        vals[c] = zero;
    }
    off = rocke_b_mul(b, r, c_pad);
    packed = rocke_b_vec_pack(b, vals, spec->c_pad, elem);
    rocke_b_global_store_vN(b, dst, off, packed, spec->c_pad, 2 * spec->c_pad);
}

static rocke_value_t* rocke_channel_pad_ptr_param(rocke_ir_builder_t* b,
                                                  const char* name,
                                                  const rocke_type_t* elem,
                                                  bool writeonly,
                                                  int align)
{
    rocke_param_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.noalias = true;
    opts.noalias_set = true;
    if(writeonly)
    {
        opts.writeonly = true;
        opts.writeonly_set = true;
    }
    else
    {
        opts.readonly = true;
        opts.readonly_set = true;
    }
    opts.align = align;
    opts.align_set = true;
    return rocke_b_param(b, name, rocke_ptr_type(b, elem, "global"), &opts);
}

rocke_kernel_def_t* rocke_build_conv_channel_pad(rocke_ir_builder_t* b,
                                                 const rocke_conv_channel_pad_spec_t* spec,
                                                 const char* arch)
{
    const rocke_type_t* elem;
    rocke_value_t *X, *W, *Xp, *Wp, *x_rows, *w_rows;
    rocke_value_t *row, *zero, *c_in, *c_pad, *w_row, *in_w;

    (void)arch; /* no arch-specific instruction is emitted */
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    {
        char reason[160];
        if(!rocke_conv_channel_pad_is_valid_spec(spec, reason, sizeof(reason)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid channel-pad spec: %s", reason);
            return NULL;
        }
    }
    elem = rocke_b_io_ir_type(b, spec->dtype);
    if(elem == NULL)
    {
        return NULL;
    }
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    X = rocke_channel_pad_ptr_param(b, "X", elem, false, 2);
    W = rocke_channel_pad_ptr_param(b, "W", elem, false, 2);
    Xp = rocke_channel_pad_ptr_param(b, "Xp", elem, true, 16);
    Wp = rocke_channel_pad_ptr_param(b, "Wp", elem, true, 16);
    x_rows = rocke_b_param(b, "x_rows", rocke_i32(), NULL);
    w_rows = rocke_b_param(b, "w_rows", rocke_i32(), NULL);

    /* row = add(mul(block_id_x, const(block_size)), thread_id_x) */
    {
        rocke_value_t* bid = rocke_b_block_id_x(b);
        rocke_value_t* cbs = rocke_b_const_i32(b, spec->block_size);
        rocke_value_t* m = rocke_b_mul(b, bid, cbs);
        rocke_value_t* tid = rocke_b_thread_id_x(b);
        row = rocke_b_add(b, m, tid);
    }
    zero = rocke_b_cast_f32_to(b, rocke_b_const_f32(b, 0.0), elem);
    c_in = rocke_b_const_i32(b, spec->c_in);
    c_pad = rocke_b_const_i32(b, spec->c_pad);

    /* with b.scf_if(row < x_rows): copy_row(X, Xp, row) */
    {
        rocke_if_t gate = rocke_b_scf_if(b, rocke_b_cmp_lt(b, row, x_rows));
        rocke_b_region_enter(b, gate.then_region);
        rocke_channel_pad_copy_row(b, spec, elem, X, Xp, row, c_in, c_pad, zero);
        rocke_b_region_leave(b);
    }
    w_row = rocke_b_sub(b, row, x_rows);
    {
        rocke_value_t* ge = rocke_b_cmp_ge(b, row, x_rows);
        rocke_value_t* lt = rocke_b_cmp_lt(b, w_row, w_rows);
        in_w = rocke_b_land(b, ge, lt);
    }
    {
        rocke_if_t gate = rocke_b_scf_if(b, in_w);
        rocke_b_region_enter(b, gate.then_region);
        rocke_channel_pad_copy_row(b, spec, elem, W, Wp, w_row, c_in, c_pad, zero);
        rocke_b_region_leave(b);
    }
    return rocke_ir_builder_status(b) == ROCKE_OK ? b->kernel : NULL;
}

rocke_kernel_def_t* rocke_build_conv_channel_pad_new(rocke_ir_builder_t* b,
                                                     const rocke_conv_channel_pad_spec_t* spec,
                                                     const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_conv_channel_pad_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_conv_channel_pad(b, spec, arch);
    });
}

/* ===================================================================== *
 *  grid / signature
 * ===================================================================== */

rocke_status_t rocke_conv_channel_pad_grid(const rocke_conv_channel_pad_spec_t* spec,
                                           int x_rows,
                                           int w_rows,
                                           int out[3])
{
    int total;
    if(spec == NULL || out == NULL || spec->block_size <= 0)
    {
        return ROCKE_ERR_VALUE;
    }
    total = x_rows + w_rows;
    out[0] = (total + spec->block_size - 1) / spec->block_size;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_conv_channel_pad_signature(rocke_arena_t* arena,
                                                const rocke_conv_channel_pad_spec_t* spec,
                                                rocke_sig_entry_t* out_items,
                                                size_t out_cap,
                                                size_t* out_count)
{
    rocke_signature_builder_t sb;
    const rocke_sig_entry_t* items;
    size_t count;
    rocke_status_t st;

    if(arena == NULL || spec == NULL || out_items == NULL || out_cap < 6)
    {
        return ROCKE_ERR_VALUE;
    }
    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
    {
        return st;
    }
    rocke_signature_builder_ptr(&sb, "X", spec->dtype, NULL);
    rocke_signature_builder_ptr(&sb, "W", spec->dtype, NULL);
    rocke_signature_builder_ptr(&sb, "Xp", spec->dtype, NULL);
    rocke_signature_builder_ptr(&sb, "Wp", spec->dtype, NULL);
    rocke_signature_builder_scalar(&sb, "x_rows", "i32");
    rocke_signature_builder_scalar(&sb, "w_rows", "i32");
    st = rocke_signature_builder_build(&sb, &items, &count);
    if(st != ROCKE_OK)
    {
        return st;
    }
    if(count > out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    memcpy(out_items, items, count * sizeof(*out_items));
    if(out_count != NULL)
    {
        *out_count = count;
    }
    return ROCKE_OK;
}

/* ===================================================================== *
 *  lower_to_llvm -- build + lower convenience. Owns its IRBuilder.
 * ===================================================================== */

static void rocke_channel_pad_copy_err(char* err, size_t err_cap, const char* m)
{
    size_t n;
    if(err == NULL || err_cap == 0)
    {
        return;
    }
    if(m == NULL)
    {
        m = "conv_channel_pad lower failed";
    }
    n = strlen(m);
    if(n >= err_cap)
    {
        n = err_cap - 1;
    }
    memcpy(err, m, n);
    err[n] = '\0';
}

rocke_status_t rocke_conv_channel_pad_lower_to_llvm(const rocke_conv_channel_pad_spec_t* spec,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        rocke_channel_pad_copy_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    kernel = rocke_build_conv_channel_pad_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        rocke_channel_pad_copy_err(err, err_cap, rocke_ir_builder_error(&b));
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

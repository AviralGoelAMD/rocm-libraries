// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_dgrad_stream.cpp -- C++ port of
 * DirectDepthwiseDgradStreamSpec / is_valid_depthwise_dgrad_stream_spec /
 * build_direct_depthwise_dgrad_streaming
 * (library/kernels/common/conv_direct_grouped.py).
 *
 * Ho-streaming depthwise dgrad (cpg = kpg = 1, any stride). A runtime loop
 * walks the dY rows KH at a time; each row adds its taps into stride * KH
 * circular dX accumulator slots per column and flushes the dX rows it
 * completes. W is preloaded into registers; the dY rows are prefetched PF
 * rows ahead and carried across iterations. A second loop flushes the dX rows
 * past the last dY row.
 *
 * Byte-identical to the Python source: every rocke_b_* call below is issued in
 * the order Python evaluates the matching expression (arguments left to
 * right, innermost first).
 */
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#include "rocke/error_boundary.hpp"

namespace
{

/* Python dilation_reason(p) (file-local twin of the static helper in
 * conv_direct_grouped_spec_and_validate.cpp; same text). */
bool dgs_dilation_reason(const rocke_direct_conv_problem_t* p, char* why, size_t why_cap)
{
    if(p->dil_h != 1 || p->dil_w != 1)
    {
        snprintf(why,
                 why_cap,
                 "this direct conv kernel has no dilation (got %dx%d); only the non-grouped "
                 "forward kernel computes a dilated convolution",
                 p->dil_h,
                 p->dil_w);
        return true;
    }
    return false;
}

/* Copy `msg` into (err, err_cap), NUL-terminated and truncated. */
void dgs_set_err(char* err, size_t err_cap, const char* msg)
{
    size_t n;
    if(err == NULL || err_cap == 0)
        return;
    if(msg == NULL)
        msg = "";
    n = strlen(msg);
    if(n >= err_cap)
        n = err_cap - 1;
    memcpy(err, msg, n);
    err[n] = '\0';
}

} // namespace

/* ===================================================================== *
 *  Spec helpers
 * ===================================================================== */

rocke_direct_depthwise_dgrad_stream_spec_t rocke_direct_depthwise_dgrad_stream_spec_default(void)
{
    rocke_direct_depthwise_dgrad_stream_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.problem = rocke_direct_conv_problem_default();
    spec.name = "direct_depthwise_dgrad_stream";
    spec.block_w = 8;
    spec.block_waves = 1;
    spec.wave_size = 64;
    return spec;
}

int rocke_direct_depthwise_dgrad_stream_threads_per_block(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

int rocke_direct_depthwise_dgrad_stream_block_ch(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec)
{
    return spec->block_waves * spec->wave_size;
}

rocke_status_t rocke_direct_depthwise_dgrad_stream_kernel_name(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec, char* out, size_t out_cap)
{
    char prob_short[128];
    const char* parts[3];
    char bw_buf[32];
    char bwv_buf[32];

    if(spec == NULL || out == NULL || out_cap == 0)
        return ROCKE_ERR_VALUE;
    if(rocke_direct_conv_problem_short(&spec->problem, prob_short, sizeof(prob_short)) != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    /* kernel_name_join(name, p.short(), f"bw{block_w}", f"bw{block_waves}wv",
     *                  flags={"bf16": dtype=="bf16"}) */
    snprintf(bw_buf, sizeof(bw_buf), "bw%d", spec->block_w);
    snprintf(bwv_buf, sizeof(bwv_buf), "bw%dwv", spec->block_waves);
    parts[0] = prob_short;
    parts[1] = bw_buf;
    parts[2] = bwv_buf;
    {
        const char* flag_names[1] = {"bf16"};
        int flag_on[1];
        const char* dt = spec->problem.dtype ? spec->problem.dtype : "fp16";
        flag_on[0] = (strcmp(dt, "bf16") == 0) ? 1 : 0;
        return rocke_kernel_name_join(
            spec->name, parts, 3, flag_names, flag_on, 1, out, out_cap, NULL);
    }
}

rocke_status_t rocke_direct_depthwise_dgrad_stream_validate(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec, char* reason, size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    p = &spec->problem;
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "DirectDepthwiseDgradStreamSpec: unsupported dtype '%s'; "
                         "expected fp16 or bf16",
                         dt);
            return ROCKE_ERR_VALUE;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "DirectDepthwiseDgradStreamSpec requires cpg=kpg=1 (got %d, %d)",
                     p->cpg,
                     p->kpg);
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

bool rocke_direct_depthwise_dgrad_stream_is_valid_spec(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    const rocke_direct_conv_problem_t* p;
    if(spec == NULL)
    {
        if(reason && reason_cap > 0)
            strncpy(reason, "null spec", reason_cap);
        return false;
    }
    if(arch == NULL)
        arch = "gfx950";
    if(rocke_archtarget_from_gfx(arch) == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    p = &spec->problem;
    {
        char dil_why[256];
        if(dgs_dilation_reason(p, dil_why, sizeof dil_why))
        {
            if(reason != NULL && reason_cap > 0)
                snprintf(reason, reason_cap, "%s", dil_why);
            return false;
        }
    }
    {
        const char* dt = p->dtype ? p->dtype : "fp16";
        if(strcmp(dt, "fp16") != 0 && strcmp(dt, "bf16") != 0)
        {
            if(reason && reason_cap > 0)
                snprintf(reason,
                         reason_cap,
                         "DirectDepthwiseDgradStreamSpec: unsupported dtype '%s'; "
                         "expected fp16 or bf16",
                         dt);
            return false;
        }
    }
    if(p->cpg != 1 || p->kpg != 1)
    {
        if(reason && reason_cap > 0)
            snprintf(reason, reason_cap, "requires cpg=kpg=1 (got %d, %d)", p->cpg, p->kpg);
        return false;
    }
    if(reason && reason_cap > 0)
    {
        strncpy(reason, "ok", reason_cap);
        reason[reason_cap - 1] = '\0';
    }
    return true;
}

/* ===================================================================== *
 *  build_direct_depthwise_dgrad_streaming
 * ===================================================================== */

namespace
{

/* Python's non-negative modulo. */
int dgs_pymod(int a, int n) { return ((a % n) + n) % n; }

/* Per-kernel values the Python closures capture. */
struct dgs_ctx
{
    rocke_ir_builder_t* b;
    const rocke_dconv_params_t* params;
    int is_bf16;
    int stride;
    int PAD;
    int BLOCK_W;
    int n_slots;
    int n_cols;
    rocke_value_t* c0;
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* zero_f32;
    rocke_value_t* c_stride_v;
    rocke_value_t* c_Ho;
    rocke_value_t* a_rsrc;
    rocke_value_t* d_rsrc;
    rocke_value_t* ch_off;
    rocke_value_t* dy_n_off;
    rocke_value_t* dx_n_off;
    rocke_value_t** col_terms;
    rocke_value_t** st_terms;
};

/* Python: load_row(y_v) -- raw dY values of row y_v, one per column. */
void dgs_load_row(const dgs_ctx* c, rocke_value_t* y_v, rocke_value_t** out)
{
    rocke_ir_builder_t* b = c->b;
    rocke_value_t* ho_ok = rocke_b_cmp_lt(b, y_v, c->c_Ho);
    rocke_value_t* y_off = rocke_b_mul(b, y_v, c->params->p_A_stride_hi);
    rocke_value_t* row_off = rocke_b_mul(b, rocke_b_add(b, c->dy_n_off, y_off), c->c_half_bytes);
    int col;
    for(col = 0; col < c->n_cols; ++col)
    {
        rocke_value_t* rc = rocke_b_add(b, row_off, c->col_terms[col]);
        rocke_value_t* s_term = rocke_b_select(b, ho_ok, rc, c->oob_sentinel);
        rocke_value_t* v_off = rocke_b_add(b, c->ch_off, s_term);
        out[col] = c->is_bf16 ? rocke_b_buffer_load_bf16(b, c->a_rsrc, v_off, c->c0)
                              : rocke_b_buffer_load_f16(b, c->a_rsrc, v_off, c->c0);
    }
}

/* Python: flush(y_v, jj, slots_flat) -- store row y_v's window, reset slots. */
void dgs_flush(const dgs_ctx* c, rocke_value_t* y_v, int jj, rocke_value_t** slots_flat)
{
    rocke_ir_builder_t* b = c->b;
    int d, j;
    for(d = 0; d < c->stride; ++d)
    {
        int slot = dgs_pymod(jj * c->stride - c->PAD + d, c->n_slots);
        rocke_value_t* y_s = rocke_b_mul(b, y_v, c->c_stride_v);
        rocke_value_t* hi_flush_v = rocke_b_add(b, y_s, rocke_b_const_i32(b, d - c->PAD));
        rocke_value_t* ge = rocke_b_cmp_ge(b, hi_flush_v, c->c0);
        rocke_value_t* lt = rocke_b_cmp_lt(b, hi_flush_v, c->params->p_Hi);
        rocke_value_t* hi_flush_ok = rocke_b_land(b, ge, lt);
        rocke_value_t* hi_off = rocke_b_mul(b, hi_flush_v, c->params->p_D_stride_ho);
        rocke_value_t* row_off
            = rocke_b_mul(b, rocke_b_add(b, c->dx_n_off, hi_off), c->c_half_bytes);
        for(j = 0; j < c->BLOCK_W; ++j)
        {
            int flat = slot * c->BLOCK_W + j;
            rocke_value_t* rs = rocke_b_add(b, row_off, c->st_terms[j]);
            rocke_value_t* safe_d = rocke_b_select(b, hi_flush_ok, rs, c->oob_sentinel);
            if(c->is_bf16)
                rocke_b_buffer_store_bf16(
                    b, c->d_rsrc, safe_d, c->c0, rocke_b_trunc_f32_to_bf16(b, slots_flat[flat]));
            else
                rocke_b_buffer_store_f16(
                    b, c->d_rsrc, safe_d, c->c0, rocke_b_trunc_f32_to_f16(b, slots_flat[flat]));
            slots_flat[flat] = c->zero_f32;
        }
    }
}

} // namespace

rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_stream(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_stream_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    const rocke_direct_conv_problem_t* p;
    rocke_dconv_params_t params;
    dgs_ctx c;
    int BLOCK_W, WAVE, BLOCK_WAVES, THREADS, BLOCK_CH, stride, KH, KW, PAD;
    int n_slots, n_cols, n_acc, PF, r_const, s_const, j, d, k, i;
    rocke_value_t *c_wave, *tid, *wave_id, *lane, *bx, *by, *n, *wi_tile_start, *ch, *ch_ok;
    rocke_value_t *b_rsrc, *c_Wi, *c_Wo, *c_KH_v, *c1, *y_end_v, *n_groups_v, *n_fma_groups_v;
    rocke_value_t** weights_f32;
    const rocke_tensor_descriptor_t* b_desc;

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx950";
    if(rocke_direct_depthwise_dgrad_stream_validate(spec, reason, sizeof reason) != ROCKE_OK
       || !rocke_direct_depthwise_dgrad_stream_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return NULL;
    }

    p = &spec->problem;
    BLOCK_W = spec->block_w;
    WAVE = spec->wave_size;
    BLOCK_WAVES = spec->block_waves;
    THREADS = BLOCK_WAVES * WAVE;
    BLOCK_CH = BLOCK_WAVES * WAVE;
    stride = p->stride;
    KH = p->KH;
    KW = p->KW;
    PAD = p->PAD;
    n_slots = stride * KH;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);

    rocke_dconv_emit_params(b, &params, "dgrad", rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16"));

    memset(&c, 0, sizeof c);
    c.b = b;
    c.params = &params;
    c.is_bf16 = (p->dtype && strcmp(p->dtype, "bf16") == 0) ? 1 : 0;
    c.stride = stride;
    c.PAD = PAD;
    c.BLOCK_W = BLOCK_W;
    c.n_slots = n_slots;

    c.c0 = rocke_b_const_i32(b, 0);
    c.c_half_bytes = rocke_b_const_i32(b, 2);
    c_wave = rocke_b_const_i32(b, WAVE);
    c.oob_sentinel = rocke_b_const_i32(b, (int32_t)(((int64_t)1 << 31) - 1));
    c.zero_f32 = rocke_b_const_f32(b, 0.0f);

    tid = rocke_b_thread_id_x(b);
    wave_id = rocke_b_div(b, tid, c_wave);
    lane = rocke_b_mod(b, tid, c_wave);

    bx = rocke_b_block_id_x(b);
    by = rocke_b_block_id_y(b);
    n = rocke_b_block_id_z(b);
    wi_tile_start = rocke_b_mul(b, bx, rocke_b_const_i32(b, BLOCK_W));
    {
        rocke_value_t* outer = rocke_b_mul(b, by, rocke_b_const_i32(b, BLOCK_CH));
        rocke_value_t* mul_wave = rocke_b_mul(b, wave_id, c_wave);
        rocke_value_t* inner = rocke_b_add(b, mul_wave, lane);
        ch = rocke_b_add(b, outer, inner);
    }
    ch_ok = rocke_b_cmp_lt(b, ch, params.p_groups);

    c.a_rsrc = rocke_b_buffer_rsrc(b, params.A, params.A_bytes);
    b_rsrc = rocke_b_buffer_rsrc(b, params.Bp, params.B_bytes);
    c.d_rsrc = rocke_b_buffer_rsrc(b, params.D, params.D_bytes);

    /* W descriptor: B[groups, KH, KW, 1] KRSC. */
    {
        static const char* const b_coords[4] = {"k", "r", "s", "c"};
        int b_len[4];
        b_len[0] = p->groups;
        b_len[1] = KH;
        b_len[2] = KW;
        b_len[3] = 1;
        b_desc = rocke_tensor_descriptor_naive(b, "B", b_len, 4, NULL, b_coords, 4);
    }

    /* Preload W[ch, r, s, 0] into f32 registers. */
    weights_f32 = (rocke_value_t**)alloca((size_t)(KH * KW) * sizeof(rocke_value_t*));
    for(r_const = 0; r_const < KH; ++r_const)
    {
        for(s_const = 0; s_const < KW; ++s_const)
        {
            const char* w_names[4] = {"k", "r", "s", "c"};
            rocke_value_t* w_vals[4];
            rocke_value_t* w_off = NULL;
            rocke_value_t *safe_w, *w_h;
            w_vals[0] = ch;
            w_vals[1] = rocke_b_const_i32(b, r_const);
            w_vals[2] = rocke_b_const_i32(b, s_const);
            w_vals[3] = c.c0;
            rocke_transforms_descriptor_offset(b, b_desc, w_names, w_vals, 4, &w_off, NULL);
            safe_w = rocke_b_select(
                b, ch_ok, rocke_b_mul(b, w_off, c.c_half_bytes), c.oob_sentinel);
            w_h = c.is_bf16 ? rocke_b_buffer_load_bf16(b, b_rsrc, safe_w, c.c0)
                            : rocke_b_buffer_load_f16(b, b_rsrc, safe_w, c.c0);
            weights_f32[r_const * KW + s_const]
                = rocke_b_select(b, ch_ok, rocke_b_cast_to_f32(b, w_h), c.zero_f32);
        }
    }

    c_Wi = params.p_Wi;
    c_Wo = params.p_Wo;
    c.c_Ho = params.p_Ho;

    c.c_stride_v = rocke_b_const_i32(b, stride);
    c_KH_v = rocke_b_const_i32(b, KH);
    c1 = rocke_b_const_i32(b, 1);
    {
        rocke_value_t* hp = rocke_b_add(b, params.p_Hi, rocke_b_const_i32(b, PAD));
        rocke_value_t* hps = rocke_b_add(b, hp, rocke_b_const_i32(b, stride - 1));
        y_end_v = rocke_b_div(b, hps, c.c_stride_v);
    }
    n_groups_v = rocke_b_div(b, rocke_b_add(b, y_end_v, rocke_b_const_i32(b, KH - 1)), c_KH_v);
    n_fma_groups_v
        = rocke_b_div(b, rocke_b_add(b, c.c_Ho, rocke_b_const_i32(b, KH - 1)), c_KH_v);

    n_acc = n_slots * BLOCK_W;

    c.ch_off = rocke_b_mul(b, ch, c.c_half_bytes);

    /* Store terms: one per dX column. */
    c.st_terms = (rocke_value_t**)alloca((size_t)BLOCK_W * sizeof(rocke_value_t*));
    for(j = 0; j < BLOCK_W; ++j)
    {
        rocke_value_t* wi = rocke_b_add(b, wi_tile_start, rocke_b_const_i32(b, j));
        rocke_value_t* wi_ok = rocke_b_land(b, rocke_b_cmp_lt(b, wi, c_Wi), ch_ok);
        rocke_value_t* wi_m = rocke_b_mul(b, wi, params.p_D_stride_wo);
        rocke_value_t* wi_off = rocke_b_add(b, rocke_b_mul(b, wi_m, c.c_half_bytes), c.ch_off);
        c.st_terms[j] = rocke_b_select(b, wi_ok, wi_off, c.oob_sentinel);
    }

    /* Column terms: one per distinct d = j - s. */
    n_cols = BLOCK_W + KW - 1;
    c.n_cols = n_cols;
    c.col_terms = (rocke_value_t**)alloca((size_t)n_cols * sizeof(rocke_value_t*));
    for(d = 1 - KW; d < BLOCK_W; ++d)
    {
        rocke_value_t* wi_p_s = rocke_b_add(b, wi_tile_start, rocke_b_const_i32(b, PAD + d));
        rocke_value_t* div_ok = NULL;
        rocke_value_t *wo, *col_ok, *col_off;
        if(stride > 1)
        {
            div_ok = rocke_b_cmp_eq(b, rocke_b_mod(b, wi_p_s, c.c_stride_v), c.c0);
            wo = rocke_b_div(b, wi_p_s, c.c_stride_v);
        }
        else
        {
            wo = wi_p_s;
        }
        {
            rocke_value_t* ge = rocke_b_cmp_ge(b, wi_p_s, c.c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wo, c_Wo);
            col_ok = rocke_b_land(b, ge, lt);
        }
        if(stride > 1)
            col_ok = rocke_b_land(b, col_ok, div_ok);
        col_off = rocke_b_mul(b, rocke_b_mul(b, wo, params.p_A_stride_wi), c.c_half_bytes);
        c.col_terms[d - (1 - KW)] = rocke_b_select(b, col_ok, col_off, c.oob_sentinel);
    }
    c.dy_n_off = rocke_b_mul(b, n, params.p_A_stride_n);
    c.dx_n_off = rocke_b_mul(b, n, params.p_D_stride_n);

    PF = (16 + n_cols - 1) / n_cols;
    if(KH < PF)
        PF = KH;

    {
        const int n_row_args = PF * n_cols;
        const int num_iargs = n_acc + n_row_args;
        rocke_iter_arg_t* iargs
            = (rocke_iter_arg_t*)alloca((size_t)num_iargs * sizeof(rocke_iter_arg_t));
        char(*names)[48] = (char(*)[48])alloca((size_t)num_iargs * 48);
        rocke_value_t** inits = (rocke_value_t**)alloca((size_t)num_iargs * sizeof(rocke_value_t*));
        /* window: PF + 1 rows of n_cols (one appended before one is popped). */
        rocke_value_t** window
            = (rocke_value_t**)alloca((size_t)((PF + 1) * n_cols) * sizeof(rocke_value_t*));
        rocke_value_t** slots_flat
            = (rocke_value_t**)alloca((size_t)n_acc * sizeof(rocke_value_t*));
        rocke_value_t** dy_row = (rocke_value_t**)alloca((size_t)n_cols * sizeof(rocke_value_t*));
        rocke_value_t** yields
            = (rocke_value_t**)alloca((size_t)num_iargs * sizeof(rocke_value_t*));
        rocke_for_t fma_loop, flush_loop;
        int slot, s, jj;

        for(slot = 0; slot < n_slots; ++slot)
            for(j = 0; j < BLOCK_W; ++j)
            {
                int idx = slot * BLOCK_W + j;
                snprintf(names[idx], 48, "dgs_acc_s%d_w%d", slot, j);
                inits[idx] = c.zero_f32;
            }
        for(k = 0; k < PF; ++k)
        {
            dgs_load_row(&c, rocke_b_const_i32(b, k), inits + n_acc + k * n_cols);
            for(d = 0; d < n_cols; ++d)
                snprintf(names[n_acc + k * n_cols + d], 48, "dgs_dy_r%d_d%d", k, d);
        }
        for(i = 0; i < num_iargs; ++i)
        {
            iargs[i].name = names[i];
            iargs[i].init = inits[i];
        }

        fma_loop = rocke_b_scf_for_iter(b,
                                        c.c0,
                                        n_fma_groups_v,
                                        c1,
                                        iargs,
                                        num_iargs,
                                        "dgs_grp",
                                        /*unroll=*/false,
                                        /*elide_trailing_barrier=*/true);
        rocke_b_region_enter(b, fma_loop.body);
        {
            rocke_value_t* y_base;
            for(i = 0; i < n_acc; ++i)
                slots_flat[i] = fma_loop.iter_vars[i];
            for(i = 0; i < n_row_args; ++i)
                window[i] = fma_loop.iter_vars[n_acc + i];
            y_base = rocke_b_mul(b, fma_loop.iv, c_KH_v);

            for(jj = 0; jj < KH; ++jj)
            {
                rocke_value_t* y_v = rocke_b_add(b, y_base, rocke_b_const_i32(b, jj));
                /* Issue row y + PF, then compute row y (the oldest in flight). */
                dgs_load_row(
                    &c, rocke_b_add(b, y_v, rocke_b_const_i32(b, PF)), window + PF * n_cols);
                for(d = 0; d < n_cols; ++d)
                    dy_row[d] = rocke_b_cast_to_f32(b, window[d]);
                memmove(window, window + n_cols, (size_t)(PF * n_cols) * sizeof(rocke_value_t*));

                for(r_const = 0; r_const < KH; ++r_const)
                {
                    slot = dgs_pymod(jj * stride + r_const - PAD, n_slots);
                    for(s = 0; s < KW; ++s)
                        for(j = 0; j < BLOCK_W; ++j)
                        {
                            int flat = slot * BLOCK_W + j;
                            slots_flat[flat] = rocke_b_fma(b,
                                                           weights_f32[r_const * KW + s],
                                                           dy_row[j - s + KW - 1],
                                                           slots_flat[flat]);
                        }
                }

                dgs_flush(&c, y_v, jj, slots_flat);
            }

            /* Keep LLVM's SLP vectorizer from pairing the FMA chains into
             * v_pk_fma_f32 (see the Python source). */
            {
                const rocke_type_t* f32_ty = rocke_f32();
                rocke_inline_asm_opts_t opts;
                memset(&opts, 0, sizeof opts);
                opts.sideeffect = false;
                opts.sideeffect_set = true;
                for(i = 0; i < n_acc; ++i)
                {
                    rocke_op_t* asm_op
                        = rocke_b_inline_asm(b, "", "=v,0", &slots_flat[i], 1, &f32_ty, 1, &opts);
                    slots_flat[i] = asm_op ? asm_op->results[0] : NULL;
                }
            }
            for(i = 0; i < n_acc; ++i)
                yields[i] = slots_flat[i];
            for(i = 0; i < n_row_args; ++i)
                yields[n_acc + i] = window[i];
            rocke_b_scf_yield(b, yields, num_iargs);
        }
        rocke_b_region_leave(b);

        /* The windows past the last dY row: no tap, only dX rows to complete. */
        for(i = 0; i < n_acc; ++i)
        {
            snprintf(names[i], 48, "dgs_tail_acc_%d", i);
            iargs[i].name = names[i];
            iargs[i].init = fma_loop.op->results[i];
        }
        flush_loop = rocke_b_scf_for_iter(b,
                                          n_fma_groups_v,
                                          n_groups_v,
                                          c1,
                                          iargs,
                                          n_acc,
                                          "dgs_tail",
                                          /*unroll=*/false,
                                          /*elide_trailing_barrier=*/true);
        rocke_b_region_enter(b, flush_loop.body);
        {
            rocke_value_t* y_base;
            for(i = 0; i < n_acc; ++i)
                slots_flat[i] = flush_loop.iter_vars[i];
            y_base = rocke_b_mul(b, flush_loop.iv, c_KH_v);
            for(jj = 0; jj < KH; ++jj)
                dgs_flush(&c, rocke_b_add(b, y_base, rocke_b_const_i32(b, jj)), jj, slots_flat);
            rocke_b_scf_yield(b, slots_flat, n_acc);
        }
        rocke_b_region_leave(b);
    }

    return rocke_ir_builder_kernel(b);
}

/* Initialises `b` with spec.kernel_name(), then builds. Caller owns `b`. */
rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_stream_new(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_stream_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke_direct_depthwise_dgrad_stream_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_direct_depthwise_dgrad_stream(b, spec, arch);
    });
}

rocke_status_t rocke_direct_depthwise_dgrad_stream_lower_to_llvm(
    const rocke_direct_depthwise_dgrad_stream_spec_t* spec,
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
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        dgs_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx950";
    kernel = rocke_build_direct_depthwise_dgrad_stream_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        dgs_set_err(err,
                    err_cap,
                    (m != NULL && m[0] != '\0') ? m
                                                : "build_direct_depthwise_dgrad_streaming failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

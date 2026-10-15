// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * fmha_bwd_mfma.cpp -- C++ port of the bwd_mode="mfma" kernels of
 * library/kernels/common/fmha_bwd.py (build_fmha_bwd_mfma) and of its
 * launch plan build_fmha_bwd_kernels.
 *
 * Byte-identical builder-call sequence vs the Python: same ops, same order,
 * same operands. Python evaluates call arguments left to right; C++ does
 * not, so every builder call that feeds another call with more than one
 * IR-emitting argument is hoisted into a named temporary in Python order.
 *
 * Stages (Python names):
 *   _build_mfma_pre   -> mfma_pre   (D = rowsum(dO * O), Lse2, optional dQ zero)
 *   _build_mfma_dkdv  -> mfma_dkdv  (KV-major dK / dV, optional fused atomic dQ)
 *   _build_mfma_dq    -> mfma_dq    (Q-major dQ)
 */
#include "rocke/instance_fmha_bwd.h"

#include <stdio.h>
#include <string.h>

#include <functional>
#include <string>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

namespace
{

using V = rocke_value_t*;
using Vec = std::vector<V>;

constexpr int kStep = 32; /* _MFMA_BWD_STEP */
constexpr int kPreThreads = 256; /* _MFMA_PRE_THREADS */

struct Ctx
{
    rocke_fmha_kernel_builder_t* kb;
    rocke_ir_builder_t* b;
    const rocke_fmha_bwd_spec_t* spec;
    const char* op; /* MFMA op id */
    const rocke_type_t* dt; /* io_ir_type(common.dtype) */
};

inline V ci(rocke_ir_builder_t* b, int x)
{
    return rocke_b_const_i32(b, x);
}

/* --------------------------------------------------------------------- *
 *  spec checks (build_fmha_bwd_mfma / _mfma_bwd_grad_dtype / _setup)
 * --------------------------------------------------------------------- */

bool is_causal(const rocke_fmha_bwd_spec_t* s)
{
    return s->common.mask_mode == ROCKE_FMHA_MASK_CAUSAL;
}

bool str_eq(const char* a, const char* b)
{
    return a != NULL && strcmp(a, b) == 0;
}

/* _mfma_bwd_grad_dtype: "" on success, else the ValueError text. */
std::string grad_dtype_error(const rocke_fmha_bwd_spec_t* s)
{
    const char* g = s->output_grad_dtype ? s->output_grad_dtype : "";
    if(!str_eq(g, "f32") && !str_eq(g, s->common.dtype))
    {
        return std::string("fmha_bwd mfma: output_grad_dtype must be 'f32' or '") + s->common.dtype
               + "' (got '" + g + "')";
    }
    if(!str_eq(g, "f32") && s->mfma_fused_dq)
    {
        return "fmha_bwd mfma_fused_dq accumulates dQ with f32 atomics: output_grad_dtype must be "
               "'f32'";
    }
    return "";
}

/* The checks of build_fmha_bwd_mfma, in Python order. "" == ok. */
std::string mfma_plan_error(const rocke_fmha_bwd_spec_t* s, const char* arch)
{
    char reason[512];
    if(!rocke_fmha_bwd_is_valid_spec(s, arch, reason, sizeof(reason)))
    {
        return std::string("invalid fmha_bwd spec: ") + reason;
    }
    const int D = s->common.shape.head_size;
    const int bn = 16 * s->mfma_dkdv_waves * s->mfma_dkdv_ktiles;
    const int bq = 16 * s->mfma_dq_waves * s->mfma_dq_qtiles;
    if(bn <= 0 || bq <= 0 || D % 32 || s->seqlen_k % bn || s->seqlen_q % bq
       || s->seqlen_q % kStep)
    {
        char m[256];
        snprintf(m,
                 sizeof(m),
                 "fmha_bwd mfma: head_size %% 32, seqlen_k %% %d, seqlen_q %% %d must be 0",
                 bn,
                 bq > 32 ? bq : 32);
        return m;
    }
    const bool causal = is_causal(s);
    if((s->common.mask_mode != ROCKE_FMHA_MASK_NONE && !causal)
       || (s->seqlen_q != s->seqlen_k && causal))
    {
        return "fmha_bwd mfma: mask_mode none, or causal with seqlen_q == seqlen_k";
    }
    if(!str_eq(s->mfma_causal_order, "none") && !str_eq(s->mfma_causal_order, "lpt")
       && !str_eq(s->mfma_causal_order, "pair"))
    {
        return std::string("fmha_bwd mfma_causal_order: none | lpt | pair (got '")
               + (s->mfma_causal_order ? s->mfma_causal_order : "") + "')";
    }
    std::string g = grad_dtype_error(s);
    if(!g.empty())
    {
        return g;
    }
    const bool pair = causal && str_eq(s->mfma_causal_order, "pair");
    const int nkb = s->seqlen_k / bn, nqb = s->seqlen_q / bq;
    if(pair && (nkb % 2 || (nqb % 2 && !s->mfma_fused_dq)))
    {
        return "fmha_bwd mfma_causal_order='pair' needs an even number of K (and Q) blocks";
    }
    return "";
}

/* Initialise `kb` (so the caller can read the sticky error) and record `msg`. */
void fail(rocke_fmha_kernel_builder_t* kb, const rocke_fmha_bwd_spec_t* spec, const std::string& msg)
{
    char kname[1024];
    if(rocke_fmha_bwd_kernel_name(spec, kname, sizeof(kname)) != ROCKE_OK)
    {
        snprintf(kname, sizeof(kname), "rocke_fmha_bwd");
    }
    if(rocke_fmha_kernel_builder_init(kb, kname, &spec->common) == ROCKE_OK)
    {
        rocke_i_set_err(&kb->b, ROCKE_ERR_VALUE, "%s", msg.c_str());
    }
}

/* --------------------------------------------------------------------- *
 *  ABI declarations
 * --------------------------------------------------------------------- */

void add_strides(rocke_fmha_kernel_builder_t* kb, std::initializer_list<const char*> names)
{
    std::vector<const char*> v(names);
    rocke_fmha_kernel_builder_add_strides(kb, v.data(), v.size());
}

void declare_pre(rocke_fmha_kernel_builder_t* kb, bool fused)
{
    rocke_fmha_kernel_builder_add_tensor(kb, "O", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "dO", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_ptr(kb, "M_saved", "f32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "L_saved", "f32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "Dsum", "f32", false, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "Lse2", "f32", false, 4);
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    add_strides(kb, {"o", "do"});
    if(fused)
    {
        rocke_fmha_kernel_builder_add_ptr(kb, "dQ", "f32", false, 4);
        add_strides(kb, {"q"});
        rocke_fmha_kernel_builder_add_scalar(kb, "stride_dq_token", "i32");
    }
}

void declare_dkdv(rocke_fmha_kernel_builder_t* kb, bool fused, const char* gdt)
{
    rocke_fmha_kernel_builder_add_tensor(kb, "Q", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "K", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "V", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "dO", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_ptr(kb, "Lse2", "f32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "Dsum", "f32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "dK", gdt, false, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "dV", gdt, false, 4);
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_inv", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    add_strides(kb, {"q", "k", "v"});
    add_strides(kb, {"do"});
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_dk_token", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_dv_token", "i32");
    if(fused)
    {
        rocke_fmha_kernel_builder_add_ptr(kb, "dQ", "f32", false, 4);
        rocke_fmha_kernel_builder_add_scalar(kb, "stride_dq_token", "i32");
    }
}

void declare_dq(rocke_fmha_kernel_builder_t* kb, const char* gdt)
{
    rocke_fmha_kernel_builder_add_tensor(kb, "Q", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "K", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "V", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_tensor(kb, "dO", NULL, true, false, 16);
    rocke_fmha_kernel_builder_add_ptr(kb, "Lse2", "f32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "Dsum", "f32", true, 4);
    rocke_fmha_kernel_builder_add_ptr(kb, "dQ", gdt, false, 4);
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_log2", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "scale_inv", "f32");
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_q", "i32");
    rocke_fmha_kernel_builder_add_scalar(kb, "seqlen_k", "i32");
    add_strides(kb, {"q", "k", "v"});
    add_strides(kb, {"do"});
    rocke_fmha_kernel_builder_add_scalar(kb, "stride_dq_token", "i32");
}

/* --------------------------------------------------------------------- *
 *  shared helpers (_mfma_bwd_*)
 * --------------------------------------------------------------------- */

struct Lane
{
    V tid, lane, wave, m16, g;
};

/* _mfma_bwd_lane */
Lane mk_lane(rocke_ir_builder_t* b, int nw)
{
    Lane l;
    l.tid = rocke_b_thread_id_x(b);
    l.lane = nw > 1 ? rocke_b_mod(b, l.tid, ci(b, 64)) : l.tid;
    l.wave = nw > 1 ? rocke_b_to_sgpr_u32(b, rocke_b_div(b, l.tid, ci(b, 64))) : ci(b, 0);
    l.m16 = rocke_b_mod(b, l.lane, ci(b, 16));
    l.g = rocke_b_div(b, l.lane, ci(b, 16));
    return l;
}

/* _mfma_bwd_perm_row */
V perm_row(rocke_ir_builder_t* b, V m16, int t, bool nat)
{
    if(nat)
    {
        return rocke_b_add(b, m16, ci(b, 16 * t));
    }
    V d = rocke_b_div(b, m16, ci(b, 4));
    V m = rocke_b_mul(b, d, ci(b, 8));
    V r = rocke_b_mod(b, m16, ci(b, 4));
    V a = rocke_b_add(b, m, r);
    return rocke_b_add(b, a, ci(b, 4 * t));
}

/* _mfma_bwd_slot_rows -> base; *t_stride */
V slot_rows(rocke_ir_builder_t* b, V g, bool nat, int* t_stride)
{
    *t_stride = nat ? 16 : 4;
    return rocke_b_mul(b, g, ci(b, nat ? 4 : 8));
}

/* _MfmaBwdStager */
struct Stager
{
    rocke_ir_builder_t* b;
    const rocke_type_t* dt;
    Vec rows;
    V col;
    Vec lds;
};

bool stager_init(Stager& st,
                 rocke_ir_builder_t* b,
                 V tid,
                 int nw,
                 int head_size,
                 const rocke_type_t* dt,
                 int pad,
                 int nsrc)
{
    const int lpr = head_size / 8;
    const int rpp = 64 * nw / lpr;
    if(rpp <= 0 || rpp > kStep || kStep % rpp)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "mfma bwd staging: %d waves do not tile %d rows", nw, kStep);
        return false;
    }
    st.b = b;
    st.dt = dt;
    for(int p = 0; p < kStep / rpp; ++p)
    {
        V d = rocke_b_div(b, tid, ci(b, lpr));
        st.rows.push_back(rocke_b_add(b, d, ci(b, p * rpp)));
    }
    V md = rocke_b_mod(b, tid, ci(b, lpr));
    st.col = rocke_b_mul(b, md, ci(b, 8));
    for(int i = 0; i < nsrc; ++i)
    {
        int shape[3] = {2, kStep, head_size + pad};
        char hint[16];
        snprintf(hint, sizeof(hint), "bwst%d", i);
        st.lds.push_back(rocke_b_smem_alloc(b, dt, shape, 3, hint));
    }
    return true;
}

/* load(base, ...): per source, per row: global_load_vN(src, fn(base + r) + col). */
template <typename RowFn>
std::vector<Vec> stager_load(Stager& st, V base, const std::vector<std::pair<V, RowFn>>& srcs)
{
    rocke_ir_builder_t* b = st.b;
    std::vector<Vec> out;
    for(const auto& sf : srcs)
    {
        Vec vs;
        for(V r : st.rows)
        {
            V tok = rocke_b_add(b, base, r);
            V rb = sf.second(tok);
            V idx = rocke_b_add(b, rb, st.col);
            vs.push_back(rocke_b_global_load_vN(b, sf.first, idx, st.dt, 8, 16));
        }
        out.push_back(vs);
    }
    return out;
}

void stager_store(Stager& st, const std::vector<Vec>& vals, V buf)
{
    for(size_t i = 0; i < st.lds.size(); ++i)
    {
        for(size_t j = 0; j < st.rows.size(); ++j)
        {
            V idx[3] = {buf, st.rows[j], st.col};
            rocke_b_smem_store_vN(st.b, st.lds[i], idx, 3, vals[i][j], 8);
        }
    }
}

/* _mfma_bwd_tr8_at */
V tr8_at(rocke_ir_builder_t* b,
         V lds,
         const Vec& prefix,
         V row0,
         V col0,
         V g,
         V m16,
         const rocke_type_t* dt,
         bool nat)
{
    V md = rocke_b_mod(b, m16, ci(b, 4));
    V mm = rocke_b_mul(b, md, ci(b, 4));
    V col = rocke_b_add(b, col0, mm);
    int ts;
    V base = slot_rows(b, g, nat, &ts);
    V dv = rocke_b_div(b, m16, ci(b, 4));
    V a = rocke_b_add(b, base, dv);
    V r0 = rocke_b_add(b, row0, a);
    Vec idx(prefix);
    idx.push_back(r0);
    idx.push_back(col);
    V lo = rocke_b_ds_read_tr16_b64(b, lds, idx.data(), (int)idx.size(), dt);
    V r1 = rocke_b_add(b, r0, ci(b, ts));
    idx[prefix.size()] = r1;
    V hi = rocke_b_ds_read_tr16_b64(b, lds, idx.data(), (int)idx.size(), dt);
    return rocke_b_vec_concat(b, lo, hi);
}

/* _mfma_bwd_tr8 */
V tr8(rocke_ir_builder_t* b, V lds, V buf, V g, V m16, int nd, const rocke_type_t* dt, bool nat)
{
    V c0 = ci(b, 0);
    V c1 = ci(b, 16 * nd);
    return tr8_at(b, lds, Vec{buf}, c0, c1, g, m16, dt, nat);
}

/* _mfma_bwd_pack */
V pack(rocke_ir_builder_t* b, const Vec& vals, const rocke_type_t* dt)
{
    V v = rocke_b_zero_vec(b, dt, (int)vals.size());
    for(size_t j = 0; j < vals.size(); ++j)
    {
        V c = rocke_b_cast_f32_to(b, vals[j], dt);
        v = rocke_b_vec_insert(b, v, c, (int)j);
    }
    return v;
}

/* _mfma_bwd_store4 */
void store4(rocke_ir_builder_t* b, V ptr, V idx, V acc, const char* gdt, V scale)
{
    const bool f32 = str_eq(gdt, "f32");
    if(scale == NULL && f32)
    {
        rocke_b_global_store_vN(b, ptr, idx, acc, 4, 16);
        return;
    }
    Vec vals;
    for(int r = 0; r < 4; ++r)
    {
        vals.push_back(rocke_b_vec_extract(b, acc, r));
    }
    if(scale != NULL)
    {
        for(V& x : vals)
        {
            x = rocke_b_fmul(b, x, scale);
        }
    }
    if(f32)
    {
        V out = rocke_b_zero_vec_f32(b, 4);
        for(int r = 0; r < 4; ++r)
        {
            out = rocke_b_vec_insert(b, out, vals[r], r);
        }
        rocke_b_global_store_vN(b, ptr, idx, out, 4, 16);
    }
    else
    {
        rocke_b_global_store_vN(b, ptr, idx, pack(b, vals, rocke_io_ir_type(gdt)), 4, 8);
    }
}

V exp2_p(rocke_ir_builder_t* b, const rocke_fmha_bwd_spec_t* s, V x)
{
    return s->mfma_exp2_fast ? rocke_b_exp2_fast(b, x) : rocke_b_exp2(b, x);
}

/* _mfma_bwd_ctas: [(tile, head), ...] */
std::vector<std::pair<V, V>>
    ctas(rocke_ir_builder_t* b, const rocke_fmha_bwd_spec_t* s, int n_blocks, int n_heads, bool heavy_low)
{
    V bx = rocke_b_block_id_x(b);
    V by = rocke_b_block_id_y(b);
    const char* order = is_causal(s) ? s->mfma_causal_order : "none";
    if(str_eq(order, "lpt"))
    {
        V m = rocke_b_mul(b, by, ci(b, n_blocks));
        V lin = rocke_b_add(b, bx, m);
        V rank = rocke_b_div(b, lin, ci(b, n_heads));
        V tile = heavy_low ? rank : rocke_b_sub(b, ci(b, n_blocks - 1), rank);
        return {{tile, rocke_b_mod(b, lin, ci(b, n_heads))}};
    }
    if(str_eq(order, "pair"))
    {
        V t2 = rocke_b_sub(b, ci(b, n_blocks - 1), bx);
        return {{bx, by}, {t2, by}};
    }
    return {{bx, by}};
}

/* _mfma_bwd_setup: MFMA op id + io type; false (sticky error) when absent. */
bool setup(rocke_fmha_kernel_builder_t* kb, const char* arch, Ctx& c)
{
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    rocke_ir_builder_t* b = rocke_fmha_kernel_builder_builder(kb);
    if(target == NULL || !target->memory.has_ds_read_tr)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "fmha_bwd mfma mode needs ds_read_*_tr_b16 (absent on %s)", arch);
        return false;
    }
    c.op = str_eq(c.spec->common.dtype, "bf16") ? "mfma_f32_16x16x32_bf16" : "mfma_f32_16x16x32_f16";
    if(rocke_mma_catalog_by_op_id(&target->mma, c.op) == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "mma op %s absent on %s", c.op, arch);
        return false;
    }
    c.dt = rocke_io_ir_type(c.spec->common.dtype);
    return c.dt != NULL;
}

/* _mfma_bwd_kb */
bool init_kb(rocke_fmha_kernel_builder_t* kb, const rocke_fmha_bwd_spec_t* s, const char* which, int nw)
{
    char kname[1024];
    if(rocke_fmha_bwd_kernel_name(s, kname, sizeof(kname)) != ROCKE_OK)
    {
        return false;
    }
    std::string name(kname);
    char t[128];
    if(str_eq(which, "dkdv"))
    {
        snprintf(t, sizeof(t), "_mfma_dkdv_w%dk%dp%d", s->mfma_dkdv_waves, s->mfma_dkdv_ktiles, s->mfma_lds_pad);
    }
    else
    {
        snprintf(t, sizeof(t), "_mfma_dq_w%dq%dp%d", s->mfma_dq_waves, s->mfma_dq_qtiles, s->mfma_lds_pad);
    }
    name += t;
    if(s->mfma_waves_per_eu)
    {
        name += "_wpe" + std::to_string(s->mfma_waves_per_eu);
    }
    if(s->mfma_exp2_fast)
    {
        name += "_fx";
    }
    if(s->mfma_natural_rows)
    {
        name += "_nat";
    }
    if(is_causal(s))
    {
        if(!str_eq(s->mfma_causal_order, "none"))
        {
            name += std::string("_") + s->mfma_causal_order;
        }
        if(s->mfma_causal_split)
        {
            name += "_cs";
        }
    }
    if(s->mfma_fused_dq && str_eq(which, "dkdv"))
    {
        name += "_fdq";
    }
    if(!str_eq(s->output_grad_dtype, "f32"))
    {
        name += std::string("_g") + s->output_grad_dtype;
    }
    if(rocke_fmha_kernel_builder_init(kb, name.c_str(), &s->common) != ROCKE_OK)
    {
        return false;
    }
    rocke_fmha_kernel_builder_block_size(kb, 64 * nw);
    if(s->mfma_waves_per_eu)
    {
        rocke_ir_builder_t* b = rocke_fmha_kernel_builder_builder(kb);
        rocke_kernel_def_t* k = rocke_fmha_kernel_builder_kernel(kb);
        rocke_attr_set_int(b, &k->attrs, "waves_per_eu", (int64_t)s->mfma_waves_per_eu);
    }
    return true;
}

/* --------------------------------------------------------------------- *
 *  _build_mfma_pre
 * --------------------------------------------------------------------- */

rocke_kernel_def_t* build_pre(rocke_fmha_kernel_builder_t* kb, const rocke_fmha_bwd_spec_t* s, const char* arch)
{
    char kname[1024];
    const bool fused = s->mfma_fused_dq;
    Ctx c{kb, NULL, s, NULL, NULL};
    if(rocke_fmha_bwd_kernel_name(s, kname, sizeof(kname)) != ROCKE_OK)
    {
        return NULL;
    }
    std::string name = std::string(kname) + (fused ? "_mfma_pre_zdq" : "_mfma_pre");
    if(rocke_fmha_kernel_builder_init(kb, name.c_str(), &s->common) != ROCKE_OK)
    {
        return NULL;
    }
    if(!setup(kb, arch, c))
    {
        return NULL;
    }
    const int HQ = s->common.shape.num_query_heads;
    const int lpr = s->common.shape.head_size / 8;
    rocke_fmha_kernel_builder_block_size(kb, kPreThreads);
    declare_pre(kb, fused);
    rocke_ir_builder_t* b = rocke_fmha_kernel_builder_builder(kb);
    const int rows = kPreThreads / lpr;
    V bxv = rocke_b_block_id_x(b);
    V q_base = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, bxv, ci(b, rows)));
    V h = rocke_b_to_sgpr_u32(b, rocke_b_block_id_y(b));
    V tid = rocke_b_thread_id_x(b);
    V q = rocke_b_add(b, q_base, rocke_b_div(b, tid, ci(b, lpr)));
    V sub = rocke_b_mod(b, tid, ci(b, lpr));
    V col = rocke_b_mul(b, sub, ci(b, 8));
    V orb = rocke_fmha_kernel_builder_row_base(kb, "o", q, h);
    V o8 = rocke_b_global_load_vN(
        b, rocke_fmha_kernel_builder_tensor(kb, "O"), rocke_b_add(b, orb, col), c.dt, 8, 16);
    V drb = rocke_fmha_kernel_builder_row_base(kb, "do", q, h);
    V do8 = rocke_b_global_load_vN(
        b, rocke_fmha_kernel_builder_tensor(kb, "dO"), rocke_b_add(b, drb, col), c.dt, 8, 16);
    if(fused)
    {
        V m1 = rocke_b_mul(b, q, rocke_fmha_kernel_builder_scalar(kb, "stride_dq_token"));
        V m2 = rocke_b_mul(b, h, rocke_fmha_kernel_builder_stride_head(kb, "q"));
        V dq_row = rocke_b_add(b, m1, m2);
        V idx = rocke_b_add(b, dq_row, col);
        V z = rocke_b_zero_vec_f32(b, 8);
        rocke_b_global_store_vN(b, rocke_fmha_kernel_builder_ptr(kb, "dQ"), idx, z, 8, 16);
    }
    V acc = rocke_b_const_f32(b, 0.0);
    for(int j = 0; j < 8; ++j)
    {
        V x = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, o8, j));
        V y = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, do8, j));
        acc = rocke_b_fma(b, x, y, acc);
    }
    for(int m = 1; m < lpr; m *= 2)
    {
        acc = rocke_b_fadd(b, acc, rocke_b_warp_shuffle_xor(b, acc, m));
    }
    rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_eq(b, sub, ci(b, 0)));
    rocke_b_region_enter(b, iff.then_region);
    {
        V hm = rocke_b_mul(b, h, rocke_fmha_kernel_builder_scalar(kb, "seqlen_q"));
        V ws = rocke_b_add(b, hm, q);
        V qm = rocke_b_mul(b, q, ci(b, HQ));
        V ml = rocke_b_add(b, qm, h);
        V mv = rocke_b_global_load_f32(b, rocke_fmha_kernel_builder_ptr(kb, "M_saved"), ml, 4);
        V lv = rocke_b_log2(b, rocke_b_global_load_f32(b, rocke_fmha_kernel_builder_ptr(kb, "L_saved"), ml, 4));
        V lse2 = rocke_b_fadd(b, mv, lv);
        rocke_b_global_store(b, rocke_fmha_kernel_builder_ptr(kb, "Dsum"), ws, acc, 4);
        rocke_b_global_store(b, rocke_fmha_kernel_builder_ptr(kb, "Lse2"), ws, lse2, 4);
    }
    rocke_b_region_leave(b);
    rocke_b_ret(b);
    return rocke_fmha_kernel_builder_kernel(kb);
}

/* Loop helper: scf_for_iter over (lo, hi) with named iter args. */
rocke_for_t for_iter(rocke_ir_builder_t* b, V lo, V hi, const std::string& pre, const char* acc, const Vec& init, const char* iv)
{
    V step = ci(b, 1);
    std::vector<std::string> names;
    for(size_t j = 0; j < init.size(); ++j)
    {
        names.push_back(pre + acc + std::to_string(j));
    }
    std::vector<rocke_iter_arg_t> args(init.size());
    for(size_t j = 0; j < init.size(); ++j)
    {
        args[j].name = names[j].c_str();
        args[j].init = init[j];
    }
    std::string ivn = pre + iv;
    return rocke_b_scf_for_iter(b, lo, hi, step, args.data(), (int)args.size(), ivn.c_str(), false, false);
}

Vec loop_results(const rocke_for_t& f)
{
    Vec r;
    for(int j = 0; f.op != NULL && j < f.op->num_results; ++j)
    {
        r.push_back(f.op->results[j]);
    }
    return r;
}

/* --------------------------------------------------------------------- *
 *  _build_mfma_dkdv
 * --------------------------------------------------------------------- */

rocke_kernel_def_t* build_dkdv(rocke_fmha_kernel_builder_t* kb, const rocke_fmha_bwd_spec_t* s, const char* arch)
{
    const bool fused = s->mfma_fused_dq;
    const int D = s->common.shape.head_size;
    const int G = s->common.shape.num_query_heads / s->common.shape.num_kv_heads;
    const int nw = s->mfma_dkdv_waves, kpw = s->mfma_dkdv_ktiles;
    const bool causal = is_causal(s);
    const bool split = causal && s->mfma_causal_split;
    const bool nat = s->mfma_natural_rows;
    const int BM = kStep;
    const int BN = 16 * kpw * nw;
    const int n_a = D / 32, n_d = D / 16;
    const char* gdt = s->output_grad_dtype;
    Ctx c{kb, NULL, s, NULL, NULL};
    if(!init_kb(kb, s, "dkdv", nw))
    {
        return NULL;
    }
    if(!setup(kb, arch, c))
    {
        return NULL;
    }
    declare_dkdv(kb, fused, gdt);
    rocke_ir_builder_t* b = rocke_fmha_kernel_builder_builder(kb);
    V Q = rocke_fmha_kernel_builder_tensor(kb, "Q");
    V K = rocke_fmha_kernel_builder_tensor(kb, "K");
    V Vt = rocke_fmha_kernel_builder_tensor(kb, "V");
    V dO = rocke_fmha_kernel_builder_tensor(kb, "dO");
    V Lse2 = rocke_fmha_kernel_builder_ptr(kb, "Lse2");
    V Dsum = rocke_fmha_kernel_builder_ptr(kb, "Dsum");
    V scale_log2 = rocke_fmha_kernel_builder_scalar(kb, "scale_log2");
    V seqlen_q = rocke_fmha_kernel_builder_scalar(kb, "seqlen_q");
    V scale_inv = rocke_fmha_kernel_builder_scalar(kb, "scale_inv");
    Lane L = mk_lane(b, nw);
    V m16 = L.m16, g = L.g, wave = L.wave;
    int t_stride;
    V row_base = slot_rows(b, g, nat, &t_stride);
    Stager stg;
    if(!stager_init(stg, b, L.tid, nw, D, c.dt, s->mfma_lds_pad, 2))
    {
        return NULL;
    }
    V q_lds = stg.lds[0], do_lds = stg.lds[1];
    V k_lds = NULL, ds_lds = NULL, st_row = NULL, st_col = NULL, d_w = NULL;
    int rpp = 0, ndw = 0;
    if(fused)
    {
        const int lpr = D / 8;
        rpp = 64 * nw / lpr;
        int ks[2] = {BN, D + s->mfma_lds_pad};
        k_lds = rocke_b_smem_alloc(b, c.dt, ks, 2, "bwk");
        int dss[3] = {2, BN, BM + 16};
        ds_lds = rocke_b_smem_alloc(b, c.dt, dss, 3, "bwds");
        st_row = rocke_b_div(b, L.tid, ci(b, lpr));
        V md = rocke_b_mod(b, L.tid, ci(b, lpr));
        st_col = rocke_b_mul(b, md, ci(b, 8));
        d_w = rocke_b_mul(b, wave, ci(b, D / nw));
        ndw = D / (16 * nw);
    }
    const int ndg = (BN - 1 + BM - 1) / BM; /* -(-(BN - 1) // BM) */

    auto pass = [&](int p, V kblk, V kvh_in) {
        V n_cta = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, kblk, ci(b, BN)));
        V kvh = rocke_b_to_sgpr_u32(b, kvh_in);
        V wm = rocke_b_mul(b, wave, ci(b, 16 * kpw));
        V n_wave = rocke_b_add(b, n_cta, wm);

        auto kv_b = [&](V src, const char* name, int i) {
            V c16 = ci(b, 16 * i);
            V a1 = rocke_b_add(b, c16, m16);
            V tok = rocke_b_add(b, n_wave, a1);
            V row = rocke_fmha_kernel_builder_row_base(kb, name, tok, kvh);
            Vec out;
            for(int a = 0; a < n_a; ++a)
            {
                V c32 = ci(b, 32 * a);
                V gm = rocke_b_mul(b, g, ci(b, 8));
                V off = rocke_b_add(b, c32, gm);
                out.push_back(rocke_b_global_load_vN(b, src, rocke_b_add(b, row, off), c.dt, 8, 16));
            }
            return out;
        };
        std::vector<Vec> k_b, v_b;
        if(!fused)
        {
            for(int i = 0; i < kpw; ++i)
            {
                k_b.push_back(kv_b(K, "k", i));
            }
        }
        for(int i = 0; i < kpw; ++i)
        {
            v_b.push_back(kv_b(Vt, "v", i));
        }
        auto k_op = [&](int i, int a) -> V {
            if(!fused)
            {
                return k_b[i][a];
            }
            V wm2 = rocke_b_mul(b, wave, ci(b, 16 * kpw));
            V c16 = ci(b, 16 * i);
            V a1 = rocke_b_add(b, c16, m16);
            V row = rocke_b_add(b, wm2, a1);
            V c32 = ci(b, 32 * a);
            V gm = rocke_b_mul(b, g, ci(b, 8));
            V col = rocke_b_add(b, c32, gm);
            V idx[2] = {row, col};
            return rocke_b_smem_load_vN(b, k_lds, idx, 2, c.dt, 8);
        };

        V nqt = rocke_b_div(b, seqlen_q, ci(b, BM));
        V qt_lo = causal ? rocke_b_div(b, n_cta, ci(b, BM)) : ci(b, 0);
        V nq_eff = rocke_b_sub(b, nqt, qt_lo);
        V total = rocke_b_mul(b, nq_eff, ci(b, G));
        const int n_diag = split ? G * ndg : 0;
        V last = rocke_b_sub(b, total, ci(b, 1));

        auto coords = [&](V it) -> std::pair<V, V> {
            if(G == 1)
            {
                V a = rocke_b_add(b, qt_lo, it);
                return {kvh, rocke_b_mul(b, a, ci(b, BM))};
            }
            V hg, qt;
            if(split)
            {
                V in_diag = rocke_b_cmp_lt(b, it, ci(b, n_diag));
                V jj = rocke_b_select(b, in_diag, it, rocke_b_sub(b, it, ci(b, n_diag)));
                V c1 = ci(b, ndg);
                V c2 = ci(b, ndg);
                V sb = rocke_b_sub(b, nq_eff, c2);
                V per = rocke_b_select(b, in_diag, c1, sb);
                V z = ci(b, 0);
                V cn = ci(b, ndg);
                V off = rocke_b_select(b, in_diag, z, cn);
                hg = rocke_b_div(b, jj, per);
                V a = rocke_b_add(b, qt_lo, off);
                V md = rocke_b_mod(b, jj, per);
                qt = rocke_b_add(b, a, md);
            }
            else
            {
                hg = rocke_b_div(b, it, nq_eff);
                qt = rocke_b_add(b, qt_lo, rocke_b_mod(b, it, nq_eff));
            }
            V km = rocke_b_mul(b, kvh, ci(b, G));
            V hh = rocke_b_add(b, km, hg);
            return {hh, rocke_b_mul(b, qt, ci(b, BM))};
        };

        auto stats = [&](V it) {
            auto hq = coords(it);
            V hm = rocke_b_mul(b, hq.first, seqlen_q);
            V a = rocke_b_add(b, hm, hq.second);
            V ws = rocke_b_add(b, a, row_base);
            Vec out;
            for(V ptr : {Lse2, Dsum})
            {
                for(int t = 0; t < 2; ++t)
                {
                    V idx = rocke_b_add(b, ws, ci(b, t_stride * t));
                    out.push_back(rocke_b_global_load_vN(b, ptr, idx, rocke_f32(), 4, 16));
                }
            }
            return out;
        };

        using RowFn = std::function<V(V)>;
        auto load = [&](V it) {
            V cmp = rocke_b_cmp_lt(b, it, total);
            V sel = rocke_b_select(b, cmp, it, last);
            auto hq = coords(sel);
            V h = hq.first;
            std::vector<std::pair<V, RowFn>> srcs = {
                {Q, [&, h](V r) { return rocke_fmha_kernel_builder_row_base(kb, "q", r, h); }},
                {dO, [&, h](V r) { return rocke_fmha_kernel_builder_row_base(kb, "do", r, h); }},
            };
            return stager_load(stg, hq.second, srcs);
        };

        auto sdp = [&](V it, V buf, const Vec& st, bool masked) {
            auto hq = coords(it);
            V q0 = hq.second;
            std::vector<std::vector<Vec>> ops; /* [lds][t][a] */
            for(V lds : {q_lds, do_lds})
            {
                std::vector<Vec> per_t;
                for(int t = 0; t < 2; ++t)
                {
                    Vec per_a;
                    for(int a = 0; a < n_a; ++a)
                    {
                        V pr = perm_row(b, m16, t, nat);
                        V c32 = ci(b, 32 * a);
                        V gm = rocke_b_mul(b, g, ci(b, 8));
                        V col = rocke_b_add(b, c32, gm);
                        V idx[3] = {buf, pr, col};
                        per_a.push_back(rocke_b_smem_load_vN(b, lds, idx, 3, c.dt, 8));
                    }
                    per_t.push_back(per_a);
                }
                ops.push_back(per_t);
            }
            V q_lane = rocke_b_add(b, q0, row_base);
            Vec out;
            for(int i = 0; i < kpw; ++i)
            {
                V c16 = ci(b, 16 * i);
                V a1 = rocke_b_add(b, c16, m16);
                V key = rocke_b_add(b, n_wave, a1);
                Vec ps, dss;
                for(int t = 0; t < 2; ++t)
                {
                    V s_acc = rocke_b_zero_vec_f32(b, 4);
                    V dp_acc = rocke_b_zero_vec_f32(b, 4);
                    for(int a = 0; a < n_a; ++a)
                    {
                        V kb_op = k_op(i, a);
                        s_acc = rocke_b_mma(b, c.op, ops[0][t][a], kb_op, s_acc, NULL, 0);
                    }
                    for(int a = 0; a < n_a; ++a)
                    {
                        dp_acc = rocke_b_mma(b, c.op, ops[1][t][a], v_b[i][a], dp_acc, NULL, 0);
                    }
                    for(int r = 0; r < 4; ++r)
                    {
                        V e = rocke_b_vec_extract(b, s_acc, r);
                        V m = rocke_b_fmul(b, e, scale_log2);
                        V l = rocke_b_vec_extract(b, st[t], r);
                        V p_ = exp2_p(b, s, rocke_b_fsub(b, m, l));
                        if(masked)
                        {
                            V qpos = rocke_b_add(b, q_lane, ci(b, t_stride * t + r));
                            V cmp = rocke_b_cmp_le(b, key, qpos);
                            p_ = rocke_b_select(b, cmp, p_, rocke_b_const_f32(b, 0.0));
                        }
                        ps.push_back(p_);
                        V e1 = rocke_b_vec_extract(b, dp_acc, r);
                        V e2 = rocke_b_vec_extract(b, st[2 + t], r);
                        dss.push_back(rocke_b_fmul(b, p_, rocke_b_fsub(b, e1, e2)));
                    }
                }
                out.push_back(pack(b, ps, c.dt));
                out.push_back(pack(b, dss, c.dt));
            }
            return out;
        };

        auto grads = [&](V buf, const Vec& pds, std::vector<Vec>& dk, std::vector<Vec>& dv) {
            for(int n = 0; n < n_d; ++n)
            {
                V do_t = tr8(b, do_lds, buf, g, m16, n, c.dt, nat);
                for(int i = 0; i < kpw; ++i)
                {
                    dv[i][n] = rocke_b_mma(b, c.op, do_t, pds[2 * i], dv[i][n], NULL, 0);
                }
                V q_t = tr8(b, q_lds, buf, g, m16, n, c.dt, nat);
                for(int i = 0; i < kpw; ++i)
                {
                    dk[i][n] = rocke_b_mma(b, c.op, q_t, pds[2 * i + 1], dk[i][n], NULL, 0);
                }
            }
        };

        auto ds_write = [&](V slot, const Vec& pds) {
            for(int i = 0; i < kpw; ++i)
            {
                V wm2 = rocke_b_mul(b, wave, ci(b, 16 * kpw));
                V c16 = ci(b, 16 * i);
                V a1 = rocke_b_add(b, c16, m16);
                V n_local = rocke_b_add(b, wm2, a1);
                for(int t = 0; t < 2; ++t)
                {
                    V v4 = rocke_b_zero_vec(b, c.dt, 4);
                    for(int r = 0; r < 4; ++r)
                    {
                        V e = rocke_b_vec_extract(b, pds[2 * i + 1], 4 * t + r);
                        v4 = rocke_b_vec_insert(b, v4, e, r);
                    }
                    V col = rocke_b_add(b, row_base, ci(b, t_stride * t));
                    V idx[3] = {slot, n_local, col};
                    rocke_b_smem_store_vN(b, ds_lds, idx, 3, v4, 4);
                }
            }
        };

        auto dq = [&](V it, V slot) {
            auto hq = coords(it);
            V h = hq.first, q0 = hq.second;
            std::vector<Vec> acc(2);
            for(int qs = 0; qs < 2; ++qs)
            {
                for(int j = 0; j < ndw; ++j)
                {
                    acc[qs].push_back(rocke_b_zero_vec_f32(b, 4));
                }
            }
            for(int ks = 0; ks < BN / 32; ++ks)
            {
                Vec a_ops, b_ops;
                for(int qs = 0; qs < 2; ++qs)
                {
                    V r0 = ci(b, 32 * ks);
                    V c0 = ci(b, 16 * qs);
                    a_ops.push_back(tr8_at(b, ds_lds, Vec{slot}, r0, c0, g, m16, c.dt, nat));
                }
                for(int j = 0; j < ndw; ++j)
                {
                    V r0 = ci(b, 32 * ks);
                    V c0 = rocke_b_add(b, d_w, ci(b, 16 * j));
                    b_ops.push_back(tr8_at(b, k_lds, Vec{}, r0, c0, g, m16, c.dt, nat));
                }
                for(int qs = 0; qs < 2; ++qs)
                {
                    for(int j = 0; j < ndw; ++j)
                    {
                        acc[qs][j] = rocke_b_mma(b, c.op, a_ops[qs], b_ops[j], acc[qs][j], NULL, 0);
                    }
                }
            }
            V head = rocke_b_mul(b, h, rocke_fmha_kernel_builder_stride_head(kb, "q"));
            for(int qs = 0; qs < 2; ++qs)
            {
                for(int j = 0; j < ndw; ++j)
                {
                    V a = rocke_b_add(b, d_w, ci(b, 16 * j));
                    V col = rocke_b_add(b, a, m16);
                    for(int r = 0; r < 4; ++r)
                    {
                        V gm = rocke_b_mul(b, g, ci(b, 4));
                        V a2 = rocke_b_add(b, gm, ci(b, 16 * qs + r));
                        V q = rocke_b_add(b, q0, a2);
                        V qm = rocke_b_mul(b, q, rocke_fmha_kernel_builder_scalar(kb, "stride_dq_token"));
                        V ih = rocke_b_add(b, qm, head);
                        V idx = rocke_b_add(b, ih, col);
                        V val = rocke_b_fmul(b, rocke_b_vec_extract(b, acc[qs][j], r), scale_inv);
                        rocke_b_global_atomic_add_f32(b, rocke_fmha_kernel_builder_ptr(kb, "dQ"), idx, val);
                    }
                }
            }
        };

        {
            V c0 = ci(b, 0);
            auto first = load(c0);
            stager_store(stg, first, ci(b, 0));
        }
        if(fused)
        {
            for(int rp = 0; rp < BN / rpp; ++rp)
            {
                V row = rocke_b_add(b, st_row, ci(b, rp * rpp));
                V tok = rocke_b_add(b, n_cta, row);
                V rb = rocke_fmha_kernel_builder_row_base(kb, "k", tok, kvh);
                V kv = rocke_b_global_load_vN(b, K, rocke_b_add(b, rb, st_col), c.dt, 8, 16);
                V idx[2] = {row, st_col};
                rocke_b_smem_store_vN(b, k_lds, idx, 2, kv, 8);
            }
        }
        rocke_b_sync(b);
        Vec state;
        for(int i = 0; i < kpw; ++i)
        {
            for(int j = 0; j < 2 * n_d; ++j)
            {
                state.push_back(rocke_b_zero_vec_f32(b, 4));
            }
        }
        struct Seg
        {
            V lo, hi;
            bool masked;
        };
        std::vector<Seg> segs;
        if(split)
        {
            V a = ci(b, 0);
            V bb = ci(b, n_diag);
            V cc = ci(b, n_diag);
            segs = {{a, bb, true}, {cc, total, false}};
        }
        else
        {
            segs = {{ci(b, 0), total, causal}};
        }
        for(size_t si = 0; si < segs.size(); ++si)
        {
            std::string pre = "p" + std::to_string(p) + "s" + std::to_string(si) + "_";
            rocke_for_t loop = for_iter(b, segs[si].lo, segs[si].hi, pre, "acc", state, "it");
            rocke_b_region_enter(b, loop.body);
            {
                V it = loop.iv;
                std::vector<Vec> dk(kpw), dv(kpw);
                for(int i = 0; i < kpw; ++i)
                {
                    for(int n = 0; n < n_d; ++n)
                    {
                        dk[i].push_back(loop.iter_vars[2 * n_d * i + n]);
                    }
                    for(int n = 0; n < n_d; ++n)
                    {
                        dv[i].push_back(loop.iter_vars[2 * n_d * i + n_d + n]);
                    }
                }
                V buf = rocke_b_mod(b, it, ci(b, 2));
                auto nxt = load(rocke_b_add(b, it, ci(b, 1)));
                Vec st = stats(it);
                Vec pds = sdp(it, buf, st, segs[si].masked);
                if(fused)
                {
                    ds_write(buf, pds);
                    rocke_if_t iff = rocke_b_scf_if(b, rocke_b_cmp_gt(b, it, ci(b, 0)));
                    rocke_b_region_enter(b, iff.then_region);
                    {
                        V a = rocke_b_sub(b, it, ci(b, 1));
                        V c1 = ci(b, 1);
                        V sl = rocke_b_sub(b, c1, buf);
                        dq(a, sl);
                    }
                    rocke_b_region_leave(b);
                }
                grads(buf, pds, dk, dv);
                {
                    V c1 = ci(b, 1);
                    stager_store(stg, nxt, rocke_b_sub(b, c1, buf));
                }
                rocke_b_sync(b);
                Vec y;
                for(int i = 0; i < kpw; ++i)
                {
                    y.insert(y.end(), dk[i].begin(), dk[i].end());
                    y.insert(y.end(), dv[i].begin(), dv[i].end());
                }
                rocke_b_scf_yield(b, y.data(), (int)y.size());
            }
            rocke_b_region_leave(b);
            state = loop_results(loop);
        }
        if(fused)
        {
            dq(last, rocke_b_mod(b, last, ci(b, 2)));
        }
        for(int i = 0; i < kpw; ++i)
        {
            V c16 = ci(b, 16 * i);
            V a1 = rocke_b_add(b, c16, m16);
            V key = rocke_b_add(b, n_wave, a1);
            V k1 = rocke_b_mul(b, key, rocke_fmha_kernel_builder_scalar(kb, "stride_dk_token"));
            V k2 = rocke_b_mul(b, kvh, rocke_fmha_kernel_builder_stride_head(kb, "k"));
            V dk_row = rocke_b_add(b, k1, k2);
            V v1 = rocke_b_mul(b, key, rocke_fmha_kernel_builder_scalar(kb, "stride_dv_token"));
            V v2 = rocke_b_mul(b, kvh, rocke_fmha_kernel_builder_stride_head(kb, "v"));
            V dv_row = rocke_b_add(b, v1, v2);
            for(int n = 0; n < n_d; ++n)
            {
                V c16n = ci(b, 16 * n);
                V gm = rocke_b_mul(b, g, ci(b, 4));
                V d = rocke_b_add(b, c16n, gm);
                store4(b, rocke_fmha_kernel_builder_ptr(kb, "dK"), rocke_b_add(b, dk_row, d),
                       state[2 * n_d * i + n], gdt, scale_inv);
                store4(b, rocke_fmha_kernel_builder_ptr(kb, "dV"), rocke_b_add(b, dv_row, d),
                       state[2 * n_d * i + n_d + n], gdt, NULL);
            }
        }
    };

    auto cs = ctas(b, s, s->seqlen_k / BN, s->common.shape.num_kv_heads, true);
    for(size_t p = 0; p < cs.size(); ++p)
    {
        if(p)
        {
            rocke_b_sync(b);
        }
        pass((int)p, cs[p].first, cs[p].second);
    }
    rocke_b_ret(b);
    return rocke_fmha_kernel_builder_kernel(kb);
}

/* --------------------------------------------------------------------- *
 *  _build_mfma_dq
 * --------------------------------------------------------------------- */

rocke_kernel_def_t* build_dq(rocke_fmha_kernel_builder_t* kb, const rocke_fmha_bwd_spec_t* s, const char* arch)
{
    const int D = s->common.shape.head_size;
    const int G = s->common.shape.num_query_heads / s->common.shape.num_kv_heads;
    const int nw = s->mfma_dq_waves, qpw = s->mfma_dq_qtiles;
    const bool causal = is_causal(s);
    const bool split = causal && s->mfma_causal_split;
    const bool nat = s->mfma_natural_rows;
    const int BK = kStep;
    const int BQ = 16 * qpw * nw;
    const int n_a = D / 32, n_d = D / 16;
    const char* gdt = s->output_grad_dtype;
    Ctx c{kb, NULL, s, NULL, NULL};
    if(!init_kb(kb, s, "dq", nw))
    {
        return NULL;
    }
    if(!setup(kb, arch, c))
    {
        return NULL;
    }
    declare_dq(kb, gdt);
    rocke_ir_builder_t* b = rocke_fmha_kernel_builder_builder(kb);
    V Q = rocke_fmha_kernel_builder_tensor(kb, "Q");
    V K = rocke_fmha_kernel_builder_tensor(kb, "K");
    V Vt = rocke_fmha_kernel_builder_tensor(kb, "V");
    V dO = rocke_fmha_kernel_builder_tensor(kb, "dO");
    V scale_log2 = rocke_fmha_kernel_builder_scalar(kb, "scale_log2");
    V seqlen_q = rocke_fmha_kernel_builder_scalar(kb, "seqlen_q");
    V scale_inv = rocke_fmha_kernel_builder_scalar(kb, "scale_inv");
    Lane L = mk_lane(b, nw);
    V m16 = L.m16, g = L.g, wave = L.wave;
    V d_lane = rocke_b_mul(b, g, ci(b, 8));
    int t_stride;
    V row_base = slot_rows(b, g, nat, &t_stride);
    V kvh_cur = NULL;
    Stager stg;
    if(!stager_init(stg, b, L.tid, nw, D, c.dt, s->mfma_lds_pad, 2))
    {
        return NULL;
    }
    V k_lds = stg.lds[0], v_lds = stg.lds[1];

    auto pass = [&](int p, V qblk, V h_in) {
        V q_cta = rocke_b_to_sgpr_u32(b, rocke_b_mul(b, qblk, ci(b, BQ)));
        V h = rocke_b_to_sgpr_u32(b, h_in);
        kvh_cur = G > 1 ? rocke_b_to_sgpr_u32(b, rocke_b_div(b, h, ci(b, G))) : h;
        V wm = rocke_b_mul(b, wave, ci(b, 16 * qpw));
        V q_wave = rocke_b_add(b, q_cta, wm);

        std::vector<Vec> q_b, do_b;
        Vec lse, dsum, q_pos;
        for(int i = 0; i < qpw; ++i)
        {
            V c16 = ci(b, 16 * i);
            V a1 = rocke_b_add(b, c16, m16);
            V q = rocke_b_add(b, q_wave, a1);
            q_pos.push_back(q);
            V qr = rocke_fmha_kernel_builder_row_base(kb, "q", q, h);
            V dr = rocke_fmha_kernel_builder_row_base(kb, "do", q, h);
            Vec qa, da;
            for(int a = 0; a < n_a; ++a)
            {
                V off = rocke_b_add(b, ci(b, 32 * a), d_lane);
                qa.push_back(rocke_b_global_load_vN(b, Q, rocke_b_add(b, qr, off), c.dt, 8, 16));
            }
            q_b.push_back(qa);
            for(int a = 0; a < n_a; ++a)
            {
                V off = rocke_b_add(b, ci(b, 32 * a), d_lane);
                da.push_back(rocke_b_global_load_vN(b, dO, rocke_b_add(b, dr, off), c.dt, 8, 16));
            }
            do_b.push_back(da);
            V hm = rocke_b_mul(b, h, seqlen_q);
            V ws = rocke_b_add(b, hm, q);
            lse.push_back(rocke_b_global_load_f32(b, rocke_fmha_kernel_builder_ptr(kb, "Lse2"), ws, 4));
            dsum.push_back(rocke_b_global_load_f32(b, rocke_fmha_kernel_builder_ptr(kb, "Dsum"), ws, 4));
        }

        V nkt = rocke_b_div(b, rocke_fmha_kernel_builder_scalar(kb, "seqlen_k"), ci(b, BK));
        if(causal)
        {
            V a = rocke_b_add(b, q_cta, ci(b, BQ - 1));
            V dv = rocke_b_div(b, a, ci(b, BK));
            V need = rocke_b_add(b, dv, ci(b, 1));
            V cmp = rocke_b_cmp_lt(b, need, nkt);
            nkt = rocke_b_select(b, cmp, need, nkt);
        }
        V last = rocke_b_sub(b, nkt, ci(b, 1));

        using RowFn = std::function<V(V)>;
        auto load = [&](V kt) {
            V cmp = rocke_b_cmp_lt(b, kt, nkt);
            V sel = rocke_b_select(b, cmp, kt, last);
            V base = rocke_b_mul(b, sel, ci(b, BK));
            V kvh = kvh_cur;
            std::vector<std::pair<V, RowFn>> srcs = {
                {K, [&, kvh](V r) { return rocke_fmha_kernel_builder_row_base(kb, "k", r, kvh); }},
                {Vt, [&, kvh](V r) { return rocke_fmha_kernel_builder_row_base(kb, "v", r, kvh); }},
            };
            return stager_load(stg, base, srcs);
        };

        auto sdp = [&](V kt, V buf, bool masked) {
            std::vector<std::vector<Vec>> ops;
            for(V lds : {k_lds, v_lds})
            {
                std::vector<Vec> per_t;
                for(int t = 0; t < 2; ++t)
                {
                    Vec per_a;
                    for(int a = 0; a < n_a; ++a)
                    {
                        V pr = perm_row(b, m16, t, nat);
                        V col = rocke_b_add(b, ci(b, 32 * a), d_lane);
                        V idx[3] = {buf, pr, col};
                        per_a.push_back(rocke_b_smem_load_vN(b, lds, idx, 3, c.dt, 8));
                    }
                    per_t.push_back(per_a);
                }
                ops.push_back(per_t);
            }
            V km = rocke_b_mul(b, kt, ci(b, BK));
            V k_lane = rocke_b_add(b, km, row_base);
            Vec ds_vec;
            for(int i = 0; i < qpw; ++i)
            {
                Vec dss;
                for(int t = 0; t < 2; ++t)
                {
                    V s_acc = rocke_b_zero_vec_f32(b, 4);
                    V dp_acc = rocke_b_zero_vec_f32(b, 4);
                    for(int a = 0; a < n_a; ++a)
                    {
                        s_acc = rocke_b_mma(b, c.op, ops[0][t][a], q_b[i][a], s_acc, NULL, 0);
                    }
                    for(int a = 0; a < n_a; ++a)
                    {
                        dp_acc = rocke_b_mma(b, c.op, ops[1][t][a], do_b[i][a], dp_acc, NULL, 0);
                    }
                    for(int r = 0; r < 4; ++r)
                    {
                        V e = rocke_b_vec_extract(b, s_acc, r);
                        V m = rocke_b_fmul(b, e, scale_log2);
                        V p_ = exp2_p(b, s, rocke_b_fsub(b, m, lse[i]));
                        if(masked)
                        {
                            V key = rocke_b_add(b, k_lane, ci(b, t_stride * t + r));
                            V cmp = rocke_b_cmp_le(b, key, q_pos[i]);
                            p_ = rocke_b_select(b, cmp, p_, rocke_b_const_f32(b, 0.0));
                        }
                        V e1 = rocke_b_vec_extract(b, dp_acc, r);
                        dss.push_back(rocke_b_fmul(b, p_, rocke_b_fsub(b, e1, dsum[i])));
                    }
                }
                ds_vec.push_back(pack(b, dss, c.dt));
            }
            return ds_vec;
        };

        auto grads = [&](V buf, const Vec& ds_vec, std::vector<Vec>& dq) {
            for(int n = 0; n < n_d; ++n)
            {
                V k_t = tr8(b, k_lds, buf, g, m16, n, c.dt, nat);
                for(int i = 0; i < qpw; ++i)
                {
                    dq[i][n] = rocke_b_mma(b, c.op, k_t, ds_vec[i], dq[i][n], NULL, 0);
                }
            }
        };

        {
            V c0 = ci(b, 0);
            auto first = load(c0);
            stager_store(stg, first, ci(b, 0));
        }
        rocke_b_sync(b);
        Vec state;
        for(int j = 0; j < qpw * n_d; ++j)
        {
            state.push_back(rocke_b_zero_vec_f32(b, 4));
        }
        struct Seg
        {
            V lo, hi;
            bool masked;
        };
        std::vector<Seg> segs;
        if(split)
        {
            V kd = rocke_b_div(b, q_cta, ci(b, BK));
            segs = {{ci(b, 0), kd, false}, {kd, nkt, true}};
        }
        else
        {
            segs = {{ci(b, 0), nkt, causal}};
        }
        for(size_t si = 0; si < segs.size(); ++si)
        {
            std::string pre = "p" + std::to_string(p) + "s" + std::to_string(si) + "_";
            rocke_for_t loop = for_iter(b, segs[si].lo, segs[si].hi, pre, "dq", state, "kt");
            rocke_b_region_enter(b, loop.body);
            {
                V kt = loop.iv;
                std::vector<Vec> dq(qpw);
                for(int i = 0; i < qpw; ++i)
                {
                    for(int n = 0; n < n_d; ++n)
                    {
                        dq[i].push_back(loop.iter_vars[n_d * i + n]);
                    }
                }
                V buf = rocke_b_mod(b, kt, ci(b, 2));
                auto nxt = load(rocke_b_add(b, kt, ci(b, 1)));
                Vec dsv = sdp(kt, buf, segs[si].masked);
                grads(buf, dsv, dq);
                {
                    V c1 = ci(b, 1);
                    stager_store(stg, nxt, rocke_b_sub(b, c1, buf));
                }
                rocke_b_sync(b);
                Vec y;
                for(int i = 0; i < qpw; ++i)
                {
                    y.insert(y.end(), dq[i].begin(), dq[i].end());
                }
                rocke_b_scf_yield(b, y.data(), (int)y.size());
            }
            rocke_b_region_leave(b);
            state = loop_results(loop);
        }

        for(int i = 0; i < qpw; ++i)
        {
            V r1 = rocke_b_mul(b, q_pos[i], rocke_fmha_kernel_builder_scalar(kb, "stride_dq_token"));
            V r2 = rocke_b_mul(b, h, rocke_fmha_kernel_builder_stride_head(kb, "q"));
            V row = rocke_b_add(b, r1, r2);
            for(int n = 0; n < n_d; ++n)
            {
                V c16n = ci(b, 16 * n);
                V gm = rocke_b_mul(b, g, ci(b, 4));
                V col = rocke_b_add(b, c16n, gm);
                store4(b, rocke_fmha_kernel_builder_ptr(kb, "dQ"), rocke_b_add(b, row, col),
                       state[n_d * i + n], gdt, scale_inv);
            }
        }
    };

    auto cs = ctas(b, s, s->seqlen_q / BQ, s->common.shape.num_query_heads, false);
    for(size_t p = 0; p < cs.size(); ++p)
    {
        if(p)
        {
            rocke_b_sync(b);
        }
        pass((int)p, cs[p].first, cs[p].second);
    }
    rocke_b_ret(b);
    return rocke_fmha_kernel_builder_kernel(kb);
}

/* --------------------------------------------------------------------- *
 *  launch plan
 * --------------------------------------------------------------------- */

struct Plan
{
    std::string err;
    std::vector<rocke_fmha_bwd_stage_info_t> stages;
};

Plan make_plan(const rocke_fmha_bwd_spec_t* s, const char* arch)
{
    Plan pl;
    if(str_eq(s->bwd_mode, "atomic"))
    {
        rocke_fmha_bwd_stage_info_t st = {"atomic", {0, 0, 0}, {64, 1, 1}};
        rocke_fmha_bwd_grid(s, &st.grid[0], &st.grid[1], &st.grid[2]);
        pl.stages.push_back(st);
        return pl;
    }
    if(!str_eq(s->bwd_mode, "mfma"))
    {
        pl.err = std::string("fmha_bwd bwd_mode: atomic | mfma (got '") + (s->bwd_mode ? s->bwd_mode : "")
                 + "')";
        return pl;
    }
    pl.err = mfma_plan_error(s, arch);
    if(!pl.err.empty())
    {
        return pl;
    }
    const int D = s->common.shape.head_size;
    const int HQ = s->common.shape.num_query_heads, HK = s->common.shape.num_kv_heads;
    const int bn = 16 * s->mfma_dkdv_waves * s->mfma_dkdv_ktiles;
    const int bq = 16 * s->mfma_dq_waves * s->mfma_dq_qtiles;
    const int rows_pre = kPreThreads / (D / 8);
    const int div = (is_causal(s) && str_eq(s->mfma_causal_order, "pair")) ? 2 : 1;
    pl.stages.push_back({"pre", {s->seqlen_q / rows_pre, HQ, 1}, {kPreThreads, 1, 1}});
    pl.stages.push_back({"dkdv", {s->seqlen_k / bn / div, HK, 1}, {64 * s->mfma_dkdv_waves, 1, 1}});
    if(!s->mfma_fused_dq)
    {
        pl.stages.push_back({"dq", {s->seqlen_q / bq / div, HQ, 1}, {64 * s->mfma_dq_waves, 1, 1}});
    }
    return pl;
}

} // namespace

extern "C" rocke_status_t rocke_fmha_bwd_launch_plan(const rocke_fmha_bwd_spec_t* spec,
                                                     const char* arch,
                                                     rocke_fmha_bwd_stage_info_t* out,
                                                     int* out_count,
                                                     char* err,
                                                     size_t err_cap)
{
    if(spec == NULL || out == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    Plan pl = make_plan(spec, arch ? arch : "gfx950");
    if(!pl.err.empty())
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "%s", pl.err.c_str());
        }
        return ROCKE_ERR_VALUE;
    }
    for(size_t i = 0; i < pl.stages.size(); ++i)
    {
        out[i] = pl.stages[i];
    }
    *out_count = (int)pl.stages.size();
    return ROCKE_OK;
}

extern "C" rocke_kernel_def_t* rocke_build_fmha_bwd_kernels_stage(rocke_fmha_kernel_builder_t* kb,
                                                                  const rocke_fmha_bwd_spec_t* spec,
                                                                  const char* arch,
                                                                  int stage)
{
    if(kb == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    Plan pl = make_plan(spec, arch);
    if(!pl.err.empty() || stage < 0 || stage >= (int)pl.stages.size())
    {
        fail(kb, spec, pl.err.empty() ? std::string("fmha_bwd launch plan: stage out of range") : pl.err);
        return NULL;
    }
    const char* which = pl.stages[stage].stage;
    if(str_eq(which, "atomic"))
    {
        return rocke_build_fmha_bwd(kb, spec, arch);
    }
    return ckc::guard_builder(rocke_fmha_kernel_builder_builder(kb), [&]() -> rocke_kernel_def_t* {
        if(str_eq(which, "pre"))
        {
            return build_pre(kb, spec, arch);
        }
        if(str_eq(which, "dkdv"))
        {
            return build_dkdv(kb, spec, arch);
        }
        return build_dq(kb, spec, arch);
    });
}

extern "C" rocke_status_t rocke_fmha_bwd_stage_signature(rocke_fmha_kernel_builder_t* kb,
                                                         const rocke_fmha_bwd_spec_t* spec,
                                                         const char* arch,
                                                         int stage,
                                                         rocke_arena_t* arena,
                                                         const rocke_sig_entry_t** out_items,
                                                         size_t* out_count)
{
    if(kb == NULL || spec == NULL || arena == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    Plan pl = make_plan(spec, arch ? arch : "gfx950");
    if(!pl.err.empty() || stage < 0 || stage >= (int)pl.stages.size())
    {
        return ROCKE_ERR_VALUE;
    }
    const char* which = pl.stages[stage].stage;
    if(str_eq(which, "atomic"))
    {
        return rocke_fmha_bwd_signature(kb, spec, arena, out_items, out_count);
    }
    std::string probe = std::string("rocke_fmha_bwd_mfma_") + which + "_sig_probe";
    rocke_status_t st = rocke_fmha_kernel_builder_init(kb, probe.c_str(), &spec->common);
    if(st != ROCKE_OK)
    {
        return st;
    }
    if(str_eq(which, "pre"))
    {
        declare_pre(kb, spec->mfma_fused_dq);
    }
    else if(str_eq(which, "dkdv"))
    {
        declare_dkdv(kb, spec->mfma_fused_dq, spec->output_grad_dtype);
    }
    else
    {
        declare_dq(kb, spec->output_grad_dtype);
    }
    return rocke_fmha_kernel_builder_signature(kb, arena, out_items, out_count);
}

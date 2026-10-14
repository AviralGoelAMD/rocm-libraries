// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx950_attention_tiled_3d_gfx950_attention_tiled_3d_segment_loop_epilogue.c
 *
 * Chunked C99 port of the segment-kernel online-softmax COMPUTE region of
 * rocke/instances/gfx950/attention_tiled_3d.py (gfx950 WIDE-K variant).
 *
 * SCOPE (this translation unit): the "segment loop + epilogue" slice of
 *   build_unified_attention_3d_tiled (lines 253-950):
 *
 *   rocke_gfx950_attention_tiled_3d_emit_loop_init        lines 536-562, 715-718
 *       - sinks-conditioned m_inits / l_inits / acc_inits / cur_buf_init carry
 *       - async DMA infra + paged-KV descriptor (via emit_async_infra)
 *       - first K load (_issue_k(tile_start, 0))
 *   rocke_gfx950_attention_tiled_3d_emit_softmax_loop     lines 720-898
 *       - scf.for over [tile_start, tile_end): buffer swap, s_waitcnt + sync,
 *         QK wide 16x16x32 MFMA, V/next-K prefetch, alibi/softcap/qq_bias/mask,
 *         online (m,l) update via warp_xor_reduce_max/sum, P_lds store, PV MFMA
 *         (16x16x32 via two ds_read_tr16_b64 + vec_concat when PV_K_STEP==32,
 *         else 16x16x16 via one ds_read_tr16_b64), carry yield
 *       - stashes m_final/l_final/acc_final into ctx
 *   rocke_gfx950_attention_tiled_3d_emit_epilogue         lines 900-948
 *       - guarded segm_output / segm_max / segm_expsum stores
 *
 * Everything is consumed through the internal header; peers (params/prologue/
 * descriptors/issuers/_mfma_16x16_c_row/emit_async_infra) are called via that
 * header. The builder call sequence mirrors the Python op-for-op so the emitted
 * IR is byte-identical.
 */

#include <stdio.h> /* snprintf */
#include <string.h>

#include "rocke/instance_gfx950_attention_tiled_3d_internal.h"

/* ============================================================ *
 * Local (file-private) helpers
 * ============================================================ */

static bool rocke__dtype_is(const rocke_type_t* dt, const char* nm)
{
    return dt != NULL && dt->name != NULL && nm != NULL && (strcmp(dt->name, nm) == 0);
}

/* mfma_16x16x32_for_dtype (helpers/attention.py): dispatch
 * mfma_f32_16x16x32_<dtype> for fp16 / bf16. Any other dtype is a Python
 * ValueError -> NULL. */
static rocke_value_t* rocke__mfma_16x16x32(rocke_ir_builder_t* b,
                                           const rocke_type_t* dtype,
                                           rocke_value_t* a,
                                           rocke_value_t* bv,
                                           rocke_value_t* c)
{
    if(rocke__dtype_is(dtype, "f16"))
    {
        return rocke_b_mfma_f32_16x16x32_f16(b, a, bv, c);
    }
    if(rocke__dtype_is(dtype, "bf16"))
    {
        return rocke_b_mfma_f32_16x16x32_bf16(b, a, bv, c);
    }
    return NULL;
}

/* mfma_16x16x16_for_dtype (the narrow atom for the PV_K_STEP==16 path). The
 * ported C helper lives in helper_helper_rocke.helpers.attention.h. */
static rocke_value_t* rocke__mfma_16x16x16(rocke_ir_builder_t* b,
                                           const rocke_type_t* dtype,
                                           rocke_value_t* a,
                                           rocke_value_t* bv,
                                           rocke_value_t* c)
{
    return rocke_mfma_16x16x16_for_dtype(b, dtype, a, bv, c);
}

/* warp_xor_reduce_max(b, v, stages=4) (helpers/attention.py): wave64 16-lane
 * butterfly max via warp_shuffle_xor (masks 1,2,4,8) + fmax. Not exported as a
 * C helper symbol, so reproduce its byte-faithful op order here. */
static rocke_value_t* rocke__warp_xor_reduce_max(rocke_ir_builder_t* b, rocke_value_t* v)
{
    rocke_value_t* cur = v;
    int k;
    for(k = 0; k < 4; k++)
    {
        rocke_value_t* remote = rocke_b_warp_shuffle_xor(b, cur, 1 << k);
        cur = rocke_b_fmax(b, cur, remote);
    }
    return cur;
}

/* _mfma_16x16_c_row peer (lines 81-96) via the internal header. */
static rocke_value_t* rocke__c_row(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx, int reg)
{
    return rocke_gfx950_attention_tiled_3d_mfma_16x16_c_row(ctx, ctx->tid, reg);
}

/* Combine v with lane l ^ 32 (one VALU permlane32 swap) -- Python _xor32_reduce. */
static rocke_value_t* rocke__xor32_reduce(rocke_ir_builder_t* b, rocke_value_t* v, bool is_sum)
{
    rocke_value_t* vi = rocke_b_bitcast(b, v, rocke_i32());
    rocke_value_t* lo = NULL;
    rocke_value_t* hi = NULL;
    rocke_value_t* lo_f;
    rocke_value_t* hi_f;
    rocke_b_permlane32_swap(b, vi, vi, &lo, &hi);
    lo_f = rocke_b_bitcast(b, lo, rocke_f32());
    hi_f = rocke_b_bitcast(b, hi, rocke_f32());
    return is_sum ? rocke_b_fadd(b, lo_f, hi_f) : rocke_b_fmax(b, lo_f, hi_f);
}

/* Combine v across the 4 lane groups lane / 16 (same lane % 16) -- Python
 * _group_reduce (fmax when !is_sum, fadd when is_sum). */
static rocke_value_t* rocke__group_reduce(rocke_ir_builder_t* b, rocke_value_t* v, bool is_sum)
{
    rocke_value_t* other = rocke_b_warp_shuffle_xor(b, v, 16);
    rocke_value_t* c = is_sum ? rocke_b_fadd(b, v, other) : rocke_b_fmax(b, v, other);
    return rocke__xor32_reduce(b, c, is_sum);
}

/* Query row (within BLOCK_M) of softmax state slot r of this lane (Python
 * _state_row): the lane's own row when transposed, else the MFMA C row. */
static rocke_value_t* rocke__state_row(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx, int r)
{
    return ctx->cfg.TRANS ? ctx->lane_col : rocke__c_row(ctx, r);
}

/* Append a block-id list (TILE_MULT entries, or NULL) to a prefetch carry. */
static void rocke__append_pf(rocke_value_t** dst, int* n, rocke_value_t* const* src, int count)
{
    int i;
    if(src == NULL)
    {
        return;
    }
    for(i = 0; i < count; i++)
    {
        dst[(*n)++] = src[i];
    }
}

/* ============================================================ *
 * emit_loop_init: transposed-softmax Q operands, softmax/acc carry inits,
 * async DMA infra, and the first K/V issue (incl. the prefetch carry).
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_loop_init(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx950_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    const rocke_type_t* dtype = cfg->dtype;
    int n, r, k;

    if(cfg->TRANS)
    {
        /* Lane owns query row lane % 16; Q is the loop-invariant B operand of
         * the transposed QK MFMA, held in registers for the whole loop. */
        for(k = 0; k < cfg->QK_K_ITERS; k++)
        {
            rocke_value_t* q_col_c = rocke_b_const_i32(b, k * 32);
            rocke_value_t* q_col_m = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 8));
            rocke_value_t* q_col_off = rocke_b_add(b, q_col_c, q_col_m);
            rocke_value_t* idx[3];
            idx[0] = rocke_b_const_i32(b, 1);
            idx[1] = ctx->lane_col;
            idx[2] = q_col_off;
            ctx->q_kits[k] = rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, dtype, 8);
        }
    }

    /* ---- m_inits (sinks-conditioned) ---- */
    if(cfg->USE_SINKS)
    {
        /* Triton's 3D applies sinks only when segm_idx == 0. */
        rocke_value_t* seg0 = rocke_b_cmp_eq(b, ctx->seg_idx, rocke_b_const_i32(b, 0));
        for(r = 0; r < cfg->NR; r++)
        {
            rocke_value_t* row = rocke__state_row(ctx, r);
            rocke_value_t* qh_mul
                = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, cfg->NQK));
            rocke_value_t* qh_mod = rocke_b_mod(b, row, rocke_b_const_i32(b, cfg->NQK));
            rocke_value_t* qh = rocke_b_add(b, qh_mul, qh_mod);
            rocke_value_t* qh_in = rocke_b_cmp_lt(b, qh, rocke_b_const_i32(b, cfg->NUM_QH));
            rocke_value_t* sink_h = rocke_b_global_load(b, ctx->sinks, qh, dtype, 2);
            rocke_value_t* sink_f = rocke_b_fmul(b, rocke_b_cast_to_f32(b, sink_h), ctx->rcp_ln2);
            rocke_value_t* sink_with_mask = rocke_b_select(b, qh_in, sink_f, ctx->neg_inf);
            ctx->m_inits[r] = rocke_b_select(b, seg0, sink_with_mask, ctx->neg_inf);
        }
    }
    else
    {
        for(r = 0; r < cfg->NR; r++)
        {
            ctx->m_inits[r] = ctx->neg_inf;
        }
    }
    if(cfg->TRANS)
    {
        /* The row sum is a per-lane partial over this lane's keys, reduced
         * across the 4 lane groups after the loop: seed the 1 on group 0 only. */
        rocke_value_t* g0 = rocke_b_cmp_eq(b, ctx->lane_rg, rocke_b_const_i32(b, 0));
        ctx->l_inits[0] = rocke_b_select(b, g0, ctx->one_f, ctx->zero_f);
    }
    else
    {
        for(r = 0; r < 4; r++)
        {
            ctx->l_inits[r] = ctx->one_f;
        }
    }

    /* ---- acc_inits ---- */
    {
        rocke_value_t* acc_zero = rocke_b_zero_vec_f32(b, 4);
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            ctx->acc_inits[n] = acc_zero;
        }
    }

    /* ---- async DMA infra + paged-KV descriptor ----
     * Emitted here (not in emit_prologue) so the SSA op order matches Python's
     * single linear build: acc_zero precedes the buffer rsrc / seq_base / desc. */
    rocke_gfx950_attention_tiled_3d_emit_async_infra(ctx);

    /* ---- first K (and V) issue + prefetch carry ---- */
    ctx->n_pf = 0;
    ctx->pf_inits = NULL;
    if(cfg->PREFETCH)
    {
        int cap = 2 * cfg->fp8_chunks_per_thread + cfg->TILE_MULT;
        ctx->pf_inits
            = (rocke_value_t**)rocke_arena_alloc(&b->arena, (size_t)cap * sizeof(rocke_value_t*));
        if(ctx->pf_inits == NULL)
        {
            return;
        }
    }
    if(cfg->FP8_REG_PREFETCH)
    {
        /* tile_start straight into LDS buffer 0; tile_start+1 rides in
         * registers; block ids run one tile further ahead. */
        rocke_value_t** raw = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)(2 * cfg->fp8_chunks_per_thread) * sizeof(rocke_value_t*));
        rocke_value_t** ids0;
        rocke_value_t** ids1;
        rocke_value_t** ids2;
        rocke_value_t* nxt0;
        rocke_value_t* nxt1;
        if(raw == NULL)
        {
            return;
        }
        ids0 = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, ctx->tile_start);
        rocke_gfx950_attention_tiled_3d_fp8_raw_tile(ctx, ctx->tile_start, ids0, raw);
        rocke_gfx950_attention_tiled_3d_fp8_stage_tile(ctx, raw, rocke_b_const_i32(b, 0));
        nxt0 = rocke_b_add(b, ctx->tile_start, rocke_b_const_i32(b, 1));
        ids1 = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, nxt0);
        rocke_gfx950_attention_tiled_3d_fp8_raw_tile(ctx, nxt0, ids1, ctx->pf_inits);
        ctx->n_pf = 2 * cfg->fp8_chunks_per_thread;
        nxt1 = rocke_b_add(b, ctx->tile_start, rocke_b_const_i32(b, 2));
        ids2 = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, nxt1);
        rocke__append_pf(ctx->pf_inits, &ctx->n_pf, ids2, cfg->TILE_MULT);
    }
    else if(cfg->PREFETCH && cfg->KV_SPLIT_WAIT)
    {
        /* Block ids ahead of the DMAs, so the partial vmcnt waits in the loop
         * count K/V calls only (VMEM returns in issue order). */
        rocke_value_t** ids0 = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, ctx->tile_start);
        rocke_value_t* nxt = rocke_b_add(b, ctx->tile_start, rocke_b_const_i32(b, 1));
        rocke_value_t** ids1 = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, nxt);
        rocke_gfx950_attention_tiled_3d_issue_k_load(
            ctx, ctx->tile_start, rocke_b_const_i32(b, 0), ids0);
        rocke_gfx950_attention_tiled_3d_issue_v_load(
            ctx, ctx->tile_start, rocke_b_const_i32(b, 0), ids0);
        rocke__append_pf(ctx->pf_inits, &ctx->n_pf, ids1, cfg->TILE_MULT);
    }
    else if(cfg->PREFETCH)
    {
        rocke_value_t** ids0 = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, ctx->tile_start);
        rocke_value_t* nxt;
        rocke_value_t** ids1;
        rocke_gfx950_attention_tiled_3d_issue_k_load(
            ctx, ctx->tile_start, rocke_b_const_i32(b, 0), ids0);
        rocke_gfx950_attention_tiled_3d_issue_v_load(
            ctx, ctx->tile_start, rocke_b_const_i32(b, 0), ids0);
        nxt = rocke_b_add(b, ctx->tile_start, rocke_b_const_i32(b, 1));
        ids1 = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, nxt);
        rocke__append_pf(ctx->pf_inits, &ctx->n_pf, ids1, cfg->TILE_MULT);
    }
    else
    {
        rocke_gfx950_attention_tiled_3d_issue_k(ctx, ctx->tile_start, rocke_b_const_i32(b, 0));
    }
    ctx->cur_buf_init = rocke_b_const_i32(b, 0);
}

/* ============================================================ *
 * emit_softmax_loop: the online-softmax scf.for over [tile_start, tile_end).
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_softmax_loop(
    rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx950_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    const rocke_type_t* dtype = cfg->dtype;
    const rocke_type_t* f32 = rocke_f32();
    const int NR = cfg->NR;
    const bool TRANS = cfg->TRANS;
    const int TM = cfg->TILE_MULT;
    int r, n, k, reg;

    /* iter_args: m<r>,l<r> (r < NR), acc0..accN-1, cur_buf, kvpf<i> */
    int num_iter = 2 * NR + cfg->PV_N_TILES + 1 + ctx->n_pf;
    rocke_iter_arg_t* iter_args
        = (rocke_iter_arg_t*)rocke_arena_alloc(&b->arena, (size_t)num_iter * sizeof(*iter_args));
    if(iter_args == NULL)
    {
        return;
    }
    {
        int ai = 0;
        int i;
        char buf[24];
        for(r = 0; r < NR; r++)
        {
            snprintf(buf, sizeof(buf), "m%d", r);
            iter_args[ai].name = rocke_arena_strdup(&b->arena, buf);
            iter_args[ai].init = ctx->m_inits[r];
            ai++;
            snprintf(buf, sizeof(buf), "l%d", r);
            iter_args[ai].name = rocke_arena_strdup(&b->arena, buf);
            iter_args[ai].init = ctx->l_inits[r];
            ai++;
        }
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            snprintf(buf, sizeof(buf), "acc%d", n);
            iter_args[ai].name = rocke_arena_strdup(&b->arena, buf);
            iter_args[ai].init = ctx->acc_inits[n];
            ai++;
        }
        iter_args[ai].name = rocke_arena_strdup(&b->arena, "cur_buf");
        iter_args[ai].init = ctx->cur_buf_init;
        ai++;
        for(i = 0; i < ctx->n_pf; i++)
        {
            snprintf(buf, sizeof(buf), "kvpf%d", i);
            iter_args[ai].name = rocke_arena_strdup(&b->arena, buf);
            iter_args[ai].init = ctx->pf_inits[i];
            ai++;
        }
    }

    /* LICM hoist: per-row invariants constant across all KV tiles. qp_r, qh_r,
     * row_ok, and causal_lim depend only on CTA-level constants and the
     * per-thread row index. ALiBi slopes are per query-head: also invariant. */
    {
        int hoist_reg;
        for(hoist_reg = 0; hoist_reg < NR; hoist_reg++)
        {
            rocke_value_t* row = rocke__state_row(ctx, hoist_reg);
            rocke_value_t* qp_r_div = rocke_b_div(b, row, rocke_b_const_i32(b, cfg->NQK));
            rocke_value_t* qp_r = rocke_b_add(b, ctx->qb_start_pos, qp_r_div);
            rocke_value_t* qh_mul
                = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, cfg->NQK));
            rocke_value_t* qh_mod = rocke_b_mod(b, row, rocke_b_const_i32(b, cfg->NQK));
            rocke_value_t* qh_r = rocke_b_add(b, qh_mul, qh_mod);
            rocke_value_t* ok_a = rocke_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
            rocke_value_t* ok_b = rocke_b_cmp_lt(b, qh_r, rocke_b_const_i32(b, cfg->NUM_QH));
            ctx->hoist_qp_r[hoist_reg] = qp_r;
            ctx->hoist_qh_r[hoist_reg] = qh_r;
            ctx->hoist_row_ok[hoist_reg] = rocke_b_land(b, ok_a, ok_b);
            ctx->hoist_causal_lim[hoist_reg] = rocke_b_add(b, ctx->context_len, qp_r);
            ctx->hoist_alibi[hoist_reg] = NULL;
        }
        if(cfg->USE_ALIBI)
        {
            for(hoist_reg = 0; hoist_reg < 4; hoist_reg++)
            {
                rocke_value_t* qh_r = ctx->hoist_qh_r[hoist_reg];
                rocke_value_t* qh_ok = rocke_b_cmp_lt(b, qh_r, rocke_b_const_i32(b, cfg->NUM_QH));
                ctx->hoist_alibi[hoist_reg] = rocke_b_masked_global_load(
                    b, ctx->alibi_slopes_ptr, qh_r, qh_ok, rocke_b_const_f32(b, 0.0), f32, 4);
            }
        }
    }

    if(TRANS)
    {
        /* Transposed QK: A = K rows, B = Q (q_kits), C[key_slot, query]. Lane
         * (g = lane/16, c = lane%16) gets key slots g*4 + r of each 16-key
         * n-tile. For 32-key PV steps, n-tile 2kk+t loads K row
         * kk*32 + (c//4)*8 + t*4 + c%4 into A slot c, so slot g*4 + r holds key
         * kk*32 + g*8 + t*4 + r: the lane's 8 P values over (t, r) are the 8
         * consecutive keys the V transpose-read gives the lane. */
        if(cfg->PV_K_STEP == 32)
        {
            rocke_value_t* c_div = rocke_b_div(b, ctx->lane_col, rocke_b_const_i32(b, 4));
            rocke_value_t* c_mul = rocke_b_mul(b, c_div, rocke_b_const_i32(b, 8));
            rocke_value_t* c_mod = rocke_b_mod(b, ctx->lane_col, rocke_b_const_i32(b, 4));
            ctx->k_row_lane = rocke_b_add(b, c_mul, c_mod);
            ctx->key_lane = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 8));
        }
        else
        {
            ctx->k_row_lane = ctx->lane_col;
            ctx->key_lane = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 4));
        }
    }

    rocke_for_t kvloop = rocke_b_scf_for_iter(b,
                                              ctx->tile_start,
                                              ctx->tile_end,
                                              rocke_b_const_i32(b, 1),
                                              iter_args,
                                              num_iter,
                                              "kv_tile",
                                              false,
                                              true);

    rocke_b_region_enter(b, kvloop.body);
    {
        rocke_value_t* kv_tile_iv = kvloop.iv;
        rocke_value_t** carry = kvloop.iter_vars;
        const int pf_base = 2 * NR + cfg->PV_N_TILES + 1;

        rocke_value_t* m_vals[4];
        rocke_value_t* l_vals[4];
        rocke_value_t** acc_vals;
        rocke_value_t* cur_buf;
        rocke_value_t* nxt_buf;
        rocke_value_t* tile_off;
        rocke_value_t* next_tile_iv_raw;
        rocke_value_t* in_range_next;
        rocke_value_t* safe_next_tile;

        rocke_value_t** A_kits;
        rocke_value_t** S_n;
        rocke_value_t* alibi_per_row[4];
        rocke_value_t* m_new[4];
        rocke_value_t* l_local[4];
        rocke_value_t* alpha_regs[4];
        rocke_value_t* new_l_vals[4];
        rocke_value_t** new_acc;
        rocke_value_t** new_pf;
        int n_new_pf = 0;
        rocke_value_t** p_t = NULL; /* TRANS: P per (n, r), indexed n*4+r */
        rocke_value_t** p_ops = NULL; /* TRANS: PV B operand per k step   */

        /* masked[(n,reg)] indexed n*4+reg; s_local[(reg,n)] indexed reg*N+n */
        rocke_value_t** masked;

        acc_vals = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)cfg->PV_N_TILES * sizeof(rocke_value_t*));
        new_acc = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)cfg->PV_N_TILES * sizeof(rocke_value_t*));
        A_kits = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)cfg->QK_K_ITERS * sizeof(rocke_value_t*));
        S_n = (rocke_value_t**)rocke_arena_alloc(&b->arena,
                                                 (size_t)cfg->QK_N_TILES * sizeof(rocke_value_t*));
        masked = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)(cfg->QK_N_TILES * 4) * sizeof(rocke_value_t*));
        new_pf = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)(ctx->n_pf + 1) * sizeof(rocke_value_t*));
        if(acc_vals == NULL || new_acc == NULL || A_kits == NULL || S_n == NULL || masked == NULL
           || new_pf == NULL)
        {
            rocke_b_region_leave(b);
            return;
        }

        for(r = 0; r < NR; r++)
        {
            m_vals[r] = carry[2 * r];
            l_vals[r] = carry[2 * r + 1];
        }
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            acc_vals[n] = carry[2 * NR + n];
        }
        cur_buf = carry[2 * NR + cfg->PV_N_TILES];
        nxt_buf = rocke_b_sub(b, rocke_b_const_i32(b, 1), cur_buf);
        tile_off = rocke_b_mul(b, kv_tile_iv, rocke_b_const_i32(b, cfg->T));

        next_tile_iv_raw = rocke_b_add(b, kv_tile_iv, rocke_b_const_i32(b, 1));
        in_range_next = rocke_b_cmp_lt(b, next_tile_iv_raw, ctx->tile_end);
        safe_next_tile = rocke_b_select(b, in_range_next, next_tile_iv_raw, kv_tile_iv);

        if(cfg->FP8_REG_PREFETCH)
        {
            /* LDS[cur] was staged last iteration; only LDS needs draining, the
             * raw fp8 registers for tile+1 stay in flight until staged here. */
            const int n_raw = 2 * cfg->fp8_chunks_per_thread;
            rocke_value_t** raw = carry + pf_base;
            rocke_value_t** ids_n2 = carry + pf_base + n_raw;
            rocke_value_t* nxt2;
            rocke_value_t* nxt3;
            rocke_value_t** ids_n3;
            rocke_b_sync_lds_only(b);
            rocke_gfx950_attention_tiled_3d_fp8_stage_tile(ctx, raw, nxt_buf);
            nxt2 = rocke_b_add(b, kv_tile_iv, rocke_b_const_i32(b, 2));
            rocke_gfx950_attention_tiled_3d_fp8_raw_tile(ctx, nxt2, ids_n2, new_pf);
            n_new_pf = n_raw;
            nxt3 = rocke_b_add(b, kv_tile_iv, rocke_b_const_i32(b, 3));
            ids_n3 = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, nxt3);
            rocke__append_pf(new_pf, &n_new_pf, ids_n3, TM);
        }
        else if(cfg->KV_SPLIT_WAIT)
        {
            /* In flight here (issue order): ids[+1], K[cur], V[cur]. Wait for
             * K[cur] only; V[cur] lands under QK + softmax. */
            rocke_value_t** ids_n1 = cfg->PREFETCH_IDS ? carry + pf_base : NULL;
            rocke_if_t g;
            rocke_b_s_waitcnt(b, cfg->kv_calls_per_tile, 0, -1);
            rocke_b_s_barrier_bare(b);
            if(cfg->PREFETCH_IDS)
            {
                rocke_value_t* nxt2 = rocke_b_add(b, kv_tile_iv, rocke_b_const_i32(b, 2));
                rocke_value_t** ids_n2 = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, nxt2);
                rocke__append_pf(new_pf, &n_new_pf, ids_n2, TM);
            }
            g = rocke_b_scf_if(b, in_range_next);
            rocke_b_region_enter(b, g.then_region);
            rocke_gfx950_attention_tiled_3d_issue_k_load(ctx, next_tile_iv_raw, nxt_buf, ids_n1);
            rocke_gfx950_attention_tiled_3d_issue_v_load(ctx, next_tile_iv_raw, nxt_buf, ids_n1);
            rocke_b_region_leave(b);
        }
        else
        {
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_sync(b);
            if(cfg->PREFETCH)
            {
                rocke_value_t** ids_n1 = cfg->PREFETCH_IDS ? carry + pf_base : NULL;
                rocke_if_t g = rocke_b_scf_if(b, in_range_next);
                rocke_b_region_enter(b, g.then_region);
                rocke_gfx950_attention_tiled_3d_issue_k_load(
                    ctx, next_tile_iv_raw, nxt_buf, ids_n1);
                rocke_gfx950_attention_tiled_3d_issue_v_load(
                    ctx, next_tile_iv_raw, nxt_buf, ids_n1);
                rocke_b_region_leave(b);
                if(cfg->PREFETCH_IDS)
                {
                    rocke_value_t* nxt2 = rocke_b_add(b, kv_tile_iv, rocke_b_const_i32(b, 2));
                    rocke_value_t** ids_n2
                        = rocke_gfx950_attention_tiled_3d_tile_block_ids(ctx, nxt2);
                    rocke__append_pf(new_pf, &n_new_pf, ids_n2, TM);
                }
            }
        }

        /* ---------------- QK (wide 16x16x32, K-step 32) ---------------- */
        if(TRANS)
        {
            /* S^T n-tile = K(permuted rows) x Q^T; Q operands live in q_kits. */
            for(n = 0; n < cfg->QK_N_TILES; n++)
            {
                rocke_value_t* acc_v = rocke_b_zero_vec_f32(b, 4);
                const int row_c = (cfg->PV_K_STEP == 32) ? (n / 2) * 32 + (n % 2) * 4 : n * 16;
                rocke_value_t* k_row = rocke_b_add(b, rocke_b_const_i32(b, row_c), ctx->k_row_lane);
                for(k = 0; k < cfg->QK_K_ITERS; k++)
                {
                    rocke_value_t* kc_c = rocke_b_const_i32(b, k * 32);
                    rocke_value_t* kc_m = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 8));
                    rocke_value_t* kc_off = rocke_b_add(b, kc_c, kc_m);
                    rocke_value_t* idx[3];
                    rocke_value_t* K_v;
                    idx[0] = cur_buf;
                    idx[1] = k_row;
                    idx[2] = kc_off;
                    K_v = rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, dtype, 8);
                    acc_v = rocke__mfma_16x16x32(b, dtype, K_v, ctx->q_kits[k], acc_v);
                }
                S_n[n] = acc_v;
            }
        }
        else
        {
            for(k = 0; k < cfg->QK_K_ITERS; k++)
            {
                rocke_value_t* q_col_c = rocke_b_const_i32(b, k * 32);
                rocke_value_t* q_col_m = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 8));
                rocke_value_t* q_col_off = rocke_b_add(b, q_col_c, q_col_m);
                rocke_value_t* idx[2];
                idx[0] = ctx->lane_col;
                idx[1] = q_col_off;
                A_kits[k] = rocke_b_smem_load_vN(b, ctx->Q_lds, idx, 2, dtype, 8);
            }
            for(n = 0; n < cfg->QK_N_TILES; n++)
            {
                rocke_value_t* acc_v = rocke_b_zero_vec_f32(b, 4);
                for(k = 0; k < cfg->QK_K_ITERS; k++)
                {
                    rocke_value_t* kc_c = rocke_b_const_i32(b, k * 32);
                    rocke_value_t* kc_m = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 8));
                    rocke_value_t* kc_off = rocke_b_add(b, kc_c, kc_m);
                    rocke_value_t* k_row
                        = rocke_b_add(b, rocke_b_const_i32(b, n * 16), ctx->lane_col);
                    rocke_value_t* idx[3];
                    rocke_value_t* B_v;
                    idx[0] = cur_buf;
                    idx[1] = k_row;
                    idx[2] = kc_off;
                    B_v = rocke_b_smem_load_vN(b, ctx->K_lds, idx, 3, dtype, 8);
                    acc_v = rocke__mfma_16x16x32(b, dtype, A_kits[k], B_v, acc_v);
                }
                S_n[n] = acc_v;
            }
        }

        if(!cfg->PREFETCH)
        {
            rocke_gfx950_attention_tiled_3d_issue_v(ctx, kv_tile_iv, cur_buf);
            rocke_gfx950_attention_tiled_3d_issue_k(ctx, safe_next_tile, nxt_buf);
        }

        /* alibi_per_row: use hoisted slopes (NULL when USE_ALIBI is false). */
        for(reg = 0; reg < 4; reg++)
        {
            alibi_per_row[reg] = ctx->hoist_alibi[reg];
        }

        /* ---------------- masked scores ---------------- */
        if(TRANS)
        {
            /* One query row per lane; slot (n, r) is key tile_key + _trans_key. */
            rocke_value_t* row_ok = ctx->hoist_row_ok[0];
            rocke_value_t* causal_lim = ctx->hoist_causal_lim[0];
            rocke_value_t* tile_key = rocke_b_add(b, tile_off, ctx->key_lane);
            for(n = 0; n < cfg->QK_N_TILES; n++)
            {
                for(r = 0; r < 4; r++)
                {
                    const int tkey
                        = (cfg->PV_K_STEP == 32) ? (n / 2) * 32 + (n % 2) * 4 + r : n * 16 + r;
                    rocke_value_t* col_abs = rocke_b_add(b, tile_key, rocke_b_const_i32(b, tkey));
                    rocke_value_t* causal_ok = rocke_b_cmp_le(b, col_abs, causal_lim);
                    rocke_value_t* in_prefix = rocke_b_cmp_lt(b, col_abs, ctx->max_seq_prefix_len);
                    rocke_value_t* m_ok
                        = rocke_b_land(b, rocke_b_land(b, row_ok, causal_ok), in_prefix);
                    rocke_value_t* s_scaled;
                    if(cfg->SLIDING_WINDOW > 0)
                    {
                        rocke_value_t* dist = rocke_b_sub(b, causal_lim, col_abs);
                        m_ok = rocke_b_land(b, m_ok, rocke_b_cmp_lt(b, dist, ctx->sw_const));
                    }
                    s_scaled = rocke_b_fmul(b, rocke_b_vec_extract(b, S_n[n], r), ctx->qk_scale);
                    if(cfg->USE_SOFTCAP)
                    {
                        s_scaled = rocke_b_fmul(
                            b, rocke_apply_softcap_log2(b, s_scaled, ctx->softcap_p), ctx->rcp_ln2);
                    }
                    masked[n * 4 + r] = rocke_b_select(b, m_ok, s_scaled, ctx->neg_inf);
                }
            }
        }
        for(reg = 0; reg < (TRANS ? 0 : 4); reg++)
        {
            rocke_value_t* qp_r = ctx->hoist_qp_r[reg];
            rocke_value_t* row_ok = ctx->hoist_row_ok[reg];
            rocke_value_t* causal_lim = ctx->hoist_causal_lim[reg];
            for(n = 0; n < cfg->QK_N_TILES; n++)
            {
                rocke_value_t* ca_n = rocke_b_const_i32(b, n);
                rocke_value_t* ca_16 = rocke_b_const_i32(b, 16);
                rocke_value_t* col_abs = rocke_b_add(
                    b, rocke_b_add(b, tile_off, rocke_b_mul(b, ca_n, ca_16)), ctx->lane_col);
                rocke_value_t* causal_ok = rocke_b_cmp_le(b, col_abs, causal_lim);
                rocke_value_t* in_prefix = rocke_b_cmp_lt(b, col_abs, ctx->max_seq_prefix_len);
                rocke_value_t* m_ok
                    = rocke_b_land(b, rocke_b_land(b, row_ok, causal_ok), in_prefix);
                rocke_value_t* s_raw;
                rocke_value_t* s_scaled;
                if(cfg->SLIDING_WINDOW > 0)
                {
                    rocke_value_t* dist = rocke_b_sub(b, causal_lim, col_abs);
                    m_ok = rocke_b_land(b, m_ok, rocke_b_cmp_lt(b, dist, ctx->sw_const));
                }
                s_raw = rocke_b_vec_extract(b, S_n[n], reg);
                s_scaled = rocke_b_fmul(b, s_raw, ctx->qk_scale);
                if(cfg->USE_SOFTCAP)
                {
                    s_scaled = rocke_b_fmul(
                        b, rocke_apply_softcap_log2(b, s_scaled, ctx->softcap_p), ctx->rcp_ln2);
                }
                if(cfg->USE_ALIBI)
                {
                    rocke_value_t* pos_off = rocke_b_sub(b, col_abs, ctx->context_len);
                    rocke_value_t* pos_f = rocke_b_sitofp_f32(b, pos_off);
                    rocke_value_t* add_term
                        = rocke_b_fmul(b, rocke_b_fmul(b, alibi_per_row[reg], pos_f), ctx->rcp_ln2);
                    s_scaled = rocke_b_fadd(b, s_scaled, add_term);
                }
                if(cfg->USE_QQ_BIAS)
                {
                    /* Sequenced so the op order matches Python's left-to-right
                     * argument evaluation. */
                    rocke_value_t* krp = rocke_b_sub(b, col_abs, ctx->context_len);
                    rocke_value_t* krp_ge = rocke_b_cmp_ge(b, krp, rocke_b_const_i32(b, 0));
                    rocke_value_t* krp_lt = rocke_b_cmp_lt(b, krp, ctx->qq_bias_stride0_p);
                    rocke_value_t* krp_ok = rocke_b_land(b, krp_ge, krp_lt);
                    rocke_value_t* qq_ok = rocke_b_land(b, row_ok, krp_ok);
                    rocke_value_t* qp_safe
                        = rocke_b_select(b, row_ok, qp_r, rocke_b_const_i32(b, 0));
                    rocke_value_t* qq_idx
                        = rocke_b_add(b, rocke_b_mul(b, qp_safe, ctx->qq_bias_stride0_p), krp);
                    rocke_value_t* qq_zero = rocke_b_const_f32(b, 0.0);
                    rocke_value_t* qq_v = rocke_b_masked_global_load(
                        b, ctx->qq_bias_ptr, qq_idx, qq_ok, qq_zero, f32, 4);
                    s_scaled = rocke_b_fadd(b, s_scaled, rocke_b_fmul(b, qq_v, ctx->rcp_ln2));
                }
                masked[n * 4 + reg] = rocke_b_select(b, m_ok, s_scaled, ctx->neg_inf);
            }
        }

        /* ---------------- online (m,l) update ---------------- */
        if(TRANS)
        {
            /* Row max: in-lane over the tile, then across the 4 lane groups. */
            rocke_value_t* local_max = ctx->neg_inf;
            rocke_value_t* full_max_raw;
            rocke_value_t* ok;
            rocke_value_t* sum_p = ctx->zero_f;
            p_t = (rocke_value_t**)rocke_arena_alloc(
                &b->arena, (size_t)(cfg->QK_N_TILES * 4) * sizeof(rocke_value_t*));
            if(p_t == NULL)
            {
                rocke_b_region_leave(b);
                return;
            }
            for(n = 0; n < cfg->QK_N_TILES; n++)
            {
                for(r = 0; r < 4; r++)
                {
                    local_max = rocke_b_fmax(b, local_max, masked[n * 4 + r]);
                }
            }
            full_max_raw = rocke_b_fmax(b, m_vals[0], rocke__group_reduce(b, local_max, false));
            ok = rocke_b_fcmp(b, "ogt", full_max_raw, ctx->neg_inf);
            m_new[0] = rocke_b_select(b, ok, full_max_raw, ctx->zero_f);
            for(n = 0; n < cfg->QK_N_TILES; n++)
            {
                for(r = 0; r < 4; r++)
                {
                    p_t[n * 4 + r] = rocke_b_exp2(b, rocke_b_fsub(b, masked[n * 4 + r], m_new[0]));
                    sum_p = rocke_b_fadd(b, sum_p, p_t[n * 4 + r]);
                }
            }
            alpha_regs[0] = rocke_b_exp2(b, rocke_b_fsub(b, m_vals[0], m_new[0]));
            /* Per-lane partial row sum; reduced across lane groups after the loop. */
            new_l_vals[0] = rocke_b_fadd(b, rocke_b_fmul(b, l_vals[0], alpha_regs[0]), sum_p);
        }
        else
        {
            rocke_value_t** s_local = (rocke_value_t**)rocke_arena_alloc(
                &b->arena, (size_t)(4 * cfg->QK_N_TILES) * sizeof(rocke_value_t*));
            if(s_local == NULL)
            {
                rocke_b_region_leave(b);
                return;
            }
            for(reg = 0; reg < 4; reg++)
            {
                rocke_value_t* local_max = ctx->neg_inf;
                rocke_value_t* full_max_raw;
                rocke_value_t* ok;
                for(n = 0; n < cfg->QK_N_TILES; n++)
                {
                    rocke_value_t* v = masked[n * 4 + reg];
                    s_local[reg * cfg->QK_N_TILES + n] = v;
                    local_max = rocke_b_fmax(b, local_max, v);
                }
                full_max_raw = rocke__warp_xor_reduce_max(b, local_max);
                /* Online-softmax running max: max(m_old, tile_max). A bare tile
                 * max lets alpha = exp2(m_old - tile_max) exceed 1 and overflow
                 * f32 once logits spread more than 128 (log2 units) across tiles. */
                full_max_raw = rocke_b_fmax(b, m_vals[reg], full_max_raw);
                ok = rocke_b_fcmp(b, "ogt", full_max_raw, ctx->neg_inf);
                m_new[reg] = rocke_b_select(b, ok, full_max_raw, ctx->zero_f);
            }

            for(reg = 0; reg < 4; reg++)
            {
                rocke_value_t* row = rocke__c_row(ctx, reg);
                rocke_value_t* sum_p = ctx->zero_f;
                for(n = 0; n < cfg->QK_N_TILES; n++)
                {
                    rocke_value_t* p = rocke_b_exp2(
                        b, rocke_b_fsub(b, s_local[reg * cfg->QK_N_TILES + n], m_new[reg]));
                    rocke_value_t* pcol_n = rocke_b_const_i32(b, n);
                    rocke_value_t* pcol_16 = rocke_b_const_i32(b, 16);
                    rocke_value_t* col
                        = rocke_b_add(b, rocke_b_mul(b, pcol_n, pcol_16), ctx->lane_col);
                    rocke_value_t* idx[2];
                    idx[0] = row;
                    idx[1] = col;
                    rocke_b_smem_store_vN(
                        b, ctx->P_lds, idx, 2, rocke_b_cast_f32_to(b, p, dtype), 1);
                    sum_p = rocke_b_fadd(b, sum_p, p);
                }
                l_local[reg] = rocke_warp_xor_reduce_sum(b, sum_p, 4);
            }

            for(r = 0; r < 4; r++)
            {
                alpha_regs[r] = rocke_b_exp2(b, rocke_b_fsub(b, m_vals[r], m_new[r]));
            }
            for(r = 0; r < 4; r++)
            {
                new_l_vals[r]
                    = rocke_b_fadd(b, rocke_b_fmul(b, l_vals[r], alpha_regs[r]), l_local[r]);
            }
        }

        if(cfg->PREFETCH)
        {
            if(cfg->KV_SPLIT_WAIT)
            {
                /* In flight: V[cur], ids[+2], then K/V[next] when issued.
                 * Leave the younger loads outstanding; V[cur] must land. */
                const int n_ids = (cfg->PREFETCH_IDS && !cfg->SCALAR_BLOCK_IDS) ? TM : 0;
                rocke_if_t g = rocke_b_scf_if(b, in_range_next);
                rocke_if_t g_last;
                rocke_b_region_enter(b, g.then_region);
                rocke_b_s_waitcnt(b, 2 * cfg->kv_calls_per_tile + n_ids, -1, -1);
                rocke_b_region_leave(b);
                g_last = rocke_b_scf_if(b, rocke_b_cmp_ge(b, next_tile_iv_raw, ctx->tile_end));
                rocke_b_region_enter(b, g_last.then_region);
                rocke_b_s_waitcnt(b, n_ids, -1, -1);
                rocke_b_region_leave(b);
            }
            if(!TRANS)
            {
                /* K/V[cur] already landed; only the P tile written above must
                 * be visible. Next tile's loads stay in flight. */
                rocke_b_sync_lds_only(b);
            }
            else if(cfg->KV_SPLIT_WAIT)
            {
                rocke_b_s_barrier_bare(b);
            }
        }
        else if(cfg->KV_FP8)
        {
            /* FP8 sync loader has no in-flight async work. */
            rocke_b_s_waitcnt(b, 0, 0, -1);
            rocke_b_sync(b);
        }
        else
        {
            rocke_b_s_waitcnt(b, cfg->kv_calls_per_tile, cfg->kv_calls_per_tile, -1);
            rocke_b_sync(b);
        }

        /* Keys >= seq_len in the sequence's last tile hold whatever the cache
         * has in the page slack (or a clamped page). Their P is 0, but
         * 0 * NaN = NaN inside the PV MFMA, so zero those V rows in LDS before
         * PV. Uniform branch, taken only on the tile that crosses seq_len. */
        {
            rocke_value_t* tail_end = rocke_b_add(b, tile_off, rocke_b_const_i32(b, cfg->T));
            rocke_if_t tail = rocke_b_scf_if(b, rocke_b_cmp_gt(b, tail_end, ctx->seq_len));
            int v_vecs_per_row = cfg->HD / 8;
            int li;
            rocke_b_region_enter(b, tail.then_region);
            for(li = 0; li < (cfg->T * cfg->HD / 8) / cfg->THREADS; li++)
            {
                rocke_value_t* v_vid
                    = rocke_b_add(b, rocke_b_const_i32(b, li * cfg->THREADS), ctx->tid);
                rocke_value_t* v_row = rocke_b_div(b, v_vid, rocke_b_const_i32(b, v_vecs_per_row));
                rocke_value_t* v_mod = rocke_b_mod(b, v_vid, rocke_b_const_i32(b, v_vecs_per_row));
                rocke_value_t* v_col = rocke_b_mul(b, v_mod, rocke_b_const_i32(b, 8));
                rocke_value_t* key = rocke_b_add(b, tile_off, v_row);
                rocke_if_t dead = rocke_b_scf_if(b, rocke_b_cmp_ge(b, key, ctx->seq_len));
                rocke_value_t* v_idx[3];
                rocke_b_region_enter(b, dead.then_region);
                v_idx[0] = cur_buf;
                v_idx[1] = v_row;
                v_idx[2] = v_col;
                rocke_b_smem_store_vN(b, ctx->V_lds, v_idx, 3, ctx->z8, 8);
                rocke_b_region_leave(b);
            }
            rocke_b_sync_lds_only(b);
            rocke_b_region_leave(b);
        }

        if(TRANS)
        {
            /* P^T operands straight from registers: PV step k covers keys
             * k*PV_K_STEP + key_lane + (t*4 + r), the lane's V transpose-read rows. */
            p_ops = (rocke_value_t**)rocke_arena_alloc(
                &b->arena, (size_t)cfg->PV_K_ITERS * sizeof(rocke_value_t*));
            if(p_ops == NULL)
            {
                rocke_b_region_leave(b);
                return;
            }
            for(k = 0; k < cfg->PV_K_ITERS; k++)
            {
                rocke_value_t* comps[8];
                int nc = 0;
                if(cfg->PV_K_STEP == 32)
                {
                    int t;
                    for(t = 0; t < 2; t++)
                    {
                        for(r = 0; r < 4; r++)
                        {
                            comps[nc++] = rocke_b_cast_f32_to(b, p_t[(2 * k + t) * 4 + r], dtype);
                        }
                    }
                }
                else
                {
                    for(r = 0; r < 4; r++)
                    {
                        comps[nc++] = rocke_b_cast_f32_to(b, p_t[k * 4 + r], dtype);
                    }
                }
                p_ops[k] = rocke_b_vec_pack(b, comps, nc, dtype);
            }
        }

        /* ---------------- PV (wide 16x16x32 or narrow 16x16x16 via ds_read_tr) -- */
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            rocke_value_t* scaled_comps[4];
            rocke_value_t* acc_v;
            rocke_value_t* n_col_base;
            for(reg = 0; reg < 4; reg++)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, acc_vals[n], reg);
                scaled_comps[reg] = rocke_b_fmul(b, e, alpha_regs[TRANS ? 0 : reg]);
            }
            acc_v = rocke_b_vec_pack(b, scaled_comps, 4, f32);

            {
                rocke_value_t* ncb_n = rocke_b_const_i32(b, n);
                rocke_value_t* ncb_16 = rocke_b_const_i32(b, 16);
                n_col_base = rocke_b_add(b, rocke_b_mul(b, ncb_n, ncb_16), ctx->tr_col_lane);
            }

            for(k = 0; k < cfg->PV_K_ITERS; k++)
            {
                if(cfg->PV_K_STEP == 32)
                {
                    rocke_value_t* A_p = NULL;
                    rocke_value_t* row_r0;
                    rocke_value_t* row_r1;
                    rocke_value_t* B_r0;
                    rocke_value_t* B_r1;
                    rocke_value_t* B_v;
                    rocke_value_t* b0_idx[3];
                    rocke_value_t* b1_idx[3];
                    if(!TRANS)
                    {
                        rocke_value_t* p_off_c = rocke_b_const_i32(b, k * 32);
                        rocke_value_t* p_off_m
                            = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 8));
                        rocke_value_t* p_off = rocke_b_add(b, p_off_c, p_off_m);
                        rocke_value_t* p_idx[2];
                        p_idx[0] = ctx->lane_col;
                        p_idx[1] = p_off;
                        A_p = rocke_b_smem_load_vN(b, ctx->P_lds, p_idx, 2, dtype, 8);
                    }
                    row_r0 = rocke_bound_transpose_lds_reader_row(b, ctx->pv_tr_reader, k * 32, 0);
                    row_r1 = rocke_bound_transpose_lds_reader_row(b, ctx->pv_tr_reader, k * 32, 1);
                    b0_idx[0] = cur_buf;
                    b0_idx[1] = row_r0;
                    b0_idx[2] = n_col_base;
                    B_r0 = rocke_b_ds_read_tr16_b64(b, ctx->V_lds, b0_idx, 3, dtype);
                    b1_idx[0] = cur_buf;
                    b1_idx[1] = row_r1;
                    b1_idx[2] = n_col_base;
                    B_r1 = rocke_b_ds_read_tr16_b64(b, ctx->V_lds, b1_idx, 3, dtype);
                    B_v = rocke_b_vec_concat(b, B_r0, B_r1);
                    if(TRANS)
                    {
                        /* O^T += V^T P^T: the V read is the A operand. */
                        acc_v = rocke__mfma_16x16x32(b, dtype, B_v, p_ops[k], acc_v);
                    }
                    else
                    {
                        acc_v = rocke__mfma_16x16x32(b, dtype, A_p, B_v, acc_v);
                    }
                }
                else
                {
                    rocke_value_t* A_p = NULL;
                    rocke_value_t* row_lane;
                    rocke_value_t* B_v;
                    rocke_value_t* bv_idx[3];
                    if(!TRANS)
                    {
                        rocke_value_t* p_off_c = rocke_b_const_i32(b, k * 16);
                        rocke_value_t* p_off_m
                            = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 4));
                        rocke_value_t* p_off = rocke_b_add(b, p_off_c, p_off_m);
                        rocke_value_t* p_idx[2];
                        p_idx[0] = ctx->lane_col;
                        p_idx[1] = p_off;
                        A_p = rocke_b_smem_load_vN(b, ctx->P_lds, p_idx, 2, dtype, 4);
                    }
                    row_lane
                        = rocke_bound_transpose_lds_reader_row(b, ctx->pv_tr_reader, k * 16, 0);
                    bv_idx[0] = cur_buf;
                    bv_idx[1] = row_lane;
                    bv_idx[2] = n_col_base;
                    B_v = rocke_b_ds_read_tr16_b64(b, ctx->V_lds, bv_idx, 3, dtype);
                    if(TRANS)
                    {
                        acc_v = rocke__mfma_16x16x16(b, dtype, B_v, p_ops[k], acc_v);
                    }
                    else
                    {
                        acc_v = rocke__mfma_16x16x16(b, dtype, A_p, B_v, acc_v);
                    }
                }
            }
            new_acc[n] = acc_v;
        }

        /* ---------------- carry yield ---------------- */
        {
            rocke_value_t** yields = (rocke_value_t**)rocke_arena_alloc(
                &b->arena, (size_t)num_iter * sizeof(rocke_value_t*));
            int yi = 0;
            int i;
            if(yields == NULL)
            {
                rocke_b_region_leave(b);
                return;
            }
            for(r = 0; r < NR; r++)
            {
                yields[yi++] = m_new[r];
                yields[yi++] = new_l_vals[r];
            }
            for(n = 0; n < cfg->PV_N_TILES; n++)
            {
                yields[yi++] = new_acc[n];
            }
            yields[yi++] = nxt_buf;
            for(i = 0; i < n_new_pf; i++)
            {
                yields[yi++] = new_pf[i];
            }
            rocke_b_scf_yield(b, yields, yi);
        }
    }
    rocke_b_region_leave(b);

    /* ---------------- stash final (m,l,acc) into ctx ---------- */
    {
        rocke_value_t** final = kvloop.op->results;
        for(r = 0; r < NR; r++)
        {
            ctx->m_final[r] = final[2 * r];
            ctx->l_final[r] = final[2 * r + 1];
        }
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            ctx->acc_final[n] = final[2 * NR + n];
        }
    }
}

/* Transposed-softmax epilogue: lane holds O[row = lane % 16][d = n*16 +
 * (lane/16)*4 + 0..3], one 16-byte store per n-tile. Lane group 0 writes the
 * row's m and the row sum reduced over the 4 lane groups. */
static void rocke__emit_epilogue_trans(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx950_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    rocke_value_t* row_ok = ctx->hoist_row_ok[0];
    rocke_value_t* qtoken = rocke_b_add(b, ctx->cu_q_start, ctx->hoist_qp_r[0]);
    rocke_value_t* d_lane = rocke_b_mul(b, ctx->lane_rg, rocke_b_const_i32(b, 4));
    rocke_value_t* l_row;
    rocke_value_t* ml_idx = NULL;
    rocke_value_t* lane0;
    rocke_if_t ml_guard;
    int n;
    for(n = 0; n < cfg->PV_N_TILES; n++)
    {
        const char* names[4] = {"token", "head", "seg", "dim"};
        rocke_value_t* vals[4];
        rocke_value_t* seg_acc_idx = NULL;
        rocke_if_t guard;
        vals[0] = qtoken;
        vals[1] = ctx->hoist_qh_r[0];
        vals[2] = ctx->seg_idx;
        vals[3] = rocke_b_add(b, rocke_b_const_i32(b, n * 16), d_lane);
        rocke_transforms_descriptor_offset(
            b, ctx->seg_acc_desc, names, vals, 4, &seg_acc_idx, NULL);
        guard = rocke_b_scf_if(b, row_ok);
        rocke_b_region_enter(b, guard.then_region);
        rocke_b_global_store_vN(b, ctx->segm_output_ptr, seg_acc_idx, ctx->acc_final[n], 4, 16);
        rocke_b_region_leave(b);
    }
    l_row = rocke__group_reduce(b, ctx->l_final[0], true);
    {
        const char* names[3] = {"token", "head", "seg"};
        rocke_value_t* vals[3];
        vals[0] = qtoken;
        vals[1] = ctx->hoist_qh_r[0];
        vals[2] = ctx->seg_idx;
        rocke_transforms_descriptor_offset(b, ctx->ml_desc, names, vals, 3, &ml_idx, NULL);
    }
    lane0 = rocke_b_cmp_eq(b, ctx->lane_rg, rocke_b_const_i32(b, 0));
    ml_guard = rocke_b_scf_if(b, rocke_b_land(b, row_ok, lane0));
    rocke_b_region_enter(b, ml_guard.then_region);
    rocke_b_global_store(b, ctx->segm_max_ptr, ml_idx, ctx->m_final[0], 4);
    rocke_b_global_store(b, ctx->segm_expsum_ptr, ml_idx, l_row, 4);
    rocke_b_region_leave(b);
}

/* ============================================================ *
 * emit_epilogue (lines 900-948)
 * ============================================================ */
void rocke_gfx950_attention_tiled_3d_emit_epilogue(rocke_gfx950_attention_tiled_3d_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gfx950_attn_tiled_3d_config_t* cfg = &ctx->cfg;
    int n, reg;
    rocke_value_t* lane_writes_ml;

    if(cfg->FOLD_SCALE)
    {
        /* fp8 K/V dequantized without v_scale: apply it to the segment output. */
        for(n = 0; n < cfg->PV_N_TILES; n++)
        {
            rocke_value_t* comps[4];
            for(reg = 0; reg < 4; reg++)
            {
                rocke_value_t* e = rocke_b_vec_extract(b, ctx->acc_final[n], reg);
                comps[reg] = rocke_b_fmul(b, e, ctx->v_scale_p);
            }
            ctx->acc_final[n] = rocke_b_vec_pack(b, comps, 4, rocke_f32());
        }
    }
    if(cfg->TRANS)
    {
        rocke__emit_epilogue_trans(ctx);
        return;
    }

    /* ---- segm_output stores (lines 912-933) ---- */
    for(n = 0; n < cfg->PV_N_TILES; n++)
    {
        for(reg = 0; reg < 4; reg++)
        {
            rocke_value_t* row = rocke__c_row(ctx, reg);
            rocke_value_t* col;
            rocke_value_t* qp_r;
            rocke_value_t* qh_r;
            rocke_value_t* row_ok;
            rocke_value_t* qtoken;
            rocke_value_t* seg_acc_idx = NULL;
            rocke_value_t* v_acc;
            rocke_if_t guard;
            {
                rocke_value_t* col_n = rocke_b_const_i32(b, n);
                rocke_value_t* col_16 = rocke_b_const_i32(b, 16);
                rocke_value_t* col_mul = rocke_b_mul(b, col_n, col_16);
                col = rocke_b_add(b, col_mul, ctx->lane_col);
            }
            qp_r = rocke_b_add(
                b, ctx->qb_start_pos, rocke_b_div(b, row, rocke_b_const_i32(b, cfg->NQK)));
            /* Python evaluates b.add(b.mul(...), b.mod(...)) left-to-right (mul
             * before mod) and b.land(cmp_lt(qp_r), cmp_lt(qh_r)) likewise. C
             * argument evaluation order is unspecified; sequence with temporaries
             * so the value ids match Python exactly. */
            {
                rocke_value_t* qh_mul
                    = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, cfg->NQK));
                rocke_value_t* qh_mod = rocke_b_mod(b, row, rocke_b_const_i32(b, cfg->NQK));
                qh_r = rocke_b_add(b, qh_mul, qh_mod);
            }
            {
                rocke_value_t* ok_a = rocke_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
                rocke_value_t* ok_b = rocke_b_cmp_lt(b, qh_r, rocke_b_const_i32(b, cfg->NUM_QH));
                row_ok = rocke_b_land(b, ok_a, ok_b);
            }
            qtoken = rocke_b_add(b, ctx->cu_q_start, qp_r);
            {
                const char* names[4];
                rocke_value_t* vals[4];
                names[0] = "token";
                vals[0] = qtoken;
                names[1] = "head";
                vals[1] = qh_r;
                names[2] = "seg";
                vals[2] = ctx->seg_idx;
                names[3] = "dim";
                vals[3] = col;
                rocke_transforms_descriptor_offset(
                    b, ctx->seg_acc_desc, names, vals, 4, &seg_acc_idx, NULL);
            }
            v_acc = rocke_b_vec_extract(b, ctx->acc_final[n], reg);
            guard = rocke_b_scf_if(b, row_ok);
            rocke_b_region_enter(b, guard.then_region);
            rocke_b_global_store(b, ctx->segm_output_ptr, seg_acc_idx, v_acc, 4);
            rocke_b_region_leave(b);
        }
    }

    /* ---- segm_max / segm_expsum stores (lines 935-948) ---- */
    {
        rocke_value_t* lwm_mod = rocke_b_mod(b, ctx->tid, rocke_b_const_i32(b, 16));
        lane_writes_ml = rocke_b_cmp_eq(b, lwm_mod, rocke_b_const_i32(b, 0));
    }
    for(reg = 0; reg < 4; reg++)
    {
        rocke_value_t* row = rocke__c_row(ctx, reg);
        rocke_value_t* qp_r;
        rocke_value_t* qh_r;
        rocke_value_t* row_ok;
        rocke_value_t* qtoken;
        rocke_value_t* ml_idx = NULL;
        rocke_value_t* do_write;
        rocke_if_t guard;
        qp_r = rocke_b_add(
            b, ctx->qb_start_pos, rocke_b_div(b, row, rocke_b_const_i32(b, cfg->NQK)));
        /* Left-to-right sequencing to match Python op-creation order (see the
         * segm_output store loop above for the rationale). */
        {
            rocke_value_t* qh_mul
                = rocke_b_mul(b, ctx->kv_head_idx, rocke_b_const_i32(b, cfg->NQK));
            rocke_value_t* qh_mod = rocke_b_mod(b, row, rocke_b_const_i32(b, cfg->NQK));
            qh_r = rocke_b_add(b, qh_mul, qh_mod);
        }
        {
            rocke_value_t* ok_a = rocke_b_cmp_lt(b, qp_r, ctx->cur_batch_q_len);
            rocke_value_t* ok_b = rocke_b_cmp_lt(b, qh_r, rocke_b_const_i32(b, cfg->NUM_QH));
            row_ok = rocke_b_land(b, ok_a, ok_b);
        }
        qtoken = rocke_b_add(b, ctx->cu_q_start, qp_r);
        {
            const char* names[3];
            rocke_value_t* vals[3];
            names[0] = "token";
            vals[0] = qtoken;
            names[1] = "head";
            vals[1] = qh_r;
            names[2] = "seg";
            vals[2] = ctx->seg_idx;
            rocke_transforms_descriptor_offset(b, ctx->ml_desc, names, vals, 3, &ml_idx, NULL);
        }
        do_write = rocke_b_land(b, lane_writes_ml, row_ok);
        guard = rocke_b_scf_if(b, do_write);
        rocke_b_region_enter(b, guard.then_region);
        rocke_b_global_store(b, ctx->segm_max_ptr, ml_idx, ctx->m_final[reg], 4);
        rocke_b_global_store(b, ctx->segm_expsum_ptr, ml_idx, ctx->l_final[reg], 4);
        rocke_b_region_leave(b);
    }
}

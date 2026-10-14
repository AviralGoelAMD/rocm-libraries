// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gemm_build_compute.c -- one chunk of the C99 port of
 * build_universal_gemm (rocke/instances/common/gemm_universal.py).
 *
 * SCOPE (this TU):
 *   - emit_mfma_phase            (rocke_gemm_emit_mfma_phase)
 *   - emit_compute_and_epilogue  (rocke_gemm_emit_compute_and_epilogue)
 *   - _emit_kloop_db             (rocke_gemm_emit_kloop_db)
 *   - _emit_kloop_simple         (rocke_gemm_emit_kloop_simple)
 *   - _emit_kloop_prefetch       (rocke_gemm_emit_kloop_prefetch)
 *
 * These map to the Python closures of the same names (gemm_universal.py
 * ~1466-1782). Every other closure/helper is a peer TU reached via
 * rocke/instance_gemm_internal.h. The emission order + builder-call sequence is a
 * faithful 1:1 port so the lowered op stream is byte-identical to Python.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "rocke/instance_gemm_internal.h"

/* ===================================================================== *
 *  emit_mfma_phase
 *
 *  One K-tile worth of MMAs across all per-warp atom positions and every K
 *  atom step inside this K-tile. On WMMA (RDNA) this delegates to the fully
 *  contract-driven _emit_wmma_phase. On MFMA (CDNA) it keeps the byte-identical
 *  arch-formula fragment load; the matmul + accumulator length are
 *  contract-driven (rocke_gemm_emit_mma / op->c_frag_len).
 *
 *  lds_parity is split (per the internal-header convention) into a compile-time
 *  int (parity_imm) OR a runtime Value (parity_v != NULL == Python isinstance
 *  Value). Only the prefetch/db paths pass a non-zero parity.
 * ===================================================================== */
void rocke_gemm_emit_mfma_phase(rocke_gemm_build_ctx_t* ctx,
                                rocke_value_t* A_src,
                                rocke_value_t* B_src,
                                rocke_value_t* const* iter_vars,
                                int num_iter_vars,
                                int parity_imm,
                                rocke_value_t* parity_v,
                                rocke_value_t** out_accs)
{
    rocke_ir_builder_t* b = ctx->b;
    const rocke_gemm_tile_spec_t* t = &ctx->spec->tile;

    int mfmas_m = ctx->mfmas_m;
    int mfmas_n = ctx->mfmas_n;
    int k_atoms = ctx->k_atoms;

    /* parity carry: a runtime Value form (a_par_row_v / b_par_row_v) when the
     * parity is itself a Value; a compile-time additive offset otherwise. */
    rocke_value_t* a_par_row_v = NULL;
    rocke_value_t* b_par_row_v = NULL;
    int a_row_parity_off = 0;
    int b_row_parity_off = 0;

    rocke_value_t* m_in_atom;
    rocke_value_t* k_blk;
    rocke_value_t* n_in_atom;
    rocke_value_t* warp_m_off;
    rocke_value_t* warp_n_off;
    rocke_value_t* k_blk_kbase;

    int i;
    int kk;

    if(ctx->is_wmma)
    {
        rocke_gemm_emit_wmma_phase(ctx, A_src, B_src, iter_vars, num_iter_vars, out_accs);
        return;
    }

    if(ctx->prefetch || ctx->db)
    {
        if(parity_v != NULL)
        {
            a_par_row_v = rocke_b_mul(b, parity_v, rocke_b_const_i32(b, ctx->block_m));
            b_par_row_v = rocke_b_mul(b, parity_v, rocke_b_const_i32(b, ctx->block_n));
            a_row_parity_off = 0; /* carried as a Value addition below */
            b_row_parity_off = 0;
        }
        else
        {
            a_par_row_v = NULL;
            b_par_row_v = NULL;
            a_row_parity_off = parity_imm * ctx->block_m;
            b_row_parity_off = parity_imm * ctx->block_n;
        }
    }
    else
    {
        a_par_row_v = NULL;
        b_par_row_v = NULL;
        a_row_parity_off = 0;
        b_row_parity_off = 0;
    }
    /* dtl_lds_alias: the ring slot (ping-pong half) every operand read of this
     * K-tile touches. */
    rocke_lds_alias_t a_tag;
    rocke_lds_alias_t b_tag;
    const rocke_lds_alias_t* a_alias
        = rocke_gemm_ring_slot(ctx, ctx->a_lds_ring, parity_imm, parity_v, &a_tag);
    const rocke_lds_alias_t* b_alias
        = rocke_gemm_ring_slot(ctx, ctx->b_lds_ring, parity_imm, parity_v, &b_tag);

    /* Lane mapping into LDS. M-in-atom = lane % warp_tile_m, K-sub-block =
     * lane / warp_tile_m, N-in-atom = lane % warp_tile_n. */
    m_in_atom = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_m));
    k_blk = rocke_b_div(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_m));
    n_in_atom = rocke_b_mod(b, ctx->lane, rocke_b_const_i32(b, t->warp_tile_n));

    warp_m_off = rocke_b_mul(b, ctx->warp_m_idx, rocke_b_const_i32(b, mfmas_m * t->warp_tile_m));
    warp_n_off = rocke_b_mul(b, ctx->warp_n_idx, rocke_b_const_i32(b, mfmas_n * t->warp_tile_n));

    /* Lane-local K base inside the current K-tile, hoisted once. */
    k_blk_kbase = rocke_b_mul(b, k_blk, rocke_b_const_i32(b, ctx->a_per_lane));

    /* new_accs = list(iter_vars) */
    for(i = 0; i < num_iter_vars; ++i)
        out_accs[i] = iter_vars[i];

    for(kk = 0; kk < k_atoms; ++kk)
    {
        /* a_rows[mi] = _read_a(kk, mi); b_cols[ni] = _read_b(kk, ni). */
        rocke_value_t* a_rows[ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* b_cols[ROCKE_GEMM_MAX_ACCS];
        int mi;
        int ni;

        for(mi = 0; mi < mfmas_m; ++mi)
        {
            /* _read_a(kk, mi) */
            rocke_value_t* col_base
                = rocke_b_add(b, k_blk_kbase, rocke_b_const_i32(b, kk * t->warp_tile_k));
            rocke_value_t* a_row = rocke_b_add(
                b,
                warp_m_off,
                rocke_b_add(
                    b, rocke_b_const_i32(b, mi * t->warp_tile_m + a_row_parity_off), m_in_atom));
            if(a_par_row_v != NULL)
                a_row = rocke_b_add(b, a_row, a_par_row_v);
            a_rows[mi] = rocke_gemm_emit_smem_load_alias(b,
                                                         A_src,
                                                         a_row,
                                                         rocke_gemm_swz_col(ctx, col_base, a_row),
                                                         ctx->a_per_lane,
                                                         ctx->storage_dtype,
                                                         a_alias);
        }

        for(ni = 0; ni < mfmas_n; ++ni)
        {
            /* _read_b(kk, ni) */
            rocke_value_t* col_base
                = rocke_b_add(b, k_blk_kbase, rocke_b_const_i32(b, kk * t->warp_tile_k));
            rocke_value_t* b_row = rocke_b_add(
                b,
                warp_n_off,
                rocke_b_add(
                    b, rocke_b_const_i32(b, ni * t->warp_tile_n + b_row_parity_off), n_in_atom));
            if(b_par_row_v != NULL)
                b_row = rocke_b_add(b, b_row, b_par_row_v);
            b_cols[ni] = rocke_gemm_emit_smem_load_alias(b,
                                                         B_src,
                                                         b_row,
                                                         rocke_gemm_swz_col(ctx, col_base, b_row),
                                                         ctx->b_per_lane,
                                                         ctx->storage_dtype,
                                                         b_alias);
        }

        /* _mma_cluster(a_rows, b_cols): bracket each m-row's MFMA cluster
         * with s_setprio(1)/(0) + sched_barrier(0) only when both the
         * swizzle (reference-faithful mode) AND the schedule-hint policy are
         * on. _sched_hints defaults OFF on gfx950 (take the measured uplift). */
        for(mi = 0; mi < mfmas_m; ++mi)
        {
            if(ctx->swz && ctx->sched_hints)
                rocke_b_s_setprio(b, 1);
            for(ni = 0; ni < mfmas_n; ++ni)
            {
                int flat = mi * mfmas_n + ni;
                out_accs[flat]
                    = rocke_gemm_emit_mma(b, ctx->op, a_rows[mi], b_cols[ni], out_accs[flat]);
            }
            if(ctx->swz && ctx->sched_hints)
            {
                rocke_b_s_setprio(b, 0);
                rocke_b_sched_barrier(b, 0);
            }
        }
    }

    /* Quantitative two-stage HotLoop schedule, issued ONCE per K-tile.
     * Suppressed when _sched_hints is off (gfx950 default). */
    if(ctx->sched_hints && ctx->spec->trait.pipeline
       && (strcmp(ctx->spec->trait.pipeline, "compv3") == 0
           || strcmp(ctx->spec->trait.pipeline, "compv4") == 0))
    {
        rocke_gemm_emit_hotloop_schedule(b, ctx->spec, ctx->load_vec);
    }
}

/* ===================================================================== *
 *  emit_compute_and_epilogue: prefetch/db/simple K-loop selector + epilogue.
 * ===================================================================== */
void rocke_gemm_emit_compute_and_epilogue(rocke_gemm_build_ctx_t* ctx)
{
    if(ctx->prefetch)
        rocke_gemm_emit_kloop_prefetch(ctx);
    else if(ctx->db)
        rocke_gemm_emit_kloop_db(ctx);
    else
        rocke_gemm_emit_kloop_simple(ctx);
    rocke_gemm_emit_epilogue(ctx);
}

/* ===================================================================== *
 *  _emit_kloop_db: compv4 double-buffered K-loop (non-DTL, VGPR-staged).
 * ===================================================================== */
void rocke_gemm_emit_kloop_db(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c1_i32;
    rocke_value_t* K_minus_one_tile;
    rocke_iter_arg_t loop_args[1 + ROCKE_GEMM_MAX_ACCS];
    int num_loop_args;
    rocke_for_t for_op;
    int i;

    c1_i32 = rocke_b_const_i32(b, 1);
    /* K-loop upper bound is the slice end (k_upper == K when not split-K). */
    K_minus_one_tile = rocke_b_sub(b, ctx->k_upper, ctx->c_block_k);

    /* Prologue: issue tile 0's load into half 0 (slice base k_lo in split-K). */
    rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, ctx->k_lo, 0, NULL);

    /* loop_args = [("par", c0)] + list(accs) */
    loop_args[0].name = "par";
    loop_args[0].init = ctx->c0;
    for(i = 0; i < ctx->num_accs; ++i)
    {
        loop_args[1 + i].name = ctx->acc_names[i];
        loop_args[1 + i].init = ctx->acc_inits[i];
    }
    num_loop_args = 1 + ctx->num_accs;

    for_op = rocke_b_scf_for_iter(b,
                                  ctx->k_lo,
                                  K_minus_one_tile,
                                  ctx->c_block_k,
                                  loop_args,
                                  num_loop_args,
                                  "k0",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* parity = for_op.iter_vars[0];
        rocke_value_t* const* acc_iter = &for_op.iter_vars[1];
        int num_acc_iter = for_op.num_iter_vars - 1;
        rocke_value_t* next_parity = rocke_b_sub(b, c1_i32, parity);
        rocke_value_t* k_next = rocke_b_add(b, for_op.iv, ctx->c_block_k);
        rocke_value_t* new_accs[ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* yield_vals[1 + ROCKE_GEMM_MAX_ACCS];

        /* Start-of-iter barrier (RAW-safe read of half(parity), WAR-safe
         * overwrite of half(next)). */
        rocke_b_sync(b);
        /* Issue next-tile global->VGPR->LDS copy into the other half. */
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, k_next, 0, next_parity);
        rocke_gemm_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, acc_iter, num_acc_iter, 0, parity, new_accs);

        yield_vals[0] = next_parity;
        for(i = 0; i < num_acc_iter; ++i)
            yield_vals[1 + i] = new_accs[i];
        rocke_b_scf_yield(b, yield_vals, 1 + num_acc_iter);
    }
    rocke_b_region_leave(b);

    /* Epilogue: drain the last in-loop load, barrier, MFMA that tile. */
    {
        rocke_value_t* final_parity = for_op.op->results[0];
        rocke_value_t* const* tail_accs = &for_op.op->results[1];
        int num_tail = for_op.op->num_results - 1;
        rocke_value_t* epi_accs[ROCKE_GEMM_MAX_ACCS];

        rocke_b_sync(b);
        rocke_gemm_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, tail_accs, num_tail, 0, final_parity, epi_accs);

        for(i = 0; i < num_tail; ++i)
            ctx->for_results[i] = epi_accs[i];
        ctx->num_for_results = num_tail;
    }
}

/* ===================================================================== *
 *  _emit_kloop_simple: single-buffer load-then-compute K-loop.
 * ===================================================================== */
void rocke_gemm_emit_kloop_simple(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_iter_arg_t loop_args[ROCKE_GEMM_MAX_ACCS];
    rocke_for_t for_op;
    int i;

    for(i = 0; i < ctx->num_accs; ++i)
    {
        loop_args[i].name = ctx->acc_names[i];
        loop_args[i].init = ctx->acc_inits[i];
    }

    for_op = rocke_b_scf_for_iter(b,
                                  ctx->k_lo,
                                  ctx->k_upper,
                                  ctx->c_block_k,
                                  loop_args,
                                  ctx->num_accs,
                                  "k0",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* new_accs[ROCKE_GEMM_MAX_ACCS];

        /* Single-buffer load-then-compute pipeline. parity is compile-time 0
         * (Python passes no lds_parity -> default 0). */
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, for_op.iv, 0, NULL);
        rocke_b_sync(b);

        rocke_gemm_emit_mfma_phase(ctx,
                                   ctx->A_smem,
                                   ctx->B_smem,
                                   for_op.iter_vars,
                                   for_op.num_iter_vars,
                                   0,
                                   NULL,
                                   new_accs);

        rocke_b_sync(b);
        rocke_b_scf_yield(b, new_accs, for_op.num_iter_vars);
    }
    rocke_b_region_leave(b);

    for(i = 0; i < for_op.op->num_results; ++i)
        ctx->for_results[i] = for_op.op->results[i];
    ctx->num_for_results = for_op.op->num_results;
}

/* ===================================================================== *
 *  _emit_kloop_prefetch_ring: _emit_kloop_prefetch steady state + tail for
 *  dtl_lds_alias. The ping-pong loop is unrolled 2x with static parity so
 *  each copy's A/B ring slot tag is a compile-time fact: a pair iteration
 *  runs tile 2j from slot 0 (DMA fills slot 1) and tile 2j+1 from slot 1
 *  (DMA fills slot 0). A 0/1-trip remainder loop runs an odd last
 *  steady-state tile from slot 0 and carries the final tile's slot, which
 *  the tail reads through any_slot after its full drain. The next DMA is
 *  issued after each barrier, so no LDS-DMA op is in flight at a barrier
 *  (keep_lds_dma=0).
 * ===================================================================== */
static void gemm_emit_kloop_prefetch_ring(rocke_gemm_build_ctx_t* ctx,
                                          rocke_value_t* K_minus_one_tile)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* c_pair = rocke_b_const_i32(b, 2 * ctx->block_k);
    rocke_value_t* trips;
    rocke_value_t* pair_end;
    rocke_iter_arg_t pair_args[ROCKE_GEMM_MAX_ACCS];
    rocke_iter_arg_t rem_args[1 + ROCKE_GEMM_MAX_ACCS];
    char rem_names[ROCKE_GEMM_MAX_ACCS][96];
    rocke_for_t pair_op;
    rocke_for_t rem_op;
    int i;

    {
        rocke_value_t* span = rocke_b_sub(b, K_minus_one_tile, ctx->k_lo);
        rocke_value_t* round = rocke_b_const_i32(b, ctx->block_k - 1);
        trips = rocke_b_div(b, rocke_b_add(b, span, round), ctx->c_block_k);
    }
    {
        rocke_value_t* half = rocke_b_div(b, trips, rocke_b_const_i32(b, 2));
        pair_end = rocke_b_add(b, ctx->k_lo, rocke_b_mul(b, half, c_pair));
    }

    for(i = 0; i < ctx->num_accs; ++i)
    {
        pair_args[i].name = ctx->acc_names[i];
        pair_args[i].init = ctx->acc_inits[i];
    }
    pair_op = rocke_b_scf_for_iter(b,
                                   ctx->k_lo,
                                   pair_end,
                                   c_pair,
                                   pair_args,
                                   ctx->num_accs,
                                   "k0",
                                   /*unroll=*/false,
                                   /*elide_trailing_barrier=*/true);
    rocke_b_region_enter(b, pair_op.body);
    {
        rocke_value_t* acc_a[ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* acc_b[ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* const* cur = pair_op.iter_vars;
        int n = pair_op.num_iter_vars;
        int s;
        for(s = 0; s < 2; ++s)
        {
            rocke_value_t* k_next;
            rocke_value_t** out = (s == 0) ? acc_a : acc_b;
            rocke_b_sync_keep_lds_dma(b, 0);
            k_next = rocke_b_add(b, pair_op.iv, s == 0 ? ctx->c_block_k : c_pair);
            rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, k_next, 1 - s, NULL);
            rocke_gemm_emit_mfma_phase(ctx, ctx->A_smem, ctx->B_smem, cur, n, s, NULL, out);
            cur = out;
        }
        rocke_b_scf_yield(b, acc_b, n);
    }
    rocke_b_region_leave(b);

    rem_args[0].name = "rpar";
    rem_args[0].init = ctx->c0;
    for(i = 0; i < ctx->num_accs; ++i)
    {
        snprintf(rem_names[i], sizeof(rem_names[i]), "r_%s", ctx->acc_names[i]);
        rem_args[1 + i].name = rem_names[i];
        rem_args[1 + i].init = pair_op.op->results[i];
    }
    rem_op = rocke_b_scf_for_iter(b,
                                  pair_end,
                                  K_minus_one_tile,
                                  ctx->c_block_k,
                                  rem_args,
                                  1 + ctx->num_accs,
                                  "k0r",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);
    rocke_b_region_enter(b, rem_op.body);
    {
        rocke_value_t* new_accs[ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* yield_vals[1 + ROCKE_GEMM_MAX_ACCS];
        int n = rem_op.num_iter_vars - 1;
        rocke_b_sync_keep_lds_dma(b, 0);
        rocke_gemm_emit_load_phase(ctx,
                                   ctx->A_smem,
                                   ctx->B_smem,
                                   rocke_b_add(b, rem_op.iv, ctx->c_block_k),
                                   1,
                                   NULL);
        rocke_gemm_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, &rem_op.iter_vars[1], n, 0, NULL, new_accs);
        yield_vals[0] = rocke_b_const_i32(b, 1);
        for(i = 0; i < n; ++i)
            yield_vals[1 + i] = new_accs[i];
        rocke_b_scf_yield(b, yield_vals, 1 + n);
    }
    rocke_b_region_leave(b);

    /* Tail: the last tile sits in slot rem_op.results[0] (run-time). */
    {
        rocke_value_t* epi_accs[ROCKE_GEMM_MAX_ACCS];
        int num_tail = rem_op.op->num_results - 1;
        rocke_b_sync_keep_lds_dma(b, 0);
        rocke_gemm_emit_mfma_phase(ctx,
                                   ctx->A_smem,
                                   ctx->B_smem,
                                   &rem_op.op->results[1],
                                   num_tail,
                                   0,
                                   rem_op.op->results[0],
                                   epi_accs);
        for(i = 0; i < num_tail; ++i)
            ctx->for_results[i] = epi_accs[i];
        ctx->num_for_results = num_tail;
    }
}

/* ===================================================================== *
 *  _emit_kloop_prefetch: DTLA ping-pong software-pipelined K-loop. Falls
 *  back to _simple when loads_per_tile > 63 (vmcnt is 6 bits on gfx950).
 * ===================================================================== */
void rocke_gemm_emit_kloop_prefetch(rocke_gemm_build_ctx_t* ctx)
{
    rocke_ir_builder_t* b = ctx->b;
    int loads_per_tile;
    rocke_value_t* K_minus_one_tile;
    rocke_value_t* c1_i32;
    rocke_iter_arg_t loop_args[1 + ROCKE_GEMM_MAX_ACCS];
    int num_loop_args;
    rocke_for_t for_op;
    int i;

    loads_per_tile = ctx->dtl_a_passes + ctx->dtl_b_passes;
    if(loads_per_tile > 63)
    {
        rocke_gemm_emit_kloop_simple(ctx);
        return;
    }

    /* Prologue: load tile 0 into half 0 (slice base k_lo in split-K). */
    rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, ctx->k_lo, 0, NULL);

    K_minus_one_tile = rocke_b_sub(b, ctx->k_upper, ctx->c_block_k);
    if(ctx->lds_alias)
    {
        gemm_emit_kloop_prefetch_ring(ctx, K_minus_one_tile);
        return;
    }
    c1_i32 = rocke_b_const_i32(b, 1);

    loop_args[0].name = "par";
    loop_args[0].init = ctx->c0;
    for(i = 0; i < ctx->num_accs; ++i)
    {
        loop_args[1 + i].name = ctx->acc_names[i];
        loop_args[1 + i].init = ctx->acc_inits[i];
    }
    num_loop_args = 1 + ctx->num_accs;

    for_op = rocke_b_scf_for_iter(b,
                                  ctx->k_lo,
                                  K_minus_one_tile,
                                  ctx->c_block_k,
                                  loop_args,
                                  num_loop_args,
                                  "k0",
                                  /*unroll=*/false,
                                  /*elide_trailing_barrier=*/true);

    rocke_b_region_enter(b, for_op.body);
    {
        rocke_value_t* parity = for_op.iter_vars[0];
        rocke_value_t* const* acc_iter = &for_op.iter_vars[1];
        int num_acc_iter = for_op.num_iter_vars - 1;
        rocke_value_t* next_parity = rocke_b_sub(b, c1_i32, parity);
        rocke_value_t* k_next = rocke_b_add(b, for_op.iv, ctx->c_block_k);
        rocke_value_t* new_accs[ROCKE_GEMM_MAX_ACCS];
        rocke_value_t* yield_vals[1 + ROCKE_GEMM_MAX_ACCS];

        /* Single-barrier software pipeline: ONE s_waitcnt + ONE WG barrier. */
        rocke_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
        rocke_b_s_barrier_bare(b);
        rocke_gemm_emit_load_phase(ctx, ctx->A_smem, ctx->B_smem, k_next, 0, next_parity);
        rocke_gemm_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, acc_iter, num_acc_iter, 0, parity, new_accs);

        yield_vals[0] = next_parity;
        for(i = 0; i < num_acc_iter; ++i)
            yield_vals[1 + i] = new_accs[i];
        rocke_b_scf_yield(b, yield_vals, 1 + num_acc_iter);
    }
    rocke_b_region_leave(b);

    /* Epilogue: drain the final tile's loads, rendezvous, MFMA last tile. */
    {
        rocke_value_t* final_parity = for_op.op->results[0];
        rocke_value_t* const* tail_accs = &for_op.op->results[1];
        int num_tail = for_op.op->num_results - 1;
        rocke_value_t* epi_accs[ROCKE_GEMM_MAX_ACCS];

        rocke_b_s_waitcnt(b, /*vmcnt=*/0, /*lgkmcnt=*/0, /*expcnt=*/-1);
        rocke_b_s_barrier_bare(b);
        rocke_gemm_emit_mfma_phase(
            ctx, ctx->A_smem, ctx->B_smem, tail_accs, num_tail, 0, final_parity, epi_accs);

        for(i = 0; i < num_tail; ++i)
            ctx->for_results[i] = epi_accs[i];
        ctx->num_for_results = num_tail;
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/driver.cuh
// =============================================================================
//
// run_nr_loop<Policy>  —  templated Newton-Raphson iteration driver
//
// Executes nb_iter fixed NR iterations.  The Policy type controls two
// compile-time decisions:
//
//   Policy::needs_fresh_jacobian  — fill_J_kernel called every iteration
//   Policy::needs_iter0_jacobian  — fill_J_kernel called at iter==0 only
//
// and one runtime decision delegated to the policy:
//
//   Policy::solve_iter(CudssBatchSolver&, ...)
//     — decides when to call set_values / factor / refactor / solve on the
//       solver (which owns the cuDSS context and descriptors).
//
// All physics kernels (fill_FP, fill_FQ, fill_J, update_Va, update_Vm) live in
// acpf_nr_kernels.{cuh,cu} and are called by the driver — never by policies.
// Adding a new policy never requires touching kernel code.
//
// include-from: CUDA .cu files only (kernel launch syntax requires NVCC).
// =============================================================================

#ifndef CONTINGENCY_DRIVER_CUH
#define CONTINGENCY_DRIVER_CUH

#include "../nr_iter_step.cuh"    // NrIterBuffers, NrIterTimings, BS
#include "../acpf_nr_kernels.cuh" // fill_FP_kernel, fill_FQ_kernel, fill_J_kernel,
                                  // update_Va_kernel, update_Vm_kernel
#include "../cuda_utils.h"        // CuSpMV, CudaTimer
#include "../timing_utils.hpp"    // BatchTimings
#include "strategies/cudss_batch_solver.cuh"

// =============================================================================
// run_nr_loop<Policy>
//
// Executes nb_iter fixed Newton-Raphson iterations on the current chunk buffers.
// Accumulates per-phase timings into t (caller-owned BatchTimings).
//
// Parameters
// ----------
//   policy        — NR policy instance; begin_chunk() and solve_iter() called here
//   linear_solver — cuDSS batch solver (owns A/x/b); passed into policy.solve_iter()
//   spmv          — cuSPARSE block-diagonal SpMV handle for this chunk
//   buf           — non-owning raw pointer bundle (d_V, d_F, d_dx, d_J, …)
//   n_bus         — number of buses (per system)
//   n_pvpq        — |PV| + |PQ| buses
//   n_pq          — |PQ| buses
//   dim_J         — Jacobian dimension = n_pvpq + n_pq
//   nnz_Y         — non-zeros in Ybus (per system)
//   nnz_J         — non-zeros in J   (per system)
//   batch_size    — number of systems per chunk (ALWAYS the full batch, including
//                   phantom slots padded to batch_size in the last chunk)
//   nb_iter       — number of NR iterations (fixed; no convergence check)
//   cs            — CUDA stream (all kernels launched on this stream)
//   timer         — stream-bound CudaTimer (reused across phases)
//   t             — BatchTimings accumulator (updated in-place)
// =============================================================================
template <typename Policy>
inline void run_nr_loop(
    Policy&              policy,
    CudssBatchSolver&    linear_solver,
    CuSpMV&              spmv,
    const NrIterBuffers& buf,
    int                  n_bus,
    int                  n_pvpq,
    int                  n_pq,
    int                  dim_J,
    int                  nnz_Y,
    int                  nnz_J,
    int                  batch_size,
    int                  nb_iter,
    cudaStream_t         cs,
    CudaTimer&           timer,
    BatchTimings&  t)
{
    policy.begin_chunk();

    for (int iter = 0; iter < nb_iter; ++iter) {
        NrIterTimings step;

        // ①  SpMV: d_Ibus_batch = Ybus_batch · d_V_batch
        timer.start();
        spmv.spmv();
        step.t_spmv = timer.stop_ms();

        // ②  Fill F: −[ΔP, ΔQ] scattered into the ledger P/Q rows
        timer.start();
        fill_FP_kernel<<<(batch_size * buf.n_p + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_F, buf.d_V, buf.d_Ibus, buf.d_Sbus, buf.d_p_buses, buf.d_p_rows,
            buf.n_p, n_bus, dim_J, batch_size, buf.sbus_stride);
        fill_FQ_kernel<<<(batch_size * buf.n_q + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_F, buf.d_V, buf.d_Ibus, buf.d_Sbus, buf.d_q_buses, buf.d_q_rows,
            buf.n_q, n_bus, dim_J, batch_size, buf.sbus_stride);
        step.t_fill_F = timer.stop_ms();

        // ③  Fill J (policy-conditional via if constexpr)
        //
        //   needs_fresh_jacobian = true  → fill every iteration
        //   needs_iter0_jacobian = true  → fill only at iter == 0
        //   both false                   → never fill (policy reuses base factors)
        if constexpr (Policy::needs_fresh_jacobian) {
            timer.start();
            fill_J_kernel<<<(batch_size * nnz_Y + BS - 1) / BS, BS, 0, cs>>>(
                buf.d_J_values, buf.d_V, buf.d_Ibus,
                buf.d_Ybus_outer, buf.d_Ybus_inner, buf.d_Ybus_values,
                buf.d_map_j11, buf.d_map_j12, buf.d_map_j21, buf.d_map_j22,
                n_bus, nnz_Y, nnz_J, batch_size);
            step.t_fill_J = timer.stop_ms();
        } else if constexpr (Policy::needs_iter0_jacobian) {
            if (iter == 0) {
                timer.start();
                fill_J_kernel<<<(batch_size * nnz_Y + BS - 1) / BS, BS, 0, cs>>>(
                    buf.d_J_values, buf.d_V, buf.d_Ibus,
                    buf.d_Ybus_outer, buf.d_Ybus_inner, buf.d_Ybus_values,
                    buf.d_map_j11, buf.d_map_j12, buf.d_map_j21, buf.d_map_j22,
                    n_bus, nnz_Y, nnz_J, batch_size);
                step.t_fill_J = timer.stop_ms();
            }
        }

        // ④  Linear solve (delegated to policy, which operates on linear_solver)
        //
        //   The policy is responsible for:
        //     • deciding whether to set_values / factor / refactor / solve
        //     • SOLVE: d_dx_batch = J^{-1} * d_F_batch
        //
        //   Returns true if REFACTORIZATION was done (false = first FACTORIZATION).
        const bool did_refactor = policy.solve_iter(
            linear_solver, buf.d_J_values, cs, timer, step);

        // ⑤  Update V: Va (pvpq) then Vm (pq) — both on cs, CUDA stream ordering
        //     serialises them without an explicit CPU event.
        timer.start();
        update_Va_kernel<<<(batch_size * buf.n_theta + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_V, buf.d_dx, buf.d_theta_buses, buf.d_theta_cols,
            buf.n_theta, n_bus, dim_J, batch_size);
        update_Vm_kernel<<<(batch_size * buf.n_vm + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_V, buf.d_dx, buf.d_vm_buses, buf.d_vm_cols,
            buf.n_vm, n_bus, dim_J, batch_size);
        step.t_update_V = timer.stop_ms();

        // Accumulate per-iteration timings into per-chunk totals.
        t.t_spmv            += step.t_spmv;
        t.t_fill_F          += step.t_fill_F;
        t.t_fill_J          += step.t_fill_J;
        t.t_first_factorize += step.t_first_factorize;
        t.t_refactorize     += step.t_refactorize;
        if (did_refactor) t.n_refactorize++;
        t.t_solve           += step.t_solve;
        t.t_update_V        += step.t_update_V;
    }
}

#endif // CONTINGENCY_DRIVER_CUH
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GPUSIM2GRID_WARMUP_HPP
#define GPUSIM2GRID_WARMUP_HPP

// ---------------------------------------------------------------------------
// warmup()
//
// Pays every one-time, size-independent CUDA cost up front so it lands
// nowhere near a measured region: CUDA-context creation on the target device,
// plus the dlopen + PTX-JIT of the cuSPARSE and cuDSS backends that the first
// real solve would otherwise trigger.
//
// Without this, the first AcPfGPU / ContingencyAnalysisGPU / InjectionSweepGPU
// built in a process absorbs tens of milliseconds of library initialization
// (reported separately as t_context_init_ms, but still inside whatever
// stopwatch the caller wrapped around construction). In a benchmark loop over
// several grids, that entire cost is charged to whichever grid happens to go
// first.
//
// The work mirrors what a real solve does, in the same order, on a trivial
// 2x2 system: a cuSPARSE complex CSR SpMV, then a cuDSS
// analyze -> factorize -> solve, both single-system and uniform-batch (the
// batch path configures CUDSS_CONFIG_UBATCH_SIZE and can load different
// kernels). Nothing is retained; only the driver/library-side initialization
// persists, which is the point.
//
// device: CUDA device ordinal to warm up, or -1 for the current device.
// Returns the wall-clock milliseconds spent. Idempotent (a second call costs
// ~nothing, which is exactly the state it establishes).
// ---------------------------------------------------------------------------
double warmup(int device = -1);

#endif  // GPUSIM2GRID_WARMUP_HPP

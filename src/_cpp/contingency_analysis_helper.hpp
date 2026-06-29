// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CONTINGENCY_ANALYSIS_HELPER_HPP
#define CONTINGENCY_ANALYSIS_HELPER_HPP

// =============================================================================
// contingency_analysis_helper.hpp
//
// Pure C++ (no CUDA) preprocessing for the contingency analysis solver.
//
// Responsibilities
// ----------------
//   • Define the Triplet / Contingency / ChunkPatchRange structs used across
//     the CPU preprocessing and GPU kernel launch sides.
//   • csr_find_k()      — binary search for a CSR flat index given (row, col).
//   • resolve_indices() — fill t.k for every triplet via csr_find_k().
//   • build_flat_patches() — convert per-contingency triplet lists into three
//     parallel host arrays (SoA) ready for device upload, and precompute the
//     slice [start, count) within those arrays that belongs to each chunk.
//   • build_blockdiag_csr() — build the outer (row pointer) and inner (column
//     index) arrays for the block-diagonal Ybus used in batched cuSPARSE SpMV.
//     Only the structure is built here; values are tiled per chunk at runtime.
//
// Precision note
// --------------
// delta_re / delta_im in the flat patch arrays are cuda_real_type (FP32 when
// GPUSIM2GRID_REAL_FLOAT is defined, FP64 otherwise) so they match the
// device-side Ybus values exactly with no implicit widening/narrowing.
// The raw Python / Eigen input (always double) is cast during build_flat_patches.
// =============================================================================

#include "dtypes.hpp"   // cuda_real_type, eigen_real_type, eigen_cplx_type

#include "Eigen/SparseCore"

#include <vector>
#include <tuple>

// ---------------------------------------------------------------------------
// SolverType
//   Selects the linear-solve strategy used by ContingencyAnalysisSolver.
//
//   DirectRefactorEvery   — fill J every iteration; FACTORIZE once globally,
//                           REFACTORIZE for all subsequent calls.  Highest
//                           accuracy; most factorize/refactorize calls.
//   DirectBaseCaseFactors — reuse base-case LU factors for all contingencies
//                           (no per-contingency factorize).  Cheapest per
//                           iteration; approximate (uses base-case J).
//   DirectIter0Only       — fill J at iter==0 per chunk; FACTORIZE/REFACTORIZE
//                           once per chunk; SOLVE-only for iter 1..nb_iter-1.
//                           Compromise between accuracy and cost.
// ---------------------------------------------------------------------------
enum class ContingencySolverType {
    DirectRefactorEvery,    // phase 1 (default)
    DirectBaseCaseFactors,  // phase 2
    DirectIter0Only,        // phase 3
    DirectRefactorEveryN    // refactor every N iterations (N set via refactor_period)
};

// ---------------------------------------------------------------------------
// Triplet
//   One (row, col, complex delta) entry of a Ybus modification.
//   k is the flat CSR index into the RowMajor Ybus values array;
//   it is filled in by resolve_indices() after construction and must not be
//   used before that call.
// ---------------------------------------------------------------------------
struct Triplet {
    int    row;
    int    col;
    double delta_re;   // real-part delta  (always double on the host side)
    double delta_im;   // imag-part delta
    int    k = -1;     // CSR flat index; -1 until resolve_indices() fills it
};

// ---------------------------------------------------------------------------
// Contingency
//   A list of Ybus modifications for one N-k contingency.
//   For a standard N-1 line trip this is typically 4 entries
//   (Y_ii, Y_ij, Y_ji, Y_jj changes), or up to 8 for a transformer.
//
//   disconnected is set to true by check_connectivity() when applying this
//   contingency would leave the Ybus graph disconnected.  Such contingencies
//   are not physically solvable; their result slots receive NaN residuals.
// ---------------------------------------------------------------------------
struct Contingency {
    std::vector<Triplet> triplets;
    bool disconnected = false;
};

// ---------------------------------------------------------------------------
// ChunkPatchRange
//   Slice of the flat device arrays that belongs to one processing chunk.
//   The flat arrays are laid out contiguously: chunk 0 patches, chunk 1
//   patches, …  chunk_ranges[c] = {start, count} indexes into them.
// ---------------------------------------------------------------------------
struct ChunkPatchRange {
    int start;   // first index in the flat arrays for this chunk
    int count;   // number of patch entries in this chunk (sum over actual_batch)
};

// ---------------------------------------------------------------------------
// csr_find_k
//   Binary search for the flat CSR index of (row, col) in a RowMajor matrix.
//   Precondition: the entry (row, col) must exist (non-zero by construction
//   for every valid contingency triplet).  An assertion fires in debug builds
//   if the entry is not found.
//
//   Parameters
//   ----------
//   row_ptr  : CSR outer index array, size n_rows + 1
//   col_ind  : CSR inner index array, size nnz
//   row, col : target entry coordinates
//
//   Returns
//   -------
//   Flat index into col_ind (and into the values array) of entry (row, col).
// ---------------------------------------------------------------------------
int csr_find_k(const int* row_ptr, const int* col_ind, int row, int col);

// ---------------------------------------------------------------------------
// resolve_indices
//   Fills t.k for every triplet in every contingency.  Must be called after
//   the RowMajor Ybus has been built (in run_contingency_analysis_gpu, before
//   constructing ContingencyAnalysisSolver) and before build_flat_patches().
//
//   Parameters
//   ----------
//   contingencies : list of contingencies whose triplets need k filled in
//   row_ptr       : RowMajor Ybus outerIndexPtr(), size n_bus + 1
//   col_ind       : RowMajor Ybus innerIndexPtr(), size nnz
// ---------------------------------------------------------------------------
void resolve_indices(
    std::vector<Contingency>& contingencies,
    const int*                row_ptr,
    const int*                col_ind);

// ---------------------------------------------------------------------------
// build_flat_patches
//   Merges triplets sharing the same CSR index k within each contingency
//   (sort-then-reduce-by-key), then converts the per-contingency triplet lists
//   into three parallel host arrays (SoA layout for coalesced GPU reads) and
//   precomputes per-chunk slices.
//
//   The merge step handles N-2 contingencies and parallel lines whose Ybus
//   modifications map to the same flat CSR entry.  contingencies is modified
//   in-place (triplet lists are sorted and deduplicated); callers must not
//   rely on the original triplet order after this call.
//
//   Compaction
//   ----------
//   Disconnected contingencies (Contingency::disconnected == true) are
//   physically unsolvable and are EXCLUDED from the batch entirely: they occupy
//   no batch slot and consume no factorize/solve work.  Only the connected
//   ("active") contingencies are packed into chunks.  The output active_to_orig
//   maps each active slot back to its original contingency index, so the caller
//   can scatter results to the correct positions (and leave disconnected slots
//   as NaN).  active_to_orig.size() == number of connected contingencies.
//
//   Layout
//   ------
//   The flat arrays concatenate patches for all chunks in order.  Within each
//   chunk the patches for active-slot 0 come first, then active-slot 1, etc.
//   The ctg_id field is chunk-relative (0 .. actual_batch - 1), so the GPU
//   kernel needs no extra offset arithmetic at runtime.
//
//   Chunks are formed over the ACTIVE set: n_chunks = ceil(n_active/batch_size).
//   The last chunk may cover fewer than batch_size active contingencies.
//
//   Parameters
//   ----------
//   contingencies : contingencies with resolved k fields (resolve_indices done)
//                   and disconnected flags (check_connectivity done);
//                   modified in-place (triplets sorted and merged by k)
//   batch_size    : number of (active) contingencies per chunk
//   h_flat_ctg_id : output — chunk-relative active-slot index per patch
//   h_flat_k      : output — CSR flat index into a single Ybus values array
//   h_flat_delta_re/im : output — FP cast deltas ready for device upload
//   chunk_ranges  : output — one ChunkPatchRange per chunk (over active set)
//   active_to_orig: output — active-slot index → original contingency index
// ---------------------------------------------------------------------------
void build_flat_patches(
    std::vector<Contingency>&    contingencies,
    int                          batch_size,
    std::vector<int>&            h_flat_ctg_id,
    std::vector<int>&            h_flat_k,
    std::vector<cuda_real_type>& h_flat_delta_re,
    std::vector<cuda_real_type>& h_flat_delta_im,
    std::vector<ChunkPatchRange>& chunk_ranges,
    std::vector<int>&            active_to_orig);

// ---------------------------------------------------------------------------
// build_blockdiag_csr
//   Builds the structural arrays (outer/inner) for a batch_size × batch_size
//   block-diagonal Ybus used as the cuSPARSE SpMV matrix.
//
//   For a single system with n_bus rows and nnz non-zeros, the block-diagonal
//   matrix for a batch of size b has:
//     outer : b * n_bus + 1 entries  (shifted by i * nnz for block i)
//     inner : b * nnz entries        (shifted by i * n_bus for block i)
//
//   Only outer and inner are built here.  The values buffer (d_Ybus_values_batch)
//   is tiled from base-case values and patched at the start of each chunk call.
//
//   Note: this function builds the structure for the MAXIMUM batch size.  For
//   the last (potentially smaller) chunk the cuSPARSE descriptor is rebuilt to
//   cover only actual_batch systems; outer/inner are still valid up to that
//   prefix because the structure within each block is identical.
//
//   Parameters
//   ----------
//   n_bus         : number of buses in one system
//   nnz           : number of non-zeros in one Ybus
//   single_outer  : Ybus outerIndexPtr(), size n_bus + 1
//   single_inner  : Ybus innerIndexPtr(), size nnz
//   batch_size    : maximum batch size (number of blocks)
//   h_batch_outer : output, size batch_size * n_bus + 1
//   h_batch_inner : output, size batch_size * nnz
// ---------------------------------------------------------------------------
void build_blockdiag_csr(
    int                  n_bus,
    int                  nnz,
    const int*           single_outer,
    const int*           single_inner,
    int                  batch_size,
    std::vector<int>&    h_batch_outer,
    std::vector<int>&    h_batch_inner);

// ---------------------------------------------------------------------------
// check_connectivity
//   Marks contingencies whose Ybus modifications disconnect the bus graph.
//
//   For each contingency, the function determines which off-diagonal Ybus
//   entries become numerically zero after the contingency is applied (i.e.
//   base_value - delta ≈ 0).  It then runs a BFS on the off-diagonal
//   sparsity graph of Ybus_rm with those edges removed.  If fewer than
//   n_bus nodes are reachable, the contingency is physically unsolvable and
//   Contingency::disconnected is set to true.
//
//   Precondition: resolve_indices() must have been called so that every
//   Triplet::k field is valid (used to index into Ybus_rm.valuePtr()).
//
//   Parameters
//   ----------
//   contingencies : contingencies to inspect; disconnected field written in-place
//   Ybus_rm       : RowMajor sparse admittance matrix (base case, with values)
// ---------------------------------------------------------------------------
void check_connectivity(
    std::vector<Contingency>&                                          contingencies,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>&      Ybus_rm);

#endif // CONTINGENCY_ANALYSIS_HELPER_HPP
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TRIPPED_BRANCH_TABLE_HPP
#define TRIPPED_BRANCH_TABLE_HPP

// =============================================================================
// contingency/tripped_branch_table.hpp
//
// TrippedBranchTable — trivial value struct of raw device pointers into a
// BatchSource's own per-active-slot tripped-branch lookup table (see
// build_tripped_branch_table in contingency_analysis_helper.hpp). Consumed by
// check_limit_violations_kernel to skip branches tripped by the contingency
// it is currently checking (their yff/yft/ytf/ytt are unchanged by a
// contingency — only the shared Ybus is patched — so the naive current
// formula would otherwise report a phantom nonzero current for them).
//
// Free-standing (not nested in ContingencyBatch) so InjectionBatch can return
// a trivial all-nullptr instance (it never trips branches) without depending
// on ContingencyBatch's header.
// =============================================================================

struct TrippedBranchTable {
    const int* d_start;        // [n_active] or nullptr
    const int* d_count;        // [n_active] or nullptr
    const int* d_branch_flat;  // [total_trips] or nullptr
};

#endif  // TRIPPED_BRANCH_TABLE_HPP

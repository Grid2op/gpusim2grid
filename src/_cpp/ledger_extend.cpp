// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// ledger_extend.cpp — see ledger_extend.hpp
// =============================================================================

#include "ledger_extend.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

namespace {

// Insert `col` into row `row` of the CSR skeleton, keeping the row's column
// list sorted. Returns true when the entry was added (false: already there).
bool csr_insert_sorted(std::vector<int>& outer, std::vector<int>& inner,
                       int row, int col)
{
    const int s = outer[static_cast<size_t>(row)];
    const int e = outer[static_cast<size_t>(row) + 1];
    auto it = std::lower_bound(inner.begin() + s, inner.begin() + e, col);
    if (it != inner.begin() + e && *it == col) return false;
    inner.insert(it, col);
    for (size_t r = static_cast<size_t>(row) + 1; r < outer.size(); ++r) ++outer[r];
    return true;
}

}  // namespace

int ledger_find_J_pos(const LedgerData& ld, int row, int col)
{
    if (row < 0 || col < 0 || row >= ld.dim_J) return -1;
    const int s = ld.J_outer[static_cast<size_t>(row)];
    const int e = ld.J_outer[static_cast<size_t>(row) + 1];
    const int* inner = ld.J_inner.data();
    const int* it = std::lower_bound(inner + s, inner + e, col);
    if (it == inner + e || *it != col) return -1;
    return static_cast<int>(it - inner);
}

void materialize_vc_custom_rows(LedgerData& ld)
{
    if (!ld.has_voltage_control()) return;
    if (!ld.vc_v_rows.empty()) return;   // already frozen
    const int ng = ld.vc_n_groups();
    ld.vc_v_rows.assign(static_cast<size_t>(ng), -1);
    ld.vc_share_rows.assign(static_cast<size_t>(ng), std::vector<int>());
    int cursor = ld.dim_J - ld.vc_n_controllers();
    for (int g = 0; g < ng; ++g) {
        ld.vc_v_rows[static_cast<size_t>(g)] = cursor++;
        const int cnt = ld.vc_grp_count[static_cast<size_t>(g)];
        for (int k = 0; k < cnt - 1; ++k)
            ld.vc_share_rows[static_cast<size_t>(g)].push_back(cursor++);
    }
}

void reserve_stranded_controller_slots(LedgerData& ld)
{
    if (!ld.has_voltage_control()) return;
    materialize_vc_custom_rows(ld);
    const int ng = ld.vc_n_groups();
    for (int g = 0; g < ng; ++g) {
        if (ld.vc_grp_count[static_cast<size_t>(g)] != 1) continue;
        const int first = ld.vc_grp_start[static_cast<size_t>(g)];
        if (ld.vc_kind[static_cast<size_t>(first)] != 0) continue;  // SVC: already has it
        const int v_row = ld.vc_v_rows[static_cast<size_t>(g)];
        const int q_col = ld.vc_q_col[static_cast<size_t>(first)];
        if (v_row < 0 || q_col < 0) continue;
        csr_insert_sorted(ld.J_outer, ld.J_inner, v_row, q_col);
    }
}

void add_switchable_vm_buses(
    LedgerData&                                                     ld,
    const std::vector<int>&                                         buses,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>&    Ybus_rm)
{
    if (buses.empty()) return;
    materialize_vc_custom_rows(ld);

    const int n_bus = ld.n_bus;
    if (static_cast<int>(Ybus_rm.rows()) != n_bus)
        throw std::runtime_error(
            "add_switchable_vm_buses: Ybus row count does not match the ledger's n_bus");

    // Rebuild the skeleton as a per-row column list (only the touched rows are
    // ever modified; new rows appended); re-serialised once at the end.
    std::vector<std::vector<int>> rows(static_cast<size_t>(ld.dim_J));
    for (int r = 0; r < ld.dim_J; ++r)
        rows[static_cast<size_t>(r)].assign(
            ld.J_inner.begin() + ld.J_outer[static_cast<size_t>(r)],
            ld.J_inner.begin() + ld.J_outer[static_cast<size_t>(r) + 1]);

    const int* Y_outer = Ybus_rm.outerIndexPtr();
    const int* Y_inner = Ybus_rm.innerIndexPtr();

    std::set<int> uniq(buses.begin(), buses.end());
    for (int b : uniq) {
        if (b < 0 || b >= n_bus)
            throw std::runtime_error(
                "add_switchable_vm_buses: bus " + std::to_string(b) + " out of range");
        // A bus that already owns a Vm unknown / Q equation (PQ, or a free-Vm
        // slack) needs nothing -- registering it twice would hand it a second
        // column the dS pass never fills (mirrors Base::register_in's guard).
        if (ld.vm_col_of_bus[static_cast<size_t>(b)] >= 0 ||
            ld.q_row_of_bus[static_cast<size_t>(b)] >= 0)
            continue;
        if (std::find(ld.switchable_vm_buses.begin(), ld.switchable_vm_buses.end(), b)
                != ld.switchable_vm_buses.end())
            continue;

        const int new_col = ld.dim_J;
        const int new_row = ld.dim_J;
        ld.dim_J += 1;
        rows.emplace_back();
        std::vector<int>& qrow = rows.back();

        for (int idx = Y_outer[b]; idx < Y_outer[b + 1]; ++idx) {
            const int k = Y_inner[idx];
            // Column vm_col(b): every neighbour's P / Q equation depends on Vm(b).
            const int pr = ld.p_row_of_bus[static_cast<size_t>(k)];
            const int qr = (k == b) ? new_row : ld.q_row_of_bus[static_cast<size_t>(k)];
            if (pr >= 0) rows[static_cast<size_t>(pr)].push_back(new_col);
            if (qr >= 0 && k != b) rows[static_cast<size_t>(qr)].push_back(new_col);
            // Row q_row(b): Q(b) depends on every neighbour's theta and Vm.
            const int tc = ld.theta_col_of_bus[static_cast<size_t>(k)];
            const int vc = (k == b) ? new_col : ld.vm_col_of_bus[static_cast<size_t>(k)];
            if (tc >= 0) qrow.push_back(tc);
            if (vc >= 0) qrow.push_back(vc);
        }
        // (new_row, new_col): the dQ/dVm diagonal -- also the identity pin slot.
        if (std::find(qrow.begin(), qrow.end(), new_col) == qrow.end())
            qrow.push_back(new_col);

        ld.vm_col_of_bus[static_cast<size_t>(b)] = new_col;
        ld.q_row_of_bus[static_cast<size_t>(b)]  = new_row;
        ld.q_buses.push_back(b);  ld.q_rows.push_back(new_row);
        ld.vm_buses.push_back(b); ld.vm_cols.push_back(new_col);
        ld.switchable_vm_buses.push_back(b);
    }
    std::sort(ld.switchable_vm_buses.begin(), ld.switchable_vm_buses.end());

    // Re-serialise: sorted, de-duplicated columns per row.
    ld.J_outer.assign(1, 0);
    ld.J_inner.clear();
    for (auto& cols : rows) {
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        ld.J_inner.insert(ld.J_inner.end(), cols.begin(), cols.end());
        ld.J_outer.push_back(static_cast<int>(ld.J_inner.size()));
    }
}

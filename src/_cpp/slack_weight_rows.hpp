// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SLACK_WEIGHT_ROWS_HPP
#define SLACK_WEIGHT_ROWS_HPP

// =============================================================================
// slack_weight_rows.hpp
//
// ScenarioSweep set_gen_slack_weights / set_storage_slack_weights (lightsim2grid PR #303):
// per row, the raw distributed-slack weight of some units, standing for theirs (lightsim2grid's
// SlackWeightTable / SlackWeightOverride). Each family has its ids (container numbering), one
// column each in a (n_rows x n_ids) matrix; a NaN cell keeps the unit's own weight. A positive
// value makes a connected unit a participant of that row's slack, flagged or not; 0 takes it out.
// CUDA-free: read by the session, gen_contingency::row_slack_weights, the redistribute_slack
// pre-pass and the generator active-power check.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Core>

namespace slack_weight_rows {
// a weight below it takes no part (upstream BaseConstants::_tol_equal_float)
constexpr double MIN_WEIGHT = 1e-7;
}  // namespace slack_weight_rows

struct SlackWeightRows {
    using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    struct Family {
        std::vector<int> ids;       // the units named, one column each
        std::vector<int> col_of_el; // [n_el] their column, -1 for none
        Mat              w;         // (n_rows, ids.size())
        bool empty() const { return ids.empty(); }
        bool has(int r, int el) const {
            if (ids.empty() || el < 0 || static_cast<size_t>(el) >= col_of_el.size() || r < 0 || r >= w.rows())
                return false;
            const int c = col_of_el[static_cast<size_t>(el)];
            return c >= 0 && !std::isnan(w(r, c));
        }
        // the row's raw weight of `el` (only meaningful where `has`)
        double of(int r, int el) const { return w(r, col_of_el[static_cast<size_t>(el)]); }
        // the row's weight of `el`, `own` where the row gives none
        double weight(int r, int el, double own) const { return has(r, el) ? of(r, el) : own; }
    };

    Family gen, sto;
    std::vector<char> row_has;   // per row: some cell, of either family, holds a number

    bool empty() const { return gen.empty() && sto.empty(); }
    int  n_rows() const { return static_cast<int>(row_has.size()); }
    bool has_row(int r) const { return r >= 0 && r < n_rows() && row_has[static_cast<size_t>(r)] != 0; }
    void refresh_rows() {
        const Eigen::Index n = std::max(gen.empty() ? 0 : gen.w.rows(), sto.empty() ? 0 : sto.w.rows());
        row_has.assign(static_cast<size_t>(n), 0);
        for (const Family* f : {&gen, &sto}) {
            if (f->empty()) continue;
            for (Eigen::Index r = 0; r < f->w.rows(); ++r)
                for (Eigen::Index c = 0; c < f->w.cols(); ++c)
                    if (!std::isnan(f->w(r, c))) { row_has[static_cast<size_t>(r)] = 1; break; }
        }
    }
};

#endif  // SLACK_WEIGHT_ROWS_HPP

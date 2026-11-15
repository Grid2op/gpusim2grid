// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CONTINGENCY_SLOT_RANGES_HPP
#define CONTINGENCY_SLOT_RANGES_HPP

// A list of slots as the few contiguous slot ranges a [n_slots x width] host
// mirror is uploaded by (BatchPfDriver::_oc_write_controls). CUDA-free.

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

// The listed slots as inclusive [first, last] ranges, in increasing order, every
// listed slot in exactly one of them (the list may be unsorted, a slot repeated).
// Two listed slots with at most `max_gap` unlisted ones between them share a
// range, and a list that still needs more than `max_ranges` gets the single range
// spanning it: the copies stay few whatever the list. Only for a mirror whose
// unlisted slots hold what the destination already holds -- they ride along.
inline std::vector<std::pair<int, int>> slot_upload_ranges(const std::vector<int>& slots,
                                                           int                     max_gap    = 8,
                                                           std::size_t             max_ranges = 32)
{
    std::vector<int> sorted(slots);
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::pair<int, int>> out;
    for (int s : sorted) {
        if (!out.empty() && s - out.back().second <= max_gap + 1) out.back().second = s;
        else out.emplace_back(s, s);
    }
    if (out.size() > max_ranges) out.assign(1, std::make_pair(out.front().first, out.back().second));
    return out;
}

#endif  // CONTINGENCY_SLOT_RANGES_HPP

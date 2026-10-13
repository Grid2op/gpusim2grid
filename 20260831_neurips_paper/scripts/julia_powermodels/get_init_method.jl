# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

#
# Julia port of ../get_init_method.py -- same per-case table (kept side-by-side so the two
# language implementations can be diffed directly; update both together if a case's init
# method changes). Substring matching mirrors python's `"caseXXX" in case_name`: `case_name`
# can be a bare filename ("case14.m"), a full path, or a stem -- only containment matters.
#

"""
    get_init_method(case_name) -> String  ("flat", "dc", or "file")

Which starting point powermodels_injection.jl should seed the base-case AC powerflow (and
hence, via warm-starting, every scenario in that case's sweep) from -- see
powermodels_injection.jl's module docstring and apply_dc_start!/apply_voltage_start!.
"""
function get_init_method(case_name::AbstractString)
    occursin("case14", case_name) && return "flat"
    occursin("case30", case_name) && return "flat"
    occursin("case_ieee30", case_name) && return "flat"
    occursin("case57", case_name) && return "flat"
    occursin("case118", case_name) && return "flat"
    occursin("case300", case_name) && return "flat"
    occursin("case_ACTIVSg500", case_name) && return "flat"
    occursin("case1354pegase", case_name) && return "flat"
    occursin("case_ACTIVSg2000", case_name) && return "flat"
    occursin("case2869pegase", case_name) && return "flat"
    occursin("case3120sp", case_name) && return "flat"
    occursin("case3375wp", case_name) && return "file"
    occursin("case6515rte", case_name) && return "dc"
    occursin("case8387pegase", case_name) && return "flat"
    occursin("case9241pegase", case_name) && return "flat"
    occursin("case13659pegase", case_name) && return "dc"
    occursin("case_ACTIVSg10k", case_name) && return "dc"
    occursin("case_ACTIVSg25k", case_name) && return "flat"
    occursin("case_ACTIVSg70k", case_name) && return "file"
    occursin("case500_goc", case_name) && return "flat"
    if occursin("case2000_goc", case_name)
        # none of flat/dc/file converge for this grid (base AC powerflow itself doesn't
        # converge in lightsim2grid); kept as "flat" since it is as good as any other option
        return "flat"
    end
    error("Unknown case $case_name")
end

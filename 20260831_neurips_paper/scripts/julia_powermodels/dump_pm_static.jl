#!/usr/bin/env julia

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


# One-time (per case, not per scenario) dump of PowerModels.jl's parsed network dict to JSON.
#
# Why this exists: PFΔ's raw sample schema is {"network": <PowerModels network dict>,
# "solution": {"solution": <PowerModels AC-PF result>}} (see ../../../pfdelta/core/datasets/
# pfdelta_dataset.py:build_heterodata). The static parts of "network" (branch admittances,
# bus vmin/vmax, gen pmin/pmax/qmin/qmax, shunts, topology) never change across an injection
# sweep -- only pd/qd/pg (inputs) and the solved vm/va/pg/qg (outputs) do. So we only need
# PowerModels.parse_file ONCE per case; the fast per-scenario solve is done separately in
# Python via lightsim2grid (see ../ml_pfdelta_bridge.py), never re-invoking PowerModels.
#
# Usage:
#   julia --project=. dump_pm_static.jl --case ../../../matpowerdata/case118.m \
#       --out ../../../matpowerdata/case118.json

import Pkg
Pkg.activate(@__DIR__)

using ArgParse
using PowerModels
using JSON

PowerModels.silence()

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--case"
            help = "path to the matpower .m case file"
            arg_type = String
            required = true
        "--out"
            help = "path of the JSON file to write (directories created as needed)"
            arg_type = String
            required = true
    end
    return parse_args(s)
end

function main()
    parsed = parse_commandline()
    data = PowerModels.parse_file(parsed["case"])
    mkpath(dirname(parsed["out"]))
    open(parsed["out"], "w") do f
        # some cases (pegase, sp, wp, ...) have genuine Inf limits (e.g. unbounded pmax/qmax)
        # in their matpower source; JSON.jl rejects Inf/NaN by default (not valid JSON), so
        # allow it explicitly rather than silently dropping/clamping real static data.
        # NOTE: JSON.print does not forward kwargs (see JSON.jl:144) -- use JSON.json directly.
        JSON.json(f, data; pretty=2, allownan=true)
    end
    println("Wrote static network ($(length(data["bus"])) buses, $(length(data["gen"])) gens, " *
            "$(length(data["load"])) loads, $(length(data["branch"])) branches) to $(parsed["out"])")
end

main()

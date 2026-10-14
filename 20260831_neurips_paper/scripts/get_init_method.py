# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

def get_init_method(case_name):
    if "case14" in case_name:
        return "flat"
    if "case30" in case_name:
        return "flat"
    if "case_ieee30" in case_name:
        return "flat"
    if "case57" in case_name:
        return "flat"
    if "case118" in case_name:
        return "flat"
    if "case300" in case_name:
        return "flat"
    if "case_ACTIVSg500" in case_name:
        return "flat"
    if "case1354pegase" in case_name:
        return "flat"
    if "case_ACTIVSg2000" in case_name:
        return "flat"
    if "case2869pegase" in case_name:
        return "flat"
    if "case3120sp" in case_name:
        return "flat"
    if "case3375wp" in case_name:
        return "file"
    if "case6515rte" in case_name:
        return "dc"
    if "case8387pegase" in case_name:
        return "flat"
    if "case9241pegase" in case_name:
        return "flat"
    if "case13659pegase" in case_name:
        return "dc"
    if "case_ACTIVSg10k" in case_name:
        return "dc"
    if "case_ACTIVSg25k" in case_name:
        return "flat"
    if "case_ACTIVSg70k" in case_name:
        return "file"
    if "case500_goc" in case_name:
        return "flat"
    if "case2000_goc" in case_name:
        # none of flat/dc/file converge for this grid (base AC powerflow itself doesn't
        # converge in lightsim2grid); kept as "flat" since it is as good as any other option
        return "flat"
    raise RuntimeError(f"Unknown case {case_name}")


def get_ratio_base(case_name):
    if "case14" in case_name:
        return 1.
    if "case_ieee30" in case_name:
        return 1.
    if "case30" in case_name:
        return 1.
    if "case57" in case_name:
        return 1.
    if "case30" in case_name:
        return 1.
    if "case118" in case_name:
        return 1.0
    if "case300" in case_name:
        return 1.0
    if "case_ACTIVSg500" in case_name:
        return 1.0
    if "case1354pegase" in case_name:
        return 1.0
    if "case_ACTIVSg2000" in case_name:
        return 1.0
    if "case2869pegase" in case_name:
        return 1.0
    if "case3120sp" in case_name:
        return 1.0
    if "case3375wp" in case_name:
        return 1.0
    if "case6515rte" in case_name:
        return 1.0
    if "case8387pegase" in case_name:
        return 1.0
    if "case9241pegase" in case_name:
        return 1.0
    if "case13659pegase" in case_name:
        return 0.90  # breaks in "FR" data mode
    if "case_ACTIVSg10k" in case_name:
        return 1.0
    if "case_ACTIVSg25k" in case_name:
        return 1.0
    if "case_ACTIVSg70k" in case_name:
        return 1.0
    if "case500_goc" in case_name:
        return 1.0
    if "case2000_goc" in case_name:
        return 1.0
    raise RuntimeError(f"Unknown case {case_name}")

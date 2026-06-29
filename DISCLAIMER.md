Copyright (c) 2026, RTE (<https://www.rte-france.com>)

See [AUTHORS.txt](AUTHORS.txt)

This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
you can obtain one at <http://mozilla.org/MPL/2.0/>.

SPDX-License-Identifier: MPL-2.0

This file is part of gpusim2grid, a GPU-accelerated power flow solver.

## Disclaimer

This disclaimer only serves as a complement to the [`LICENSE`](LICENSE) file provided
with it. It can not serve as a replacement of this file.

The simulator implemented in this package is made for speed, mainly to run AC power
flow, contingency analysis, and injection sweeps on the GPU.

This simulator is free and uses CUDA, cuDSS, cuSPARSE, Eigen and KLU for performance.
It is not meant to be used as an independent tool for power-system-focused analysis.
In particular, **it implements a raw Newton-Raphson with no outer loop.**
Some of its limitations include, but are not limited to:

- it does not enforce reactive power limits on generators
- it does not model AC/DC converters
- transformers have fixed tap ratio (though it can be changed at initialization of the solver)
- shunts have fixed tap during the Newton-Raphson algorithm (though it can be changed at the initialization of the solver)
- it does not apply distributed slack or other outer-loop corrections
- only powerflow ("steady state") can be performed

It also requires a CUDA-capable NVIDIA GPU and cannot run on CPU-only machines or on
non-NVIDIA accelerators.

## Open source options for powerflow analysis

To get free of these limitations and be able to perform state of the art powerflow
analysis, while still using open source softwares, we kindly recommend you to have a
look at:

- [power-grid-model](https://github.com/PowerGridModel/power-grid-model) which is an "*open-source library for steady-state distribution power system analysis, distributed for Python and C*"
- [Matpower](https://matpower.org/) which is a "*free, open-source tools for electric power system simulation and optimization*"
- [Pandapower](https://www.pandapower.org/) that is "*An easy to use open source tool for power system modeling, analysis and optimization with a high degree of automation.*"
- [PowerModels](https://lanl-ansi.github.io/PowerModels.jl/stable/) described by its authors as "*Julia/JuMP package for Steady-State Power Network Optimization*".
- [GridPack](https://www.pnnl.gov/projects/gridpacktm-open-source-framework-developing-high-performance-computing-simulations-power) which is described as "*An open source toolkit for developing power grid simulation applications for high performance computing architectures*"
- [Dynaωo](https://github.com/dynawo/dynawo) "*an hybrid C++/Modelica suite of simulation tools for power systems*"
- [Rustpower](https://github.com/chengts95/rustpower): "*RustPower is a cutting-edge power flow calculation library written in Rust, specifically designed for steady-state analysis of electrical power systems*"
- [PowSyBl](https://www.powsybl.org/): "*An open-source set of Power System Blocks dedicated to grid analysis, visualization, and simulation.*"

Feel free to consult the excellent
<https://github.com/jinningwang/best-of-ps?tab=readme-ov-file#steady-state-simulation>
for an updated list of power system simulation tools.

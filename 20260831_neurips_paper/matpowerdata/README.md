# Retrieve all grids

All grid files have been downloaded from matpower github here: https://github.com/MATPOWER/matpower/tree/master/data

You can download:

- case14.m
- case_ieee30.m
- case57.m
- case118.m
- case_ACTIVSg500.m
- case1354pegase.m
- case_ACTIVSg2000.m
- case2869pegase.m
- case3120sp.m
- case3375wp.m
- case6515rte.m
- case9241pegase.m
- case_ACTIVSg10k.m
- case_ACTIVSg25k.m
- case_ACTIVSg70k.m

# Some other  commands

## Pypowsybl / OLF

If you want to use the pypowsybl / OLF backend you need to convert them to '.mat' format. You can do this by installing octave (https://octave.org/) and then matpower:

- download matpower
- open octave REPL ("interactive mode")
- cd('/path/to/extracted/matpower')
- install_matpower

Once done, you can the use the "convert_to_mat.py" python script to convert the matpower ".m" data to ".mat"

## Machine learning models

If you want to use the machine learning models from pf-delta, you also need the .json representation (powermodels format) of the same grids. You need to run the `dump_pm_static_all.py` from the `../scripts` repository.

This repo consists of the python mlflux package (machine learning for air-sea flux parameterization) as well as the related notebooks. 

Install by running

`pip install -e .`

in the base directory. Option `-e` (equivalent to `--editable`) allows changes of the source code without requirement for reinstallation.

Functions to compute bulk in `mlflux/datafunc.py` are dependent on the [aerobulk](https://github.com/jbusecke/aerobulk-python) package which (as of 2024.03.31) only runs on Greene. To enable running it on a machine without aerobulk installed we save the computed bulk flux values in data files. This is done by `scripts/compute_bulk.py`.

`others/` folder contains codes that I stole from other people. 


What each file does:

`notebooks/processing/` data processing examples, specific to the in-situ flux datasets that are available to us. 
`data_test.ipynb`: a playground for testing data loading functions bulk flux computation. Some of these functions have been modularized into `mlflux/datafunc.py`.
`synthetic_vs_real.ipynb`: visualizing the distribution of state variables and flux values for both synthetic and real data. 
`mapping.ipynb`, `subERA.ipynb`, `data_process.ipynb`: older notebooks that interpolate wave hindcast to in-situ data samples. Need to be cleaned up.

`notebooks/regression/`




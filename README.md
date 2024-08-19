## Introduction
This repo consists of the python mlflux package (machine learning for air-sea flux parameterization) as well as the related notebooks. 

Install by running 

`pip install -e .`

in the base directory. Option `-e` (equivalent to `--editable`) allows changes of the source code without requirement for reinstallation.

## Dependency
Functions to compute bulk in `mlflux/datafunc.py` are dependent on the [aerobulk](https://github.com/jbusecke/aerobulk-python) package which (as of 2024.03.31) only runs on Greene. To enable running it on a machine without aerobulk installed we save the computed bulk flux values in data files. This is done by `scripts/compute_bulk.py`.

## File structure

`notebooks/processing/` data processing examples, specific to the in-situ flux datasets that are available to us. 

* `data_test.ipynb`: a playground for testing data loading functions bulk flux computation. Some of these functions have been modularized into `mlflux/datafunc.py`. Now also include visualizations of PSD and ATOMIC campaigns and subsets of training and testing.
* `synthetic_vs_real.ipynb`: visualizing the distribution of state variables and flux values for both synthetic and real data. 
* `mapping.ipynb`, `subERA.ipynb`, `data_process.ipynb`: older notebooks that interpolate wave hindcast to in-situ data samples. Need to be cleaned up.

`notebooks/regression/` fitting models, with ANNs and regreesion trees (obsolete).

* `ANN_bulk`: use bulk formula as the deterministic model. Basically residual learning. For now only one variables.
* `ANN_full`: train ANN to predict all three/four outputs.
    * Load data and assemble using RealFluxDataset from ann.py (weights function can be defined and passed-in but for now hard-coded to only depend on wind speed).
    * Initialize a model based on the number of input and output features.
    * Train and select and save.
* `ANN_one_output`: train individual ANN to predict different fluxes. Input can be 4 or 5.
* `Enesmble_select`: test some statistical metrics of each ANN (how close the stochastic residual is to Gaussian) for model selection
* `ANN_restart`: test model reload function
* `ANN_synthetic`: first tests with ANN on synthetic data.
* `ANN_test_diff`: finding best ways to transform inputs etc, but honestly ANN_full is already working pretty well.
* `ANN_wave`: test whether additional wave information (for now not co-located but from reananlysis) will make predictions better.
* `Simple_UQ`: example of simple UQ from Pavel.
* `NGBoost` and `ngboost _trian`: obsolete code with boosting algorithm.
* `Regressor`: not yet filled. Maybe useful for multi-variate Gaussian training.

 
`notebooks/bulk/` code related to bulk algorithm computations.

* `aerobulk`: initial test for the aerobulk package.
* `bulk_global`: apply bulk formula to global CM2 outputs ((right now only one time slice).
* `error_reproduce`: try to reproduce the error I see when applying aerobulk (TODO: send this to Julius).
* `coare`

`notebook/testing/` code for evaluation of ANNs

* `global`: load global CM2 data (right now only one time slice), apply ANNs to compute fluxes, and compute metrics

`gotm/` code related to [GOTM](https://gotm.net/portfolio/) model. 

* `time_series` flux preparation and off-line evaluation
* `analyses` output processing (in Hovmoller diagram format)
* `fluxop` supposed to be flux preparation but haven't moved code from time_series to this one
* some other output files 
  
`others/` folder contains codes that I stole from other people. 


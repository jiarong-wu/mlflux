## Quick orientation for AI coding agents

This repo implements mlflux: a small Python package for learning air-sea flux parameterizations (ANNs) and supporting notebooks/scripts. The goal of these instructions is to make an AI agent immediately productive when editing, extending, or debugging the code.

High-level architecture
- Core package: `mlflux/` contains model definitions (`ann.py`), data helpers (`datafunc.py`), evaluation and plotting (`eval.py`), and predictor wiring (`predictor.py`). Changes here affect training, inference, and evaluation.
- Data: `data/` holds raw and processed NetCDF datasets. Processed PSD data lives under `data/Processed/` (e.g. `psd_coare3p0_weight1_wave0.nc`). Scripts compute and cache bulk fluxes when external dependency `aerobulk` is not available.
- Scripts and notebooks: `scripts/` contains CLI entry points (training, compute_bulk). `notebooks/` are exploratory and demonstrate common usages; copy patterns into scripts when making reproducible changes.

Developer workflows (key commands)
- Install for development: `pip install -e .` (run from repo root).
- Precompute bulk fluxes if aerobulk isn't available: `python scripts/compute_bulk.py` (writes processed files under `data/Processed/`).
- Train a model: `python scripts/training.py --path <model_dir/> --rand <seed>` where `--path` contains `config.json` (see `scripts/config.json` examples under `scripts/`). The training script constructs datasets via `mlflux.ann.RealFluxDataset` and uses `mlflux.predictor.FluxANNs`.
- Load an existing model in notebooks: use `mlflux.eval.open_case(model_dir, model_name)` which expects `config.json` in `model_dir` and returns a pickled model with `.config` attached.

Conventions and patterns
- Data access: code uses xarray Datasets (see `mlflux/datafunc.py`). Typical helper: `load_psd(filepath)` returns a cleaned dataset with bulk fields appended via `applybulk`.
- Dataset shape expectations: `RealFluxDataset` flattens each variable into column vectors; most code assumes 2D arrays shaped (N_samples, N_features). When creating datasets, match `config['ikeys']`, `config['okeys']`, and `config['bkeys']` sizes.
- Weighting: Many evaluation and training functions expect a `weight` variable in the dataset; `scripts/training.py` will insert `ds['weight'] = ds['U']/ds['U']` if not using weighting. Preserve the `weight` variable name when adding or transforming datasets.
- Model components: predictor objects are composed of `mean_func` and `var_func`. `ann.ANN` supports several activation modes (`no`, `square`, `exponential`, `softplus`) to control output positivity.

Files to read first (concrete pointers)
- `mlflux/ann.py` — model & dataset classes (RealFluxDataset) and training utilities. Good for understanding input/output tensor shapes and sample weighting.
- `mlflux/datafunc.py` — data loaders, `applybulk`, `load_psd`, and dataset splitting logic (`data_split_psd`). Important: dataset field renames (e.g. `tsnk->tsea`, `qa->qair`) happen here.
- `scripts/training.py` — CLI glue: how `config.json` is read, how normalization (`compute_norm`) is handled, and calls into `predictor`/training routines.
- `mlflux/eval.py` — evaluation metrics, plotting helpers and `open_case` (how models are serialized/deserialized).

Integration notes / external deps
- aerobulk: optional dependency used in `datafunc.applybulk`. The code already handles ImportError and falls back to saved bulk values. If adding aerobulk-based changes, ensure compatibility with Greene/conda environments.
- NetCDF datasets are read/written with xarray. Large data files live in `data/`; avoid putting new large blobs in git.

Editing guidance and safe changes
- Keep public APIs stable: `RealFluxDataset` input/output keys and `FluxANNs` model save/load formats are used by notebooks and scripts. If you change serialized fields, update `mlflux/eval.open_case` and notebooks accordingly.
- Small tests: use notebooks in `notebooks/` as smoke examples. For quick automated checks, import `mlflux` in a Python REPL and run `from mlflux.datafunc import load_psd; ds = load_psd('data/Processed/psd.nc')` (adjust path) to verify I/O and `applybulk` behavior.

Examples and snippets agents should prefer
- Create dataset for model code paths:
  - Use `RealFluxDataset(ds, input_keys=ikeys, output_keys=okeys, bulk_keys=bkeys)` — this ensures shapes and `W` (weights) are set as training/eval expect.
- Load and inspect model: `model = mlflux.eval.open_case('/path/to/model_dir/', 'model_rand4.p')` — then `model.config` reveals `ikeys/okeys/bkeys/datapath`.

What not to assume
- There is no centralized unit test suite. Changes that touch data loading, normalization, or serialized model formats should be validated manually with small sample NetCDFs or the notebooks.
- Notebooks mirror workflows but are not canonical tests. Prefer modifying `scripts/` and adding small runnable scripts when introducing behavior changes.

If you need more context
- Start with `README.md` then read `mlflux/ann.py`, `mlflux/datafunc.py`, and `scripts/training.py` in that order.
- Ask for paths to sample processed data if you need to run the full training pipeline; large processed files live under `data/Processed/` but may be partial in this repo copy.

-- End of instructions. Please review and tell me any unclear areas or missing examples to include.

# abcd-cca
## Requirements
In addition to the usual suite of scientific Python packages, this repo requires the installation of [pyCirclize](https://github.com/moshi4/pyCirclize) and [CCA Zoo](https://github.com/jameschapman19/cca_zoo).

## Files
**splitting.py**: Create, save, and load splits for cross-validation.

**make_tables.py**: Create, save, and load quantitative measures (qms), resting-state functional connectivity (rsfc), and optional confounds dataframes at each timepoint.

**processing.py**: Split all dataframes into folds according to pre-determined splits, impute missing data, and optionally regress out confounds

**main.py**: Run supported CCA algorithms.

## Use
When using this repo for the first time, I recommend running splitting.py, make_tables.py, processing.py in order. This will save the processed data required for CCA algorithms. All algorithms can then be run in main.py. Savepaths might need to be adjusted for your file system. 

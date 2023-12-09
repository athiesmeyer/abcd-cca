# abcd-cca
splitting.py: Create, save, and load splits for cross-validation.
make_tables.py: Create, save, and load quantitative measures (qms), resting-state functional connectivity (rsfc), and optional confounds dataframes at each timepoint.
processing.py: Split all dataframes into folds according to pre-determined splits, impute missing data, and optionally regress out confounds
main.py: Run any CCA algorithm. Plotting will require tailored filepath setup

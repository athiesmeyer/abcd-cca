import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from pathlib import Path

SPLITS_FILEPATH = Path("data/derived/splits.json")
SITES = [f"site{nr_id:02d}" for nr_id in range(1, 22)]

def inter_site_splits(core_filepath, subject_ids, k=3, seed=0):
    '''Divides subjects first by site, then randomly into k folds (keeping family members together).

    Args:
        core_filepath: Path or str | Path to data core folder
        subjects_ids: pd.DataFrame or array-like | Subject ids
        k: int - optional | Number of folds
        seed: int | Seed for random assignment
    Returns:
        site_splits {str -> {str -> [str]}}: First level is site. Second level is fold.
        values are lists of subject ids
    '''
    if isinstance(subject_ids, pd.DataFrame):
        subject_ids = subject_ids.index.tolist()

    admin_filepath = os.path.join(core_filepath, "abcd-general/abcd_y_lt.csv")
    admin_df = pd.read_csv(admin_filepath)
    admin_df = admin_df.loc[admin_df["src_subject_id"].isin(subject_ids)]
    
    np.random.seed(seed)

    site_splits = {}
    for site_id in SITES:
        site_df = admin_df.loc[admin_df["site_id_l"] == site_id]

        family_ids = set(site_df["rel_family_id"])

        family_groups = []
        for family_id in family_ids:
            family_subjects = list(site_df.loc[site_df["rel_family_id"] == family_id]["src_subject_id"])
            family_groups.append(family_subjects)

        splits = {str(split_idx) : [] for split_idx in range(k)}
        assignments = np.random.choice(list(splits.keys()), len(family_groups))
        for family, assignment in zip(family_groups, assignments):
            splits[assignment] += family

        site_splits[site_id] = splits

    return site_splits

def set_splits(site_splits, test_sites):
    '''Collect ids across sites by folds.

    Args:
        site_splits: {str -> {str -> [str]}} | Output from inter_site_splits. 
        test_sites: str | Selected test sites
    Returns:
        splits: {str -> [str]} | Keys are folds, values are lists of ids
    '''
    splits = {split : [] for split in site_splits[test_sites[0]].keys()}

    test_ids = []
    for test_site in test_sites:
        for ids in site_splits[test_site].values():
            test_ids.extend(ids)
        splits["test"] = test_ids
        del site_splits[test_site]

    for site_split in site_splits.values():
        for split, ids in site_split.items():
            splits[split] += ids

    return splits

def save_splits(splits, filepath=SPLITS_FILEPATH):
    with open(filepath, "w") as savefile:
        json.dump(splits, savefile, indent=4)

def load_splits(filepath=SPLITS_FILEPATH):
    with open(filepath, "r") as f:
        splits = json.load(f)

    return splits

def create_splits(core_filepath, subject_ids,
                   test_sites=["site17"], k=3, seed=0,
                   savepath=SPLITS_FILEPATH):
    '''Get the splits at each site, then collect splits across sites
    by fold. Save results
    '''
    site_splits = inter_site_splits(core_filepath, subject_ids, k=k, seed=seed)
    splits = set_splits(site_splits, test_sites)
    save_splits(splits, savepath)

    return splits

def split_train_val(dfs, val_ids, test_ids=[]):
    '''Given the validation ids of a fold and the test ids,
    split a list of dataframes into training and validation
    dataframes.

    Args:
        dfs: List of pd.DataFrames 
        val_ids: Array | Subject ids in validation set
        test_ids: Array | Subject ids in test set
    Returns:
        split_dfs: List of pd.DataFrames | If dfs = [df1, df2, ...] then
        split_dfs = [df1_train, df1_val, df2_train, df2_val, ...]
    '''
    split_dfs = []
    for df in dfs:
        val = df[df["src_subject_id"].isin(val_ids)]
        train = df[~df["src_subject_id"].isin(val_ids) & ~df["src_subject_id"].isin(test_ids)]
        split_dfs.extend([train, val])

    return split_dfs

if __name__ == "__main__":
    subject_ids = np.load("data/derived/subject_ids.npy", allow_pickle=True)
    core_filepath = Path("data/original/core")
    
    splits = create_splits(core_filepath, subject_ids)
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

    Parameters:
        subjects_df (pandas.DataFrame): Subjects dataframe
    Returns:
        site_splits ({str -> {str -> [str]}}): A dictionary linking each site ID to a k-item long 
            dict linking a split ID to a subject ID list
    '''
    if isinstance(subject_ids, pd.DataFrame):
        subject_ids = subject_ids.index.tolist()

    admin_filepath = os.path.join(core_filepath, r"abcd-general\abcd_y_lt.csv")
    admin_df = pd.read_csv(admin_filepath)
    admin_df = admin_df.loc[admin_df["src_subject_id"].isin(subject_ids)]
    
    np.random.seed(seed)

    site_splits = dict()
    for site_id in SITES:
        site_df = admin_df.loc[admin_df["site_id_l"] == site_id]
        

        family_ids = set(site_df["rel_family_id"])

        family_groups = []
        for family_id in family_ids:
            family_subjects = list(site_df.loc[site_df["rel_family_id"] == family_id]["src_subject_id"])
            family_groups.append(family_subjects)

        splits = {str(split_ix) : [] for split_ix in range(k)}
        assignments = np.random.choice(list(splits.keys()), len(family_groups))
        for family, assignment in zip(family_groups, assignments):
            splits[assignment] += family

        site_splits[site_id] = splits

    return site_splits

def set_splits(site_splits, test_site):
    splits = {split : [] for split in site_splits[test_site].keys()}

    test_ids = []
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
                   test_site="site17", k=3, seed=0,
                   savepath=SPLITS_FILEPATH):
    site_splits = inter_site_splits(core_filepath, subject_ids, k=3, seed=0)
    splits = set_splits(site_splits, test_site)
    save_splits(splits, savepath)

    return splits

def train_val(dfs, ids, test_ids):
    split_dfs = []
    for df in dfs:
        val = df[df["src_subject_id"].isin(ids)]
        train = df[~df["src_subject_id"].isin(ids) & ~df["src_subject_id"].isin(test_ids)]
        split_dfs.extend([train, val])

    return split_dfs

if __name__ == "__main__":
    subject_ids = np.load(r"data\derived\subject_ids.npy", allow_pickle=True)
    core_filepath = Path("data/original/core")
    
    splits = create_splits(core_filepath, subject_ids)
    print(list(splits.keys()))

    splits = load_splits()
    print(list(splits.keys()))
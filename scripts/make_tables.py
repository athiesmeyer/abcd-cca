import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
from collections import OrderedDict
from itertools import product

CORE_FILEPATH = Path("data/original/core")
QMS_FILEPATH = Path("data/derived/qms.txt")
CONFOUNDS_FILEPATH = Path("data/derived/confounds.txt")
SUBJECT_IDS_FILEPATH = Path("data/derived/subject_ids.npy")
SAVED_QMS_DFS_FILEPATH = Path("data/derived/qms_dfs")
SAVED_RSFC_DFS_FILEPATH = Path("data/derived/rsfc_dfs")
SAVED_CONFOUNDS_DFS_FILEPATH = Path("data/derived/confounds_dfs")

NETWORKS = OrderedDict([("ad","auditory"),
            ("cgc","cing. opercular"),
            ("ca","cing. parietal"),
            ("dt","default"),
            ("dla","dorsal att."),
            ("fo","fronto parietal"),
            ("n","none"),
            ("rspltp","retrospl. temp."),
            ("sa","salience"),
            ("smh","sensorimr. hand"),
            ("smm","sensorimr. mouth"),
            ("vta","ventral att."),
            ("vs","visual")])

CONNECTIONS = ["rsfmri_c_ngd_{}_ngd_{}".format(n1, n2) for (n1, n2) in 
               product(NETWORKS.keys(), NETWORKS.keys())]

TIMEPOINTS = ["baseline_year_1_arm_1", "2_year_follow_up_y_arm_1", "4_year_follow_up_y_arm_1"]

class TableMaker():
    def __init__(self,
                 core_filepath = CORE_FILEPATH,
                 qms_filepath = QMS_FILEPATH,
                 confounds_filepath = CONFOUNDS_FILEPATH,
                 subject_ids_filepath = SUBJECT_IDS_FILEPATH,
                 saved_qms_dfs_filepath = SAVED_QMS_DFS_FILEPATH,
                 saved_rsfc_dfs_filepath = SAVED_RSFC_DFS_FILEPATH,
                 saved_confounds_dfs_filepath = SAVED_CONFOUNDS_DFS_FILEPATH,
                 timepoints = TIMEPOINTS):
        self.core_filepath = core_filepath
        self.qms_filepath = qms_filepath
        self.confounds_filepath = confounds_filepath
        self.subject_ids_filepath = subject_ids_filepath
        self.saved_qms_dfs_filepath = saved_qms_dfs_filepath
        self.saved_rsfc_dfs_filepath = saved_rsfc_dfs_filepath
        self.saved_confounds_dfs_filepath = saved_confounds_dfs_filepath
        self.timepoints = TIMEPOINTS

        self.subject_ids = self.load_subject_ids(self.subject_ids_filepath)
        self.static_measures = np.array([])

    def load_subject_ids(self, path):
        subject_ids = np.load(path, allow_pickle=True)
        subject_ids = pd.DataFrame({"src_subject_id": subject_ids})
        subject_ids = subject_ids.set_index("src_subject_id")

        self.subject_ids = subject_ids

        return subject_ids
    
    def _fix_col(self, col):
        try:
            col.astype(np.float64)
        except Exception:
            # Assume comma was placed where a decimal was 
            # supposed to go
            for i in range(len(col)):
                val = col[i]
                
                if isinstance(val, str) and "," in val:
                    halves = val.split(",")
                    col[i] = halves[0] + "." + halves[1]
        
        return col    

    def _validate_data(self, df):
        if "eventname" in df.columns:
            df = df.drop("eventname", axis=1)

        # Every column besides src_subject_id should
        # be numeric.
        for i in range(1, df.shape[1]):
            df.iloc[:, i] = self._fix_col(df.iloc[:, i].to_numpy())

        return df
    
    def _validate_confounds(self, df):
        if "eventname" in df.columns:
            df = df.drop("eventname", axis=1)

        for i in range(1, df.shape[1]):
            try:
                df.iloc[:, i] = df.iloc[:, i].astype(np.float64)
            except Exception:
                pass
        ohe_df = pd.get_dummies(df.iloc[:, 1:], drop_first=True, dtype=np.float64)
        ohe_df.insert(0, column="src_subject_id", value=df["src_subject_id"].to_numpy())
        ohe_df.insert(ohe_df.shape[1], column="intercept", value=np.ones(ohe_df.shape[0]))

        return ohe_df

    def _join(self, df, dir, cols):
        for path in os.listdir(dir):
            path = os.path.join(dir, path)
            if os.path.isdir(path):
                print(f"Searching in {path}")
                df = self._join(df, path, cols)
            else:
                long_table = pd.read_csv(path)
                long_table = long_table.loc[:, [True if col in cols else \
                                                False for col in long_table.columns]]

                if len(long_table.columns) > 2:
                    try:
                        wide_table = long_table.pivot(index="src_subject_id",
                                                    columns="eventname")
                        df = df.join(wide_table, rsuffix=path[:-3], how="left")
                    except Exception:
                        print(f"Failed import or join from {path}")

        return df

    def qms_full(self, qms):
        '''Loop through all csv files and get all data for the 
        quantitative measures for the specified 
        subject_ids. Different qms have different numbers of 
        measurements, so each qm/eventtime pairing makes up
        a column, with the data in wide format.
        '''
        if not isinstance(qms, list):
            # Assume we have a filepath
            with open(qms) as f:
                qms = f.read()

            qms = list(map(lambda x : x.strip(), qms.split(",")))
        
        cols = qms + ["src_subject_id", "eventname"]

        warnings.filterwarnings("ignore")
        qms_df = self._join(self.subject_ids, self.core_filepath, cols).reset_index()

        return qms_df

    def qms_at_timepoints(self, qms_df):
        dfs = []
        for timepoint in self.timepoints:
            mask = [True] + [True if col[1] == timepoint else False for col in qms_df.columns[1:]]
            df = qms_df.loc[:, mask]
            df.columns = [df.columns[0]] + [col[0] for col in df.columns[1:]]
            dfs.append(df)

        return dfs

    def _add_static_measures(baseline_qms, qms_dfs):
        '''For every column in the baseline dataframe, if it
        is not present in the other dataframes, add it.
        '''
        baseline_measures = set(baseline_qms.columns)
        union = set([])
        intersect = set(baseline_qms.columns).copy()
        for qms_df in qms_dfs:
            union = union.union(set(qms_df.columns))
            intersect = intersect.intersection(set(qms_df.columns))

        static_measures = baseline_measures.difference(union)
        not_static_not_full_measures = list(baseline_measures.difference(static_measures.union(intersect)))

        baseline_qms = baseline_qms.loc[:, ~baseline_qms.columns.isin(not_static_not_full_measures)]

        new_dfs = []
        for qms_df in qms_dfs:
            qms_df = qms_df.loc[:, ~qms_df.columns.isin(not_static_not_full_measures)]
            new_df = pd.concat([qms_df, baseline_qms.loc[:, list(static_measures)]],
                                axis=1)
            new_dfs.append(new_df)

        return [baseline_qms] + new_dfs, list(static_measures)
    
    def _halve_mat(self, x):
        if isinstance(x[0], str):
            empty = ""
        else:
            empty = 0

        n = int(np.sqrt(len(x)))
        x = x.reshape((n, n))
        x = np.tril(x)
        x = x.flatten()
        x = x[x != empty]

        return x

    def _halve_graph(self, df):
        # df is assumed to be a df containing rsfc df info
        # with src_subject_id as first column
        concise_graph = []
        for i in range(df.shape[0]):
            concise_graph.append(self._halve_mat(df.iloc[i, 1:].to_numpy()))
        concise_df = np.stack(concise_graph)
        edge_names = self._halve_mat(np.array(df.columns.tolist())[1:])
        concise_df = pd.DataFrame(concise_df, columns=edge_names)
        concise_df.insert(loc=0, column="src_subject_id",
                            value = df["src_subject_id"].to_numpy())

        return concise_df
    
    def rsfc_full(self):
        path = os.path.join(self.core_filepath, "imaging/mri_y_rsfmr_cor_gp_gp.csv")
        rsfc_df = pd.read_csv(path, usecols=["src_subject_id", "eventname"] + CONNECTIONS)
        rsfc_df = rsfc_df.loc[rsfc_df["src_subject_id"].isin(self.subject_ids.index.tolist()), :]\
            .reset_index(drop=True)
        
        return rsfc_df

    def rsfc_at_timepoints(self, rsfc_df):
        dfs = []
        for timepoint in self.timepoints:
            df = rsfc_df[rsfc_df["eventname"] == timepoint]
            df = df.drop("eventname", axis=1)
            df = self._halve_graph(df)
            dfs.append(df)

        return dfs

    def save_dfs(self, dfs, type):
        if type == "qms":
            dir_path = self.saved_qms_dfs_filepath

            if not self.static_measures is None:
                np.save(os.path.join(dir_path, "static_measures.npy"),
                        self.static_measures)
            else:
                print("No static measures among selected quantitative measures")
        elif type == "rsfc":
            dir_path = self.saved_rsfc_dfs_filepath
        elif type == "confounds":
            dir_path = self.saved_confounds_dfs_filepath

        for df, timepoint in zip(dfs, self.timepoints):
            savepath = os.path.join(dir_path, f"{timepoint}.csv")
            df.to_csv(savepath, index=False)

    def load_dfs(self, type):
        if type == "qms":
            dir_path = self.saved_qms_dfs_filepath
            try:
                self.static_measures = np.load(os.path.join(dir_path, "static_measures.npy"),
                                            allow_pickle=True)
            except Exception:
                print("There are no saved static measures associated with the current qms dfs")
        elif type == "rsfc":
            dir_path = self.saved_rsfc_dfs_filepath
        elif type == "confounds":
            dir_path = self.saved_confounds_dfs_filepath

        dfs = {}
        for timepoint in self.timepoints:
            path = os.path.join(dir_path, f"{timepoint}.csv")
            df = pd.read_csv(path)
            dfs[timepoint] = df

        return dfs
    
    def load_all_dfs(self, confounds = False):
        types = ["qms", "rsfc"]
        if confounds:
            types.append("confounds")

        all_dfs = {}
        for type in types:
            dfs = self.load_dfs(type)
            all_dfs[type] = dfs

        return all_dfs

    def create_qms_dfs(self):
        qms_df = self.qms_full(self.qms_filepath)
        dfs = self.qms_at_timepoints(qms_df)
        dfs, static_measures = self._add_static_measures(dfs[0], dfs[1:])
        self.static_measures = static_measures

        dfs = [self._validate_data(df) for df in dfs]
        self.save_dfs(dfs, type="qms")

        return dfs

    def create_rsfc_dfs(self):
        rsfc_df = self.rsfc_full()
        dfs = self.rsfc_at_timepoints(rsfc_df)
        dfs = [self._validate_data(df) for df in dfs]
        self.save_dfs(dfs, type="rsfc")

        return dfs
    
    def create_confounds_dfs(self):
        confounds_df = self.qms_full(self.confounds_filepath)
        dfs = self.qms_at_timepoints(confounds_df)
        dfs = [self._validate_confounds(df) for df in dfs]
        self.save_dfs(dfs, type="confounds")

        return dfs

if __name__ == "__main__":
    table_maker = TableMaker()
    dfs = table_maker.create_confounds_dfs()

    for df in dfs:
        print(df.shape)

    print(dfs[0].head())
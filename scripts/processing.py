from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, SimpleImputer
from make_tables import TableMaker, TIMEPOINTS
from splitting import load_splits, split_train_val
import numpy as np
import pickle
import json

PROCESSED_DATA_FILEPATH = "data/derived/processed_data.pickle"
QMS_VARS_TO_LABELS_FILEPATH = "data/derived/variables_to_labels.json"

class ProcessedData():
    def __init__(self,
                 processed_datasets,
                 qms_variables,
                 rsfc_variables,
                 timepoints,
                 n_folds,
                 static_measures
                ):
        self.processed_datasets = processed_datasets
        self.qms_variables = qms_variables
        self.rsfc_variables = rsfc_variables
        self.timepoints = timepoints
        self.n_folds = n_folds
        self.static_measures = static_measures

        self.qms_labels = self.qms_vars_to_labels(QMS_VARS_TO_LABELS_FILEPATH)

    def qms_vars_to_labels(self, map_path):
        '''Use exising mapper to map qms variable names to more easily 
        understood labels.
        '''
        with open(map_path, "r") as f:
            variable_to_label = json.load(f)
        qms_labels = np.array([variable_to_label[qm] for qm in self.qms_variables])

        return qms_labels

class Preprocessor():
    '''The primary functionality of this class is that given
    qms and rsfc dataframes at each timepoint and splits for cv,
    remove missing data and optionally regress out confounds 
    from these dataframes so they are in a final state ready
    to be passed to CCA operations.'''
    def __init__(self,
                 data,
                 splits,
                 static_measures,
                 timepoints = TIMEPOINTS,
                 seed = 5,
                 savepath = PROCESSED_DATA_FILEPATH):
        self.data = data
        self.splits = splits        
        self.timepoints = timepoints
        self.seed = seed
        self.savepath = savepath
        self.static_measures = static_measures
        self.n_folds = len(list(splits.keys())) - 1

    def mice_imputation(self, df, type):
        '''
        Args:
            df: Array
            type: str | Either "qms" or "rsfc"
        '''
        if type == "rsfc":
            mice = IterativeImputer(max_iter=10,
                                    n_nearest_features=20,
                                    random_state=self.seed)
        else:
            mice = IterativeImputer(max_iter=100,
                                    random_state=self.seed)
        imputed_df = mice.fit_transform(df)

        return imputed_df
    
    def regress_out(self, data, confounds):
        return data - confounds @ np.linalg.pinv(confounds) @ data

    def process_confounds(self, confounds_dfs):
        # Impute missing data in confounds dataframes using simple 
        # imputation rather than mice for simplicity
        for i, df in enumerate(confounds_dfs):
            df = df.sort_values("src_subject_id")
            df = df.drop("src_subject_id", axis=1)

            imp = SimpleImputer(strategy="median")
            df = imp.fit_transform(df)

            confounds_dfs[i] = df

        return confounds_dfs
    
    def preprocess_at_timepoints(self, timepoint, confounds):
        '''Take qms, rsfc dataframes at timepoint, split them into
        train/val folds for cv, impute missing data for each matrix
        individually, and optionally regress out confounds in each
        '''
        # dfs will be [qms_dataframe, rsfc_dataframe, confounds_dataframe]
        # for specified timepoint
        dfs = []
        for type in self.data.keys():
            df = self.data[type][timepoint]
            dfs.append(df)

        datasets_at_timepoint = {}
        for fold, ids in self.splits.items():
            print(f"Processing for fold: {fold}")
            # split_dfs will be [qms_train, qms_val, rsfc_train, rsfc_val, 
            # confounds_train, confounds_val]
            if fold != "test":
                split_dfs = split_train_val(dfs, ids, self.splits["test"])
            else:
                split_dfs = split_train_val(dfs, ids)

            if confounds:
                confounds_dfs = self.process_confounds(split_dfs[-2:])
            
            # If confounds data was passed, remove it from split_dfs
            # even if confounds arg is true
            if "confounds" in list(self.data.keys()):
                split_dfs = split_dfs[:-2].copy()

            # We have four relevant dataframes now: qms_train, qms_val,
            # rsfc_train, rsfc_val. Impute missing data and regress out
            # correct confounds df for each
            train_dfs = []
            val_dfs = []
            for j, df in enumerate(split_dfs):
                # Very specific to output of split_train_val
                is_val = j % 2
                type = "qms" if j < 2 else "rsfc"

                # Turn df from pandas dataframe to numpy array
                df = df.sort_values("src_subject_id")
                df = df.drop("src_subject_id", axis=1)
                df = df.to_numpy()

                # Impute and regress out confounds
                df = self.mice_imputation(df, type)
                if confounds:
                    confounds_df = confounds_dfs[is_val]
                    df = self.regress_out(df, confounds_df)

                if not is_val:
                    train_dfs.append(df)
                else:
                    val_dfs.append(df)

            datasets_at_timepoint[fold] = {"train": train_dfs, "val": val_dfs}

        return datasets_at_timepoint

    def preprocess(self, confounds=True, save=True):
        '''The final output of preprocessing is an object of
        type ProcessedData. ProcessedData.processed_datasets
        contains the relevant data itself, in the form of a 
        dictionary {str -> str -> str -> [qms_df, rsfc_df]} where
        the first level is timepoint, the second level is fold,
        and the third level is "train" or "val"
        '''
        processed_datasets = {}
        for timepoint in self.timepoints:
            print(f"Processing at timepoint: {timepoint}")
            datasets_at_timepoint = self.preprocess_at_timepoints(timepoint, confounds=confounds)
            processed_datasets[timepoint] = datasets_at_timepoint

        qms_variables = self.data["qms"][self.timepoints[0]].columns[1:].to_numpy()
        rsfc_variables = self.data["rsfc"][self.timepoints[0]].columns[1:].to_numpy()
        processed_data = ProcessedData(processed_datasets,
                                       qms_variables,
                                       rsfc_variables,
                                       self.timepoints,
                                       self.n_folds,
                                       self.static_measures)
        
        if save:
            self.save_datasets(processed_data)

        return processed_data      

    def combine_timepoints(self, processed_datasets):
        '''Most analysis is done on individual timepoints. This function
        takes an inefficient approach to obtaining processed data concatenated along
        time by using the processed output split at each timepoint. 

        Args:
            processed_datasets: {str -> str -> str -> [qms_df, rsfc_df]} | processed_datasets
            attribute of a ProcessedData object
        Returns:
            combined_data: {str -> str -> [qms_df, rsfc_df]} | First level is fold,
            second level 
        '''
        qms_df = self.data["qms"][self.timepoints[0]]
        non_static_idxs = np.arange(0, len(qms_df.columns))[~qms_df.columns.isin(self.static_measures)]

        combined_data = {}
        for i in range(self.n_folds + 1):
            fold = f"{i}" if i < self.n_folds else "test"

            qms_train, rsfc_train = processed_datasets[self.timepoints[0]][fold]["train"]
            qms_val, rsfc_val = processed_datasets[self.timepoints[0]][fold]["val"]

            for timepoint in self.timepoints[1:]:
                qms_train_t, rsfc_train_t = processed_datasets[timepoint][fold]["train"]
                qms_val_t, rsfc_val_t = processed_datasets[timepoint][fold]["val"]

                qms_train, rsfc_train = np.concatenate([qms_train, qms_train_t[:, non_static_idxs]], axis=1),\
                    np.concatenate([rsfc_train, rsfc_train_t], axis=1)
                qms_val, rsfc_val = np.concatenate([qms_val, qms_val_t[:, non_static_idxs]], axis=1),\
                    np.concatenate([rsfc_val, rsfc_val_t], axis=1)

            combined_data[fold] = {"train": [qms_train, rsfc_train],
                                   "val": [qms_val, rsfc_val]}
            
        return combined_data
    
    def save_datasets(self, processed_data):
        with open(self.savepath, 'wb') as f:
            pickle.dump(processed_data, f)
    
    def load_datasets(self):
        with open(self.savepath, 'rb') as f:
            processed_data = pickle.load(f)

        return processed_data

if __name__ == "__main__":
    # Load dataframes and splits
    table_maker = TableMaker()
    dfs = table_maker.load_all_dfs(confounds=True)
    splits = load_splits()

    # Preprocess
    preprocessor = Preprocessor(dfs, splits, table_maker.static_measures)
    processed_data = preprocessor.preprocess()



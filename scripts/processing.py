from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, SimpleImputer
from nipals.nipals import Nipals
from make_tables import TableMaker, TIMEPOINTS
from splitting import load_splits, train_val
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
import pickle

PROCESSED_DATA_FILEPATH = "data/derived/processed_data.pickle"

class Preprocessor():
    def __init__(self,
                 data,
                 splits,
                 timepoints = TIMEPOINTS,
                 seed = 5,
                 savepath = PROCESSED_DATA_FILEPATH):
        self.data = data
        self.splits = splits        
        self.timepoints = timepoints
        self.seed = seed
        self.savepath = savepath
        
        self.n_folds = len(list(splits.keys())) - 1

    def process_confounds(self, confounds_dfs):
        for i, df in enumerate(confounds_dfs):
            df = df.sort_values("src_subject_id")
            df = df.drop("src_subject_id", axis=1)

            imp = SimpleImputer(strategy="median")
            df = imp.fit_transform(df)

            confounds_dfs[i] = df

        return confounds_dfs
    
    def regress_out(self, data, confounds):
        # 'data' and 'confounds' are numpy arrays, not
        # pandas dataframes
        
        return data - confounds @ np.linalg.pinv(confounds) @ data

    def preprocess(self, confounds=False, save=True):
        '''Outputs a dictionary where at each timepoint we have a subdictionary containing the 
        training and validation datasets at each fold. Dictionary has three total levels.
        '''
        processed_datasets = {}
        for i, timepoint in enumerate(self.timepoints):
            print(f"Processing at timepoint: {timepoint}")
            dfs = []
            for type in self.data.keys():
                dfs.append(self.data[type][timepoint])

            datasets_at_timepoint = {}
            for fold, ids in self.splits.items():
                print(f"Processing for fold: {fold}")
                if fold != "test":
                    split_dfs = train_val(dfs, ids, self.splits["test"])
                else:
                    split_dfs = train_val(dfs, ids, [])

                if confounds:
                    confounds_dfs = self.process_confounds(split_dfs[-2:])

                train_dfs = []
                val_dfs = []
                for j, df in enumerate(split_dfs[:-2]):
                    is_val = j % 2

                    df = df.sort_values("src_subject_id")
                    df = df.drop("src_subject_id", axis=1)
                    df = df.to_numpy()

                    df = self.mice_imputation(df)
                    if confounds:
                        confounds_df = confounds_dfs[is_val]
                        df = self.regress_out(df, confounds_df)

                    if not is_val:
                        train_dfs.append(df)
                    else:
                        val_dfs.append(df)

                datasets_at_timepoint[fold] = {"train": train_dfs, "val": val_dfs}
            
            processed_datasets[timepoint] = datasets_at_timepoint
        
        if save:
            self.save_datasets(processed_datasets)

        return processed_datasets

    def mice_imputation(self, df, max_iter = 100):
        mice = IterativeImputer(max_iter = max_iter, random_state = self.seed)
        imputed_df = mice.fit_transform(df)

        return imputed_df
    
    def save_datasets(self, processed_datasets):
        with open(self.savepath, 'wb') as f:
            pickle.dump(processed_datasets, f)
    
    def load_datasets(self):
        with open(self.savepath, 'rb') as f:
            processed_data = pickle.load(f)

        return processed_data

    # def pca(self, X, ncomp):
    #     # Inefficient algo to estimate cov matrix, ignoring missing data
    #     n = X.shape[1]
    #     cov_hat = np.zeros((n, n))
    #     for i in range(n):
    #         for j in range(n):
    #             x_i = X[:, i]
    #             x_j = X[:, j]
    #             missing = np.isnan(x_i) | np.isnan(x_j)

    #             x_i = x_i[~missing]
    #             x_j = x_j[~missing]
    #             cov_hat[i, j] = np.dot(x_i - x_i.mean(), x_j - x_j.mean()) / len(x_i)
    #     cov_hat = cov_nearest(cov_hat)
    #     U, S, V_t = np.linalg.svd(cov_hat)
    #     reduced_X = U[:, :ncomp] * S[:ncomp]
    #     return reduced_X

if __name__ == "__main__":
    table_maker = TableMaker()
    dfs = table_maker.load_all_dfs(confounds=True)
    splits = load_splits()

    preprocessor = Preprocessor(dfs, splits)
    processed_data = preprocessor.preprocess(confounds=True)    



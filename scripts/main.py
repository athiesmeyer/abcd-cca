from make_tables import *
from splitting import *
from processing import *
from sklearn.cross_decomposition import CCA

PLOT_FILEPATH = Path("plots")

class CCAPipeline():
    def __init__(self, processed_data, timepoints, plot_savepath=PLOT_FILEPATH):
        self.timepoints = timepoints
        self.processed_data = processed_data
        self.plot_savepath = plot_savepath

    def _cca(self, qms_train, qms_val, rsfc_train, rsfc_val, max_iter=500):
        '''Do a simple CCA on the qms_train and rsfc_train data.
        Report the correlation in every canonical direction.
        '''
        n_comps = min(qms_train.shape[1], rsfc_train.shape[1])
        cca_model = CCA(n_components=n_comps, max_iter=max_iter)

        # Scores refers to the result we get after projecting a feature vector into
        # the canonical directions
        qms_train_scores, rsfc_train_scores = cca_model.fit_transform(qms_train, rsfc_train)
        qms_val_scores, rsfc_val_scores = cca_model.transform(qms_val, rsfc_val)

        # We find the correlation between every mode.
        train_all_corrs = np.array([np.corrcoef(qms_train_scores[:, i], rsfc_train_scores[:, i])[0, 1]\
                                    for i in range(n_comps)])
        val_all_corrs = np.array([np.corrcoef(qms_val_scores[:, i], rsfc_val_scores[:, i])[0, 1]\
                                    for i in range(n_comps)])

        return train_all_corrs, val_all_corrs, cca_model

    def cca(self, data, max_iter=500):
        # Wrapper for cca function
        train_all_corrs, test_all_corrs, cca_model = self._cca(data["train"][0], data["val"][0],
                                                                data["train"][1], data["val"][1],
                                                                max_iter=max_iter)
        return train_all_corrs, test_all_corrs, cca_model

    def pass_directions(self, time):
        '''Perform a CCA at baseline timepoint and pass canonical directions
        forward to evaluate performance at later timepoints.'''
        source_data = self.processed_data[self.timepoints[time]]
        train_all_corrs, _, cca_model = self.cca(source_data["test"])
        print(f"Correlation in the first mode on source data: {train_all_corrs[0]}")
        print(f"Total correlation on the source data: {train_all_corrs.sum()}")
        
        remaining_timepoints = [timepoint for i, timepoint in enumerate(self.timepoints) if i != time]
        for timepoint in remaining_timepoints:
            # The datasets we evaluate on are just all subjects not in the 
            # test set, which is exactly the training set for the test fold.
            qms_df = self.processed_data[timepoint]["test"]["train"][0]
            rsfc_df = self.processed_data[timepoint]["test"]["train"][1]
            qms_scores, rsfc_scores = cca_model.transform(qms_df, rsfc_df)

            all_corrs = np.array([np.corrcoef(qms_scores[:, i], rsfc_scores[:, i])[0, 1]\
                                    for i in range(qms_scores.shape[1])])

            print(f"Correlation in the first mode at {timepoint}: {all_corrs[0]}")
            print(f"Total correlation at timepoint {timepoint}: {all_corrs.sum()}")

    def run_cca_train_and_val(self, plot_results=True, verbose=True):
        corrs = []
        for timepoint in self.timepoints:
            corrs.append(self.cca_train_and_val(self.processed_data[timepoint]))

        if plot_results:
            self.plot_results(corrs)

        if verbose:
            self.summarize_results(corrs, self.timepoints)

    def cca_train_and_val(self, datasets_at_timepoint):
        '''For each fold of cross-validation, run cca on the train-val split.'''
        train_corrs = []
        val_corrs = []
        for fold, data in datasets_at_timepoint.items():
            if fold != "test":
                train_all_corrs, val_all_corrs, _ = self.cca(data)
                train_corrs.append(train_all_corrs)
                val_corrs.append(val_all_corrs)
        train_corrs = np.stack(train_corrs)
        val_corrs = np.stack(val_corrs)

        return train_corrs, val_corrs

    def plot_results(self, corrs):
        # if not any(isinstance(x, list) for x in corrs):
        #     corrs = [corrs]
        fig, axs = plt.subplots(figsize=(15, 8), ncols=len(corrs), sharey=True)
        for i in range(len(corrs)):
            ax = axs[i]
            train_corrs = corrs[i][0]
            val_corrs = corrs[i][1]

            x = np.arange(1, train_corrs.shape[1] + 1)

            ax.errorbar(x, train_corrs.mean(axis=0),
                        yerr=1.96 * train_corrs.std(axis=0) / train_corrs.shape[0],
                        label="Train")
            ax.errorbar(x, val_corrs.mean(axis=0),
                        yerr=1.96 * val_corrs.std(axis=0) / val_corrs.shape[0],
                        label="Val")
            ax.legend()

        savepath = os.path.join(self.plot_savepath, "train_val_correlations")
        plt.savefig(savepath)

    def summarize_results(self, corrs, timepoints):
        for i in range(len(corrs)):
            train_corrs = corrs[i][0]
            val_corrs = corrs[i][1]
            timepoint = timepoints[i]

            print(("Average correlation in the first mode on training data" 
                   f" at timepoint {timepoint} is {train_corrs.mean(axis=0)[0]}"))
            print(("Average correlation in the first mode on validation data" 
                   f" at timepoint {timepoint} is {val_corrs.mean(axis=0)[0]}"))
            
    def compare_scores_with_features(self, qms):
        # qms is a numpy array containing the column labels of qms_df
        for timepoint in self.timepoints:
            datasets_at_timepoint = self.processed_data[timepoint]
            qms_df = datasets_at_timepoint["test"]["train"][0]
            rsfc_df = datasets_at_timepoint["test"]["train"][1]

            n_comp = min(qms_df.shape[1], rsfc_df.shape[1])
            cca = CCA(n_components=n_comp, max_iter=1000)
            qms_scores, rsfc_scores = cca.fit_transform(qms_df, rsfc_df)
            qms_first_mode = qms_scores[:, 0]

            corrs = np.array([np.corrcoef(qms_first_mode, qms_df[:, i])[0, 1]\
                                    for i in range(qms_df.shape[1])])
            corrs = pd.DataFrame({"correlation": corrs}, index=qms)
            corrs = corrs.sort_values("correlation", ascending=False)
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(corrs)

    def overall_cca(self, n_folds, qms):
        for i in range(n_folds):
            qms_train, rsfc_train = [], []
            qms_val, rsfc_val = [], []
            for timepoint in self.timepoints:
                qms_train_df, rsfc_train_df = self.processed_data[timepoint][f"{i}"]["train"]
                qms_val_df, rsfc_val_df = self.processed_data[timepoint][f"{i}"]["val"]
                qms_train.append(qms_train_df)
                rsfc_train.append(rsfc_train_df)
                qms_val.append(qms_val_df)
                rsfc_val.append(rsfc_val_df)
            qms_train = np.concatenate(qms_train)
            rsfc_train = np.concatenate(rsfc_train)
            n_comp = min(qms_train.shape[1], rsfc_train.shape[1])
            cca = CCA(n_components=n_comp, max_iter=1000)
            cca.fit(qms_train, rsfc_train)

            j = 0
            for qms_val_df, rsfc_val_df in zip(qms_val, rsfc_val):
                qms_val_scores, rsfc_val_scores = cca.transform(qms_val_df, rsfc_val_df)
                primary_corr = np.corrcoef(qms_val_scores[:, 0], rsfc_val_scores[:, 0])[0, 1]
                print(f"Correlation in the first mode on validation data for fold {i} at timepoint {self.timepoints[j]}: {primary_corr}")
                if i == 0:
                    corrs = np.array([np.corrcoef(qms_val_scores[:, 0], qms_val_df[:, i])[0, 1]\
                                    for i in range(qms_val_df.shape[1])])
                    corrs = pd.DataFrame({"correlation": corrs}, index=qms)
                    corrs = corrs.sort_values("correlation", ascending=False)
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):
                        print(corrs)

                j += 1

if __name__ == "__main__":
    table_maker = TableMaker()

    dfs = table_maker.load_all_dfs(confounds=True)
    splits = load_splits()
    preprocessor = Preprocessor(dfs, splits)
    processed_data = preprocessor.load_datasets()

    cca = CCAPipeline(processed_data, preprocessor.timepoints)
    # cca.run_cca_train_and_val()

    qms = dfs["qms"][preprocessor.timepoints[0]].columns[1:].to_numpy()
    # cca.compare_scores_with_features(qms)

    cca.overall_cca(preprocessor.n_folds, qms)
    
    

    
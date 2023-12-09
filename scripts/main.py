from make_tables import *
from splitting import *
from processing import *
from sklearn.cross_decomposition import CCA
from pycirclize import Circos
from cca_zoo.deep import *
from cca_zoo.linear import SCCA_IPLS
import matplotlib.pyplot as plt
import torch

PLOT_FILEPATH = Path("plots")

name2color = {'ad': '#F261A8',
              'cgc': '#D27DFF',
              'ca': '#AB90F4',
              'dt': '#EB6065',
              'dla': '#A7DF8D',
              'fo': '#F6DE62',
              'n': '#D4D4D4',
              'rspltp': '#F2D3BA',
              'sa': '#848484',
              'smh': '#92E4E0',
              'smm': '#FE8855',
              'vta': '#4CB1B0',
              'vs': '#5E89F8',
              }

network_dict = OrderedDict([("au","ad"),
            ("cerc","cgc"),
            ("copa","ca"),
            ("df","dt"),
            ("dsa","dla"),
            ("fopa","fo"),
            # ("none","none"),
            ("rst","rspltp")])

class CCAPipeline():
    def __init__(self, processed_data, sub=False, plot_savepath=PLOT_FILEPATH):
        self.timepoints = processed_data.timepoints
        self.processed_datasets = processed_data.processed_datasets
        self.processed_data = processed_data
        self.plot_savepath = plot_savepath
        self.max_iter = 1000
        self.sub = sub
        self.cca_model = None
        self.X = None
        self.Y = None
        self.X_variates = None
        self.Y_variates = None
        self.X_loadings = None
        self.Y_loadings = None
        self.eps = 1e-3

    def _corrs(self, X, Y):
        '''Returns the correlation between corresponding columns of X and Y.
        X: (n, m)
        Y: (n, m)'''
        corrs = np.array([np.corrcoef(X[:, i], Y[:, i])[0, 1]\
                          for i in range(X.shape[1])])
        
        return corrs
    
    def get_loadings(self, X, Y, X_variates, Y_variates, mode=0):
        '''Return the correlation between the specified mode and every
        original feature for both X and Y'''
        X_first_mode = np.broadcast_to(np.expand_dims(X_variates[:, mode], 1), X.shape)
        X_loadings = self._corrs(X_first_mode, X)

        Y_first_mode = np.broadcast_to(np.expand_dims(Y_variates[:, mode], 1), Y.shape)
        Y_loadings = self._corrs(Y_first_mode, Y)

        return X_loadings, Y_loadings

    def cca(self, X, Y):
        '''Linear CCA between X and Y. The CCA model is saved
        and the correlations between the X and Y variates
        are returned'''
        n_comps = min(X.shape[1], Y.shape[1])
        cca_model = CCA(n_components=n_comps, max_iter=self.max_iter)

        self.cca_model = cca_model
        self.X = X
        self.Y = Y

        X_variates, Y_variates = cca_model.fit_transform(X, Y)
        self.X_variates = X_variates
        self.Y_variates = Y_variates

        X_loadings, Y_loadings = self.get_loadings(X, Y, X_variates, Y_variates)
        self.X_loadings = X_loadings
        self.Y_loadings = Y_loadings

        corrs = self._corrs(X_variates, Y_variates)

        return corrs
    
    def cca_train_val(self, data):
        ''''data' is assumed to be a subdictionary at the fold 
        level in the format of processed_data'''
        qms_train, qms_val = data["train"][0], data["val"][0]
        rsfc_train, rsfc_val = data["train"][1], data["val"][1]

        train_corrs = self.cca(qms_train, rsfc_train)
        qms_val_variates, rsfc_val_variates = self.cca_model.transform(qms_val, rsfc_val)
        val_corrs = self._corrs(qms_val_variates, rsfc_val_variates)

        qms_val_loadings, rsfc_val_loadings = self.get_loadings(qms_val, rsfc_val,
                                                                qms_val_variates, rsfc_val_variates)

        return train_corrs, val_corrs, qms_val_loadings, rsfc_val_loadings
    
    def train_crossval(self, timepoint=None, plot=True, datasets_at_timepoint=None):
        '''At a given timepoint, runs CCA at each fold 
        and returns training and validation correlations.'''
        if datasets_at_timepoint is None:
            datasets_at_timepoint = self.processed_datasets[timepoint]

        all_train_corrs = []
        all_val_corrs = []
        for fold, data in datasets_at_timepoint.items():
            if fold != "test":
                train_corrs, val_corrs, qms_val_loadings, rsfc_val_loadings = self.cca_train_val(data)
                all_train_corrs.append(train_corrs)
                all_val_corrs.append(val_corrs)
                print(f"Training correlation in the first mode at fold {fold}: {train_corrs[0]}")
                print(f"Validation correlation in the first mode at fold {fold}: {val_corrs[0]}")
                if plot:
                    self.plot_qms_loadings(self.X_loadings, self.processed_data.qms_labels,
                                           os.path.join(self.plot_savepath, 
                                                     f"crossval/{timepoint}/qms/train/{fold}.png"))
                    self.plot_qms_loadings(qms_val_loadings, self.processed_data.qms_labels,
                                           os.path.join(self.plot_savepath, 
                                                     f"crossval/{timepoint}/qms/val/{fold}.png"))
                    self.plot_rsfc_loadings(self.Y_loadings, self.processed_data.rsfc_variables,
                                            os.path.join(self.plot_savepath, 
                                                         f"crossval/{timepoint}/rsfc/train/{fold}.png"))
                    self.plot_rsfc_loadings(rsfc_val_loadings, self.processed_data.rsfc_variables,
                                            os.path.join(self.plot_savepath, 
                                                         f"crossval/{timepoint}/rsfc/val/{fold}.png"))
        all_train_corrs = np.stack(all_train_corrs)
        all_val_corrs = np.stack(all_val_corrs)

        return all_train_corrs, all_val_corrs

    def train_crossval_pipeline(self, plot_results=True, verbose=True):
        '''Runs CCA cross validation at each timepoint and captures all
        correlation results'''
        corrs = []
        for timepoint in self.timepoints:
            print(f"Doing cross-val at timepoint: {timepoint}")
            corrs.append(self.train_crossval(timepoint))

        if plot_results:
            self.plot_train_crossval_results(corrs)

        if verbose:
            for i in range(len(corrs)):
                # Get all training and val corrs at the given timepoint
                all_train_corrs = corrs[i][0]
                all_val_corrs = corrs[i][1]
                timepoint = self.timepoints[i]

                print(("Average correlation in the first mode on training data" 
                    f" at timepoint {timepoint} is {all_train_corrs.mean(axis=0)[0]}"))
                print(("Average correlation in the first mode on validation data" 
                    f" at timepoint {timepoint} is {all_val_corrs.mean(axis=0)[0]}"))
                
    def plot_train_crossval_results(self, corrs):
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
            
            ax.set_xlabel("Mode Index")
            ax.set_title(f"{self.timepoints[i]}")
            ax.legend()

        savepath = os.path.join(self.plot_savepath, "train_val_correlations")
        plt.savefig(savepath)   
        
    def pass_directions(self, source_time):
        '''Perform a CCA at one timepoint and pass canonical weights
        to other timepoints to evaluate performance at later timepoints.
        source_time: int'''
        source_timepoint = self.timepoints[source_time]
        source_data = self.processed_datasets[source_timepoint]
        _, all_val_corrs = self.train_crossval(source_timepoint)
        print(f"Correlation in the first mode on source data: {all_val_corrs.mean(axis=0)[0]}")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        modes = np.arange(all_val_corrs.shape[1])
        ax.plot(modes, all_val_corrs.mean(0), label=source_timepoint)

        remaining_timepoints = [timepoint for i, timepoint in enumerate(self.timepoints)\
                                 if i != source_time]
        for timepoint in remaining_timepoints:
            # The datasets we evaluate on are just all subjects not in the 
            # test set, which is exactly the training set for the test fold.
            _, val_corrs, _, _ = self.cca_train_val({"train": source_data["test"]["train"],
                                                     "val": self.processed_datasets[timepoint]["test"]["train"]})
            ax.plot(modes, val_corrs, label=timepoint)
            print(f"Correlation in the first mode at {timepoint}: {val_corrs[0]}") 

        ax.legend()
        ax.set_title(f"Correlation on validation data with canonical vectors from {source_timepoint}")
        ax.set_xlabel("Mode")
        plt.savefig(f"plots/pass_directions_{source_timepoint}.png")

    def plot_rsfc_loadings(self, rsfc_loadings, connections, savepath):
        if self.sub:
            networks = list(NETWORKS.keys()) + list(SUBCORTICAL.keys())
        else:
            networks = list(NETWORKS.keys())

        sectors = {network: 1 for network in networks}

        circos = Circos(sectors, space=2)
        for sector in circos.sectors:
            sector.text(sector.name, size=15)
            track = sector.add_track((95, 100))
            if sector.name in name2color:
                color = name2color[sector.name]
            else: 
                color = 'skyblue'

            track.axis(fc=color)
            #track.text(sector.name, color="#171717", size=12)
            track.xticks_by_interval(1)

        rsfc_loadings = pd.DataFrame({"connection": connections, "loading": rsfc_loadings})
        rsfc_loadings["abs_loading"] = np.abs(rsfc_loadings["loading"])
        rsfc_loadings = rsfc_loadings.sort_values("abs_loading", ascending=False)

        n_links = 20
        for i in range(n_links):
            connection, loading, abs_loading = rsfc_loadings.iloc[i, :].tolist()
            network_1 = connection.split("_")[-3]
            network_2 = connection.split("_")[-1]
            color = "#A2272C" if loading > 0 else "#145FB7"
            if network_1 in network_dict:
                network_1 = network_dict[network_1]

            if network_1 == network_2:
                circos.link((network_1, 0, abs_loading), (network_2, 1, 1 - abs_loading), color=color)
            else:
                circos.link((network_1, 0, abs_loading), (network_2, abs_loading, 0), color=color)

        fig = circos.plotfig()
        plt.savefig(savepath)

    def plot_qms_loadings(self, qms_loadings, qms_labels, savepath):
        fig, ax = plt.subplots(figsize=(15, 10))
        sorted_idxs = np.argsort(qms_loadings)

        colors = np.full((len(qms_loadings),), "skyblue")
        statics = np.isin(self.processed_data.qms_variables[sorted_idxs], self.processed_data.static_measures)
        colors[statics] = "lime"

        ax.barh(qms_labels[sorted_idxs], qms_loadings[sorted_idxs], color=colors)
        for i, loading in enumerate(qms_loadings[sorted_idxs]):
            if loading < 0:
                loading = 0
            ax.text(loading + 0.005, i, qms_labels[sorted_idxs[i]], va="center")
        ax.set_title("First Loading of Every Quantitative Measure")
        ax.set_yticks([])
        ax.set_xlabel("\nFirst Loading")
        plt.savefig(savepath)
            
    def loadings_at_timepoints(self):
        '''Plot loadings at each timepoint using the test fold'''
        for timepoint in self.timepoints:
            datasets_at_timepoint = self.processed_datasets[timepoint]
            qms_df = datasets_at_timepoint["test"]["train"][0]
            rsfc_df = datasets_at_timepoint["test"]["train"][1]

            self.cca(qms_df, rsfc_df)

            # Plot loadings
            self.plot_qms_loadings(self.X_loadings, self.processed_data.qms_labels,
                                   os.path.join(self.plot_savepath, f"overall/{timepoint}/qms.png"))
            self.plot_rsfc_loadings(self.Y_loadings, self.processed_data.rsfc_variables,
                                    os.path.join(self.plot_savepath, f"overall/{timepoint}/rsfc.png"))

    def scca(self, data):
        qms_train, qms_val = data["train"][0], data["val"][0]
        rsfc_train, rsfc_val = data["train"][1], data["val"][1]
        n_comps = min(qms_train.shape[1], rsfc_train.shape[1])

        model = SCCA_IPLS(latent_dimensions=1, alpha=[1e-1, 1e-2], early_stopping=True)
        X_variates, Y_variates = model.fit_transform([qms_train, rsfc_train])
        train_corrs = self._corrs(X_variates, Y_variates)

        X_variates_val, Y_variates_val = model.transform([qms_val, rsfc_val])
        val_corrs = self._corrs(X_variates_val, Y_variates_val)

        print(train_corrs)
        print(val_corrs)

        qms_weights = np.squeeze(model.weights[0])
        rsfc_weights = np.squeeze(model.weights[1])
        print("Selected qms vars")
        print(self.processed_data.qms_variables[qms_weights != 0])
        print("Selected rsfc vars")
        print(self.processed_data.rsfc_variables[rsfc_weights != 0])

        return train_corrs, val_corrs

    def deepcca(self, data, procedure="dcca", epochs=300):
        # Data
        qms_train, qms_val = data["train"][0], data["val"][0]
        rsfc_train, rsfc_val = data["train"][1], data["val"][1]

        n_comps = min(qms_train.shape[1], rsfc_train.shape[1])

        train_dataset = NumpyDataset((qms_train, rsfc_train))
        val_dataset = NumpyDataset((qms_val, rsfc_val))

        train_dataloader, val_dataloader = get_dataloaders(train_dataset, val_dataset,
                                                           batch_size=int(len(train_dataset) / 2))

        # Model
        dropout = 0.25
        lr = 1e-3
        weight_decay = 0

        qms_encoder = architectures.Encoder(latent_dimensions=n_comps,
                                            feature_size=qms_train.shape[1],
                                            layer_sizes=(1024, 1024),
                                            dropout=dropout)
        rsfc_encoder = architectures.Encoder(latent_dimensions=n_comps,
                                             feature_size=rsfc_train.shape[1],
                                             layer_sizes=(1024, 1024),
                                             dropout=dropout)

        if procedure == "dcca":
            dcca = DCCA(n_comps, encoders=[qms_encoder, rsfc_encoder])
        else:
            dcca = DCCA_EY(n_comps, encoders=[qms_encoder, rsfc_encoder])
    
        # Optimizer
        optimizer = torch.optim.Adam(dcca.parameters(), lr=lr, weight_decay=weight_decay)

        if procedure == "dcca":
            objective = CCALoss()

        torch.autograd.set_detect_anomaly(True)
        # Training
        best_validation_loss = float('inf')
        best_val_epoch = 0
        for epoch in range(epochs):
            for batch in train_dataloader:
                if procedure == "dcca":
                    representations = dcca(batch['views'])
                    loss = objective.loss(representations)
                else:
                    loss = dcca.loss(batch)["objective"]

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if epoch % 100 == 0:
                    print(f'Training loss: {loss}')
                if epoch == epochs - 1:
                    print("Training:")
                    X, Y = dcca(batch["views"])
                    X, Y = X.detach().numpy(), Y.detach().numpy()
                    cca_model = CCA(n_components=n_comps, max_iter=100)
                    X_variates, Y_variates = cca_model.fit_transform(X, Y)
                    train_corrs = self._corrs(X_variates, Y_variates)

                    print(train_corrs)

            # Evaluation
            for batch in val_dataloader:
                dcca.eval()
                if procedure == "dcca":
                    representations = dcca(batch['views'])
                    loss = objective.loss(representations)
                else:
                    loss = dcca.loss(batch)["objective"]

                if loss < best_validation_loss:
                    best_validation_loss = loss
                    best_val_epoch = epoch

                if epoch % 100 == 0:
                    print(f'Validation loss: {loss}')
                if epoch == epochs - 1:
                    print("Validation:")
                    X_val, Y_val = dcca(batch["views"])
                    X_val, Y_val = X_val.detach().numpy(), Y_val.detach().numpy()
                    X_val_variates, Y_val_variates = cca_model.transform(X_val, Y_val)
                    val_corrs = self._corrs(X_val_variates, Y_val_variates)
                    
                    print(val_corrs)

        print(f'The best validation loss was {best_validation_loss} and it occurred in epoch {best_val_epoch}')

# The below functions are from https://github.com/jameschapman19/cca_zoo/tree/main/cca_zoo/deep
# but were not made directly usable through the package

class NumpyDataset(torch.utils.data.Dataset):
    """
    Class that turns numpy arrays into a torch dataset
    """

    def __init__(self, views):
        """

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        """
        self.views = [view.astype(np.float32) for view in views]

    def __len__(self):
        return len(self.views[0])

    def __getitem__(self, index):
        views = [view[index] for view in self.views]
        return {"views": views}

def get_dataloaders(
    dataset,
    val_dataset=None,
    batch_size=None,
    val_batch_size=None,
    drop_last=True,
    val_drop_last=False,
    shuffle_train=False,
    pin_memory=True,
    num_workers=0,
    persistent_workers=True,
):
    """
    A utility function to allow users to quickly get hold of the dataloaders required by pytorch lightning

    :param dataset: A _CCALoss dataset used for training
    :param val_dataset: An optional _CCALoss dataset used for validation
    :param batch_size: batch size of train loader
    :param val_batch_size: batch size of val loader
    :param num_workers: number of workers used
    :param pin_memory: pin memory used by pytorch - True tends to speed up training
    :param shuffle_train: whether to shuffle training data
    :param val_drop_last: whether to drop the last incomplete batch from the validation data
    :param drop_last: whether to drop the last incomplete batch from the train data
    :param persistent_workers: whether to keep workers alive after dataloader is destroyed

    """
    if num_workers == 0:
        persistent_workers = False
    if batch_size is None:
        batch_size = len(dataset)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_train,
        persistent_workers=persistent_workers,
    )
    if val_dataset:
        if val_batch_size is None:
            val_batch_size = len(val_dataset)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            drop_last=val_drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        return train_dataloader, val_dataloader
    return train_dataloader

def inv_sqrtm(A, eps=1e-9, sigma=1e-3):
    """Compute the inverse square-root of a positive definite matrix."""
    # Add noise to matrix to hopefully allow taking the svd
    A = A + torch.normal(mean=torch.zeros(A.shape), std=torch.full(A.shape, sigma))

    # Perform eigendecomposition of covariance matrix
    U, S, V = torch.svd(A)
    # Enforce positive definite by taking a torch max() with eps
    S = torch.max(S, torch.tensor(eps, device=S.device))
    # Calculate inverse square-root
    inv_sqrt_S = torch.diag_embed(torch.pow(S, -0.5))
    # Calculate inverse square-root matrix
    B = torch.matmul(torch.matmul(U, inv_sqrt_S), V.transpose(-1, -2))
    return B

def _demean(views):
    return tuple([view - view.mean(dim=0) for view in views])
    
class CCALoss:
    """Differentiable CCA Loss. Extremely sensitive to learning rate and 
    other hyperparameters with a good chance of creating a non-positive-definite
    covariance matrix which prevents passage of the gradient when taking the inverse."""
    def __init__(self, eps: float = 1e-2):
        self.eps = eps

    def correlation(self, representations):
        """Calculate correlation."""
        latent_dims = representations[0].shape[1]
        o1 = representations[0].shape[1]
        o2 = representations[1].shape[1]

        representations = _demean(representations)

        SigmaHat12 = torch.cov(
            torch.hstack((representations[0], representations[1])).T
        )[:latent_dims, latent_dims:]
        SigmaHat11 = torch.cov(representations[0].T) + self.eps * torch.eye(
            o1, device=representations[0].device
        )
        SigmaHat22 = torch.cov(representations[1].T) + self.eps * torch.eye(
            o2, device=representations[1].device
        )

        SigmaHat11RootInv = inv_sqrtm(SigmaHat11, self.eps)
        SigmaHat22RootInv = inv_sqrtm(SigmaHat22, self.eps)

        Tval = SigmaHat11RootInv @ SigmaHat12 @ SigmaHat22RootInv
        trace_TT = Tval.T @ Tval
        eigvals = torch.linalg.eigvalsh(trace_TT)

        return eigvals

    def loss(self, views):
        """Calculate loss."""
        eigvals = self.correlation(views)
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)])
        return -eigvals.sum()

if __name__ == "__main__":
    torch.manual_seed(5)

    table_maker = TableMaker()
    dfs = table_maker.load_all_dfs(confounds=True)
    splits = load_splits()
    preprocessor = Preprocessor(dfs, splits, table_maker.static_measures)
    processed_data = preprocessor.load_datasets()

    cca = CCAPipeline(processed_data)

    # Perform linear CCA with cv at each timepoint and plot loadings and correlations
    cca.train_crossval_pipeline()

    # Plot loadings using test fold specifically
    cca.loadings_at_timepoints()

    # Pass canonical directions from one timepoint to the others
    for i in range(3):
        cca.pass_directions(i)

    # Perform Sparse CCA at one timepoint and fold
    data = processed_data.processed_datasets[TIMEPOINTS[0]]["0"]
    cca.scca(data)

    # Perform Deep CCA at one timepoint and fold
    # cca.deepcca(data)
    

    
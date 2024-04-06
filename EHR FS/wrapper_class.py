from abc import ABC, abstractmethod
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import fisher_score
import numpy as np
from gat_feat_selector.feature_selection import GATFeatSelector


class FeatureSelectionMethod(ABC):
    def __init__(self, n_features_to_select):
        self.n_features_to_select = n_features_to_select

    @abstractmethod
    def fit(self, X, y):
        pass

    def transform(self, X):
        return X[:, self.get_support()]

    @abstractmethod
    def get_support(self):
        pass


class MrmrWrapper(FeatureSelectionMethod):
    def __init__(self, n_features_to_select):
        super().__init__(n_features_to_select)
        self.num_features = None
        self.selected_idx = None

    def fit(self, X, y):
        self.num_features = X.shape[1]
        idx, _, _ = MRMR.mrmr(X, y, n_selected_features=self.n_features_to_select)
        self.selected_idx = idx
        return self

    def get_support(self):
        feat_array = np.full(self.num_features, False)
        feat_array[self.selected_idx] = True
        return feat_array


class FisherScoreWrapper(FeatureSelectionMethod):
    def __init__(self, n_features_to_select):
        super().__init__(n_features_to_select)
        self.num_features = None
        self.selected_idx = None

    def fit(self, X, y):
        self.num_features = X.shape[1]
        score = fisher_score.fisher_score(X, y)
        idx = fisher_score.feature_ranking(score)
        self.selected_idx = idx[0:self.n_features_to_select]
        return self

    def get_support(self):
        feat_array = np.full(self.num_features, False)
        feat_array[self.selected_idx] = True
        return feat_array


class GATFeatureSelectorWrapper(FeatureSelectionMethod):
    def __init__(self, num_classes, num_epochs, n_features_to_select, edge_per_node=10, dropout_rate=0.5,
                 hidden_channels=[64, 64, 64]):
        super().__init__(n_features_to_select)
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.edge_per_node = edge_per_node
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.num_features = None
        self.selected_idx = None

    def fit(self, X, y):
        self.num_features = X.shape[1]
        feat_selector = GATFeatSelector(self.num_classes, self.num_epochs, self.edge_per_node, self.dropout_rate,
                                        self.hidden_channels)
        feat_selector.process(X, y)
        idx = feat_selector.fit()
        self.selected_idx = idx[0:self.n_features_to_select]
        return self

    def get_support(self):
        feat_array = np.full(self.num_features, False)
        feat_array[self.selected_idx] = True
        return feat_array

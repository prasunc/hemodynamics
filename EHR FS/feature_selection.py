from enum import Enum
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from wrapper_class import MrmrWrapper, FisherScoreWrapper, GATFeatureSelectorWrapper
import numpy as np


class FeatureSelection(Enum):
    ANOVA = "ANOVA",
    MI = "Mutual information",
    SFS_forward = "Sequential feature selector (forward selection)",
    SFS_backward = "Sequential feature selector(backward selection)",
    RFE = "Recursive feature elimination",
    MRMR = "Minimal Redundancy Maximal Relevance",
    FisherScore = "Fisher score",
    GATFeatSelector = "Graph attention network for feature selection"


def feature_selection(method_name: FeatureSelection, X_train, X_test, y_train, y_test, num_classes, num_selected_feat):
    if method_name == FeatureSelection.ANOVA:
        model = SelectKBest(f_classif, k=num_selected_feat)
    elif method_name == FeatureSelection.MI:
        model = SelectKBest(mutual_info_classif, k=num_selected_feat)
    elif method_name == FeatureSelection.SFS_forward:
        model = SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
                                          n_features_to_select=num_selected_feat, direction='forward')
    elif method_name == FeatureSelection.SFS_backward:
        model = SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
                                          n_features_to_select=num_selected_feat, direction='backward')
    elif method_name == FeatureSelection.RFE:
        model = RFE(estimator=SVC(kernel="linear"), n_features_to_select=num_selected_feat, step=1)
    elif method_name == FeatureSelection.MRMR:
        model = MrmrWrapper(n_features_to_select=num_selected_feat)
    elif method_name == FeatureSelection.FisherScore:
        model = FisherScoreWrapper(n_features_to_select=num_selected_feat)
    elif method_name == FeatureSelection.GATFeatSelector:
        model = GATFeatureSelectorWrapper(num_classes=num_classes, num_epochs=400,
                                          n_features_to_select=num_selected_feat, edge_per_node=10,
                                          dropout_rate=0.5, hidden_channels=[64, 64, 64])
    else:
        raise Exception("This feature selection method is not still implemented")

    model = model.fit(X_train, y_train)

    X_train_selected = model.transform(X_train)
    X_test_selected = model.transform(X_test)

    classifier = SVC(kernel='rbf', probability=True)
    classifier.fit(X_train_selected, y_train)

    # y_pred = classifier.predict(X_test_selected)
    # accuracy = accuracy_score(y_test, y_pred)

    y_prob = classifier.predict_proba(X_test_selected)
    auc = roc_auc_score(y_test, y_prob[:, 1])

    print(np.where(model.get_support())[0])

    return model.get_support(), auc  # accuracy

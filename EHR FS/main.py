import numpy as np
from sklearn.model_selection import train_test_split
from feature_selection import feature_selection
from feature_selection import FeatureSelection
import scipy.io 
import pandas as pd


# Load the raw data and labels from csv file using pandas
raw_data = pd.read_csv('data.csv')
labels = pd.read_csv('labels.csv')




X_train, X_val, y_train, y_val = train_test_split(raw_data, labels, test_size=0.2)


feature_sizes = [5, 10, 15, 20, 25, 30, 35]


for feature_size in feature_sizes:
    print("Feature size: ", feature_size)
    for method in FeatureSelection:
        selected_features, auc = feature_selection(FeatureSelection[method.name], X_train, X_val, y_train, y_val
                                                   num_classes=num_classes, num_selected_feat=feature_size)
        print("Method:", method.name)
        print("Selected EHR features:", selected_features)
        print("AUC:", auc)
        print("-------------------------------------")
    print("\n\n")

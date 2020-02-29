import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

from SVM import SVMc


def run():
    X, y =  datasets.make_blobs(n_samples=100, n_features=2, centers = 2, cluster_std = 1.05, random_state = 123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state=123)
    model = SVMc()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    AUC = roc_auc_score(y_test, y_predicted)
    con_mat = confusion_matrix(y_test, y_predicted)
    return {"Model": model, "AUC": AUC, "Confusion Matrix": con_mat}

if __name__ == "__main__":
    run()

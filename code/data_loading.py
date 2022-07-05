import CONFIG
import pandas as pd
import numpy as np
import os


def load_data(dataset: str):
    X_train = np.array(pd.read_csv(
        os.path.join(CONFIG.DATA_BASE_PATH, dataset + "_train.data"),
        header=None, sep=" ").iloc[:, 0:-1])

    y_train = np.array(pd.read_csv(
        os.path.join(CONFIG.DATA_BASE_PATH, dataset + "_train.labels"),
        header=None)).ravel()

    X_test = np.array(pd.read_csv(
        os.path.join(CONFIG.DATA_BASE_PATH, dataset + "_valid.data"),
        header=None, sep=" ").iloc[:, 0:-1])

    return X_train, X_test, y_train



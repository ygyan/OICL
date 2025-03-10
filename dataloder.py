import numpy as np
import pandas as pd
import random

# ------------------------------- IHDP --------------------------------
def load_ihdp(dir="./dataset/IHDP/", seed=0):
    data = pd.read_csv(dir + "ihdp_npci_{}.csv".format(seed))
    x_cols = ["x"+str(num) for num in range(1, 26)]

    # get test data
    X = data.loc[:, x_cols].values
    T, Y = data["t"].values, data["y"].values
    mu0, mu1 = data["mu0"].values, data["mu1"].values
    gt_ate = mu1.mean() - mu0.mean()
    X_t, X_c, Y_t, Y_c = X[T == 1, :], X[T == 0, :], Y[T == 1], Y[T == 0]

    return X_t, X_c, Y_t, Y_c, gt_ate

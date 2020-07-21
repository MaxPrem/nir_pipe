from sys import stdout

import matplotlib.collections as collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import f
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from ImportModule import cut_specs, importLuzCol
from ValidationUtils import cross_table, benchmark_model



def pls_crossval(X, y, n_comp):

    opt_comp = optimal_n_comp(X, y, n_comp)

    opt_model = pls_regression(X,y, opt_comp)

    pls_scores(X,y, opt_model)


#specs = pd.read_csv("./luzrawSpectra/nirMatrix.csv")  # cut spectra
specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")  # full spectra
lab = pd.read_excel("/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx")



# input wavenumber to specs = cut_specs(specs, 4100, 5500)
# specs = cut_specs(specs, 4100, 5500)


X, y, wl, ref = importLuzCol(specs, lab, 2)



# splitting dataset
"""to be continued with test set"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#########################
# scaling and transformingfrom ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter

# scale y
y_scaler = GlobalStandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# needing unscaled


pipeline = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
        # ("variable_selection", Enet_Select())
    ]
)


pip = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
    ]
)

X_train_pip = pip.fit_transform(X_train)
X_test_pip = pip.transform(X_test)


# transforming only the spectra

""""""
# X_train = pipeline.fit_transform(X_train)
X_train_sel = pipeline.fit_transform(X_train, y_train)
X_test_sel = pipeline.transform(X_test)


#######


pls_crossval(X_train, y_train, 10)

from ValidationUtils import cal_scores, benchmark_table
cal_scores()
# MSECV
benchmark_model(X_train_sel, y_train, X_test_sel, y_test, y, ref, sel_model)

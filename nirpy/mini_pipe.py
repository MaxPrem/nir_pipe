import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#import

specs = pd.read_csv('Users/maxprem/nirpy/luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv') # full spectra
lab = pd.read_excel('/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx')





from import_Module import importLuzCol, cut_specs

# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)
#specs = cut_specs(specs, 4100, 5500)



X, y, wl = importLuzCol(specs, lab, 3)


from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from pls_utils import pls_opt_cv, pls_cv

# splitting dataset
"""to be continued with test set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#########################
# scaling and transformingfrom ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from sklearn.cross_decomposition import  PLSRegression
from pls_utils import PLSOptimizer, Outlier
from enet_var import Enet_Select

# scale y
y_scaler = GlobalStandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


# pipeline = Pipeline([
#     ("scaleing_X", GlobalStandardScaler()),
#     ("scatter_correction", EmscScaler()),
#     ("smmothing", SavgolFilter(polyorder=2,deriv=0)),
#     ("variable_selection", Enet_Select())
# ])


pipeline = Pipeline([
    ("scaleing_X", GlobalStandardScaler()),
    ("scatter_correction", EmscScaler()),
    ("smmothing", SavgolFilter(polyorder=2,deriv=0))
])




# transforming only the spectra

''''''
#X_train = pipeline.fit_transform(X_train)

X_train = pipeline.fit_transform(X_train)

X_test = pipeline.transform(X_test)


model = pls_opt_cv(X_train, y_train, 20)


###########################################
###########################################
###########################################

###########################################
###########################################

from validation_utils import da_func_ncv
da_func_ncv(X_train, y_train, X_test, y_test, model)
#
# # %% markdown
# 7 comps
# RMSE  Train/Test	0.23	0.24
# Huber Train/Test	0.0253	0.0280
#
#
# Huber2 Train/Test	0.0253	0.0280
# HuberCV Train/Test	0.0425	0.0454
# ##################################
# R2 calib Train/Test	0.9477	0.9331
# R2 CV Train/Test	0.9101	0.8905
# MSE calib Train/Test	0.0523	0.0576
# MSE CV Train/Test	0.0899	0.0943
#
# #2 componets
# RMSE  Train/Test	0.35	0.36
# Huber Train/Test	0.0548	0.0607
#
#
# Huber2 Train/Test	0.0548	0.0607
# HuberCV Train/Test	0.0607	0.0718
# ##################################
# R2 calib Train/Test	0.8806	0.8498
# R2 CV Train/Test	0.8673	0.8217
# MSE calib Train/Test	0.1194	0.1293
# MSE CV Train/Test	0.1327	0.1535
#
# ######### three components ##########
# RMSE  Train/Test	0.31	0.35
# Huber Train/Test	0.0451	0.0585
#
#
# Huber2 Train/Test	0.0451	0.0585
# HuberCV Train/Test	0.0550	0.0778
# ##################################
# R2 calib Train/Test	0.9020	0.8557
# R2 CV Train/Test	0.8788	0.8005
# MSE calib Train/Test	0.0980	0.1242
# MSE CV Train/Test	0.1212	0.1718
#
# # %% codecell
#
#
# from scipy.stats import sem
#
# from scipy import stats
# >>> a = np.arange(20).reshape(5,4)
# >>> stats.sem(a)
# a
# stats.sem(a, axis=None, ddof=0)
#
#
# y_c_test = model.predict(X_test)

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


#import

# specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv') # full spectra
lab = pd.read_excel('/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx')




from import_Module import importLuzCol, cut_specs

# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 8000)
#specs = cut_specs(specs, 4100, 5500)



X, y, wl, ref = importLuzCol(specs, lab, 4)


from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from pls_utils import pls_opt_cv, pls_cv

# splitting dataset
"""to be continued with test set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#########################
# scaling and transformingfrom ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from sklearn.cross_decomposition import  PLSRegression
from pls_utils import PLSOptimizer, Outlier
from enet_var import Enet_Select

# scale y
y_scaler = GlobalStandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


pipeline = Pipeline([
    ("scaleing_X", GlobalStandardScaler()),
    ("scatter_correction", EmscScaler()),
    ("smmothing", SavgolFilter(polyorder=2,deriv=0))
    ])




# transforming only the spectra


X_train = pipeline.fit_transform(X_train, y_train)

X_test = pipeline.transform(X_test)

model = pls_opt_cv(X_train, y_train, 4)

# #from validation_utils import cv_benchmark
#
#
#
# from math import sqrt
# cv_benchmark(X_train, y_train, X_test, y_test, model)
#
# from pls_utils import Outlier
#
# pls_out = Outlier()
# pls_out.fit(X_train, y_train, n_comps=5)
# pls_out.plot(X_train, y_train)
# X_out, y_out = pls_out.transform(X_train, y_train, 40)
# ##########################################
# # ElassticNet Variable_Selection_Pipeline#
# ##########################################
# pls_opt_cv(X_out, y_out, 6)
#
#
#
#
# _ = plt.plot(wl, X_out.T)
# X_train.shape
# X_out.shape
# var_sel = Enet_Select()
# X_out_sel, y_out_sel = var_sel.fit_transform(X_out, y_out)
#
#
# #######################
# #Full Pre-Pro-Pipeline#
# #######################
# pipe_sel = Pipeline([
#     ("scaleing_X", GlobalStandardScaler()),
#     ("scatter_correction", EmscScaler()),
#     ("smmothing", SavgolFilter(polyorder=2,deriv=0)),
#     ("variable_selection", Enet_Select()),
# ])
#
#
# X_train_sel = pipe_sel.fit_transform(X_out, y_out)
# X_test_sel = pipe_sel.transform(X_test)
#
# X_train_sel.shape
# X_test_sel.shape
#
#
# model = pls_cv(X_train_sel, y_train, 3)
#
# cv_benchmark(X_train_sel, y_train, X_test_sel, y_test, model)
# # after preprocessing and removing the noisy half of the speectra
# # the 2 component pls model perfoms well
#
# # lets try to select some variables
#
#
#
#
#
# # #############################################################################
# # Plot results functions
#
# # from sklearn.linear_model import ElasticNet
# #
# # model_enet = ElasticNet(alpha = 0.01)
# # model_enet.fit(X_train, y_train)
# # pred_train_enet= model_enet.predict(X_train)
# # print(np.sqrt(mean_squared_error(y_train,pred_train_enet)))
# # print(r2_score(y_train, pred_train_enet))
# #
# # pred_test_enet= model_enet.predict(X_test)
# # print(np.sqrt(mean_squared_error(y_test,pred_test_enet)))
# # print(r2_score(y_test, pred_test_enet))
#
# ######################
# #
# #
# #
# #
# # identitypls_opt = PLSOptimizer()
# # pls_opt.fit(X, y, max_comp=15)
# # pls_opt.plot()
# # pls_opt.predict(n_comp=4)
# #
# #
# # pls_pipe = PLSOptimizer()
# # pls_pipe.fit(X_train,y_train, 5)
# # pls_pipe.plot()
# # pls_pipe.predict()
#
#
#
#
#
# #benchmark(X_train[:,wav:], y_train, X_test[:,wav:], y_test, model)
# cv_benchmark(X_train, y_train, X_test, y_test, model)

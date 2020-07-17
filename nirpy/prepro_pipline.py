# %%
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#########################
# scaling and transformingfrom ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from enet_var import Enet_Select
from ImportModule import cut_specs, importLuzCol
from pls_utils import Outlier, PLSOptimizer, pls_cv, pls_opt_cv
from validation_utils import da_func_ncv

#import

# specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv') # full spectra
lab = pd.read_excel('/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx')


# %%


# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)
#specs = cut_specs(specs, 4100, 5500)

'''Change the Number in the import function to load another lab reference method'''

X, y, wl, ref = importLuzCol(specs, lab, 4)



# splitting dataset
"""to be continued with test set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# scale y
y_scaler = GlobalStandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)



pipeline = Pipeline([
    ("scaleing_X", GlobalStandardScaler()),
    ("scatter_correction", EmscScaler()),
    ("smmothing", SavgolFilter(polyorder=2,deriv=0)),
    #("variable_selection", Enet_Select())
])

#
# pipeline = Pipeline([
#     ("scaleing_X", GlobalStandardScaler()),
#     ("scatter_correction", EmscScaler()),
#     ("smmothing", SavgolFilter(polyorder=2,deriv=0)),
#     ("variable_selection", Enet_Select())
# ])


#https://nbviewer.jupyter.org/github/WillKoehrsen/Data-Analysis/blob/master/prediction-intervals/prediction_intervals.ipynb
#tol=1e-06


''''''
#X_train = pipeline.fit_transform(X_train)

X_train = pipeline.fit_transform(X_train, y_train)

X_test = pipeline.transform(X_test)
#pipeline['variable_selection'].plot(wl, x_pip)

# %% codecell

model = pls_opt_cv(X_train, y_train, 9)


da_func_ncv(X_train, y_train, X_test, y_test,y, ref, model)


''' conf intervall enet = tol?'''


#• Simple non-parametric interval estimation by calibrating error estimates produces intervals with coverage close to the desired level of confidence. We recommend it as a pragmatic method for interval estimation.


#
#
#
#
# %load_ext autoreload
# %autoreload 2
#
# from pls_utils import Outlier
#
# pls_out = Outlier(n_comps=4)
# pls_out.fit(X_train, y_train)
# pls_out.plot(X_train, y_train)
# X_out, y_out = pls_out.transform(X_train, y_train, 20)
#
#
# pls_opt_cv(X_out, y_out, 20)

# ##########################################
# # ElassticNet Variable_Selection_Pipeline#
# ##########################################
# pls_opt_cv(X_out, y_out, 6)
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


# %%

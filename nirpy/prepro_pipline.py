# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from CrossValPLS import optimal_n_comp, pls_regression, pls_scores
from ElasticNetVariableSelection import EnetSelect, MultiTaskElasticNet
from ImportModule import cut_specs, importLuzCol
from ValidationUtils import print_regression_benchmark, print_cv_table, cv_benchmark_model, val_regression_plot

# %%
# impor data

# specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")  # full spectra
lab = pd.read_excel("/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx")
# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 6500)
# specs = cut_specs(specs, 4100, 5500)

# %%
# assing spectra and refernce method


X, y, wl, ref = importLuzCol(specs, lab, 3)

# split dataset in train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

# %%
# storeing data in a dictonary to pass  as function parameter with **data
data = {"X": X_train, "y": y_train, "X_test": X_test, "y_test": y_test}



# %%
#transformation pipeline

# scale y
y_scaler = GlobalStandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


pipeline = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
        #("variable_selection", EnetSelect()),
    ]
)
# to perform variable selection y values are corrleated in the fit method
X_train_en = pipeline.fit_transform(X_train, y_train)
X_test_en = pipeline.transform(X_test)

data_en = {"X": X_train_en, "y": y_train, "X_test": X_test_en, "y_test": y_test}
# %%
#transformation pipeline


variable_selection_pip = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
        #("variable_selection", EnetSelect()),
    ]
)
# to perform variable selection y values are corrleated in the fit method
X_train_pip = variable_selection_pip.fit_transform(X_train, y_train)
X_test_pip = variable_selection_pip.transform(X_test)

data_pip = {"X": X_train_pip, "y": y_train, "X_test": X_test_pip, "y_test": y_test}
# %%
# pipeline['variable_selection'].plot(wl, X)

# %% codecell

opt_comp = optimal_n_comp(**data_pip, n_comp=20, plot=True)

def pls_crossval(X, y, n_comp = 10, plot=False, **kargs):
    """returns a model with the optimsed number of pls components"""

    # returns n_comp with lowest loss
    opt_comp = optimal_n_comp(X, y, n_comp, plot=plot)
    # performs regression with n_comp
    opt_model = pls_regression(X,y, opt_comp, plot=plot)
    # returns regression scores
    pls_scores(X,y, opt_model)

    return opt_model

# %%
model_en = pls_crossval(**data_en, n_comp = 10)


model_pip = pls_crossval(**data_pip, n_comp =10)
# %%



cv_benchmark_model(**data_en, y_unscaled=y, ref=ref,  model = model_en)
# %%
cv_benchmark_model(**data_pip, y_unscaled=y, ref=ref, model = model_pip)

# %%



 print_cv_table(**data_pip, model = model_pip)

# %%
print_cv_table(**data_en, model = model_en)

# benchmark_table(X_train, y_train, X_test, y_test, model)


# %% markdown

# • Simple non-parametric interval estimation by calibrating error estimates produces intervals with coverage close to the desired level of confidence. We recommend it as a pragmatic method for interval estimation.


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

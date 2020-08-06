import pandas as pd
# %%
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from CrossValPLS import (extra_plot_mse, extra_plot_variance_explained,
                         mse_minimum, pls_regression, pls_scores,
                         variance_explained, optimal_n_comp)
from ElasticNetVariableSelection import EnetSelect
from ImportModule import cut_specs, importLuzCol
from ValidationUtils import (cv_benchmark_model, print_cv_table, val_regression_plot)
from PlsOutlier import Outlier

# %%
# impor data

# specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")  # full spectra
lab = pd.read_excel("/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx")
# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)

# specs = cut_specs(specs, 4100, 5500)

# %%

X, y, wave_number, ref = importLuzCol(specs, lab, 2)

# split dataset in train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# %%
# transformation pipeline

# scale y
y_scaler = GlobalStandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


pip_dev0 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
        #("variable_selection", EnetSelect())
    ]
)

X_train_0 = pip_dev0.fit_transform(X_train, y_train)
X_test_0 = pip_dev0.transform(X_test)

# %%

variance_explained(X_train_0, y_train)


# %%
mse_minimum(X_train_0, y_train)

# %%

def pls_crossval(X, y, n_comp, **kwargs):
    # fits a pls model with a given number of components
    model = pls_regression(X, y, n_comp)
    # calculated score and mse from given model
    pls_scores(X, y, model)

    return model

pls_crossval(X_train_0, y_train, 6)

# %%

rem_outlier = Outlier()
rem_outlier.fit(X = X_train_0, y = y_train, n_comp = 2)
rem_outlier.plot(X_train_0, y_train)
X_rem, y_rem = rem_outlier.fit_transform(X_train_0, y_train, 20)
# %%
rem_outlier.plot(X_rem, y_rem)

# %%

rem_outlier2 = Outlier()
rem_outlier2.fit(X = X_rem, y = y_rem, n_comp = 2)
rem_outlier2.plot(X_train_0, y_train)
Xx, yy = rem_outlier2.fit_transform(X_train_0, y_train,20)
# %%
variance_explained(X_rem, y_rem)


# %%
mse_minimum(X_rem, y_rem)

# %%
pls_crossval(X_rem, y_rem, n_comp= 6)
# %%
pls_crossval(Xx, yy, n_comp= 6)

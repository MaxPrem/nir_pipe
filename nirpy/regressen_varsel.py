# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from CrossValPLS import (extra_plot_mse, extra_plot_variance_explained,
                         mse_minimum, pls_regression, pls_scores,
                         variance_explained, optimal_n_comp)
from ElasticNetVariableSelection import EnetSelect
from ImportModule import cut_specs, importLuzCol
from ValidationUtils import (cv_benchmark_model, print_cv_table, val_regression_plot)

# %%
# impor data

# specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")  # full spectra
lab = pd.read_excel("/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx")
# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)

# specs = cut_specs(specs, 4100, 5500)

# %%

X, y, wave_number, ref = importLuzCol(specs, lab, 4)

# split dataset in train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
        ("smmothing", SavgolFilter(polyorder=2, deriv=1)),
        #("variable_selection", EnetSelect())
    ]
)

X_train_0 = pip_dev0.fit_transform(X_train, y_train)
X_test_0 = pip_dev0.transform(X_test)

# %%

var, comp = variance_explained(X_train_0, y_train, plot=False)
var_2, comp_2 = variance_explained(X_test_0, y_test, plot=False)
extra_plot_variance_explained(var, comp, var_2, comp_2)

mse, comp = mse_minimum(X_train_0, y_train, plot=False)
mase_2, comp_2 = mse_minimum(X_test_0, y_test, plot=False)
extra_plot_mse(mse, comp, mase_2, comp_2)

# %%

def pls_crossval(X, y, n_comp, **kwargs):

    opt_comp = optimal_n_comp(X, y, n_comp)

    opt_model = pls_regression(X, y, opt_comp)

    pls_scores(X, y, opt_model)

    return opt_model

pls_crossval(X_train_0, y_train, 10)
# %%

from PLSfunctions import PLSOptimizer

pls_opt = PLSOptimizer()

pls_opt.fit(X_train_0, y_train, 2)
pls_opt.plot(wave_number, X_train_0)
X_train_pls = pls_opt.transform(X_train_0)
X_test_pls = pls_opt.transform(X_test_0)

# %%

data = {"X": X_train_pls, "y": y_train, "X_test": X_test_pls, "y_test": y_test}

model = pls_regression(**data, n_comp=2)

cv_benchmark_model(**data ,y_unscaled=y, ref=ref, model = model)

# %%

var, comp = variance_explained(X_train_pls, y_train, plot=False)
var_2, comp_2 = variance_explained(X_test_pls, y_test, plot=False)
extra_plot_variance_explained(var, comp, var_2, comp_2)

mse, comp = mse_minimum(X_train_pls, y_train, plot=False)
mase_2, comp_2 = mse_minimum(X_test_pls, y_test, plot=False)
extra_plot_mse(mse, comp, mase_2, comp_2)

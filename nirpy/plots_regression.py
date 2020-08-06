# %%
from ValidationUtils import val_regression_plot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from CrossValPLS import mse_minimum, pls_regression, pls_scores, variance_explained
from ElasticNetVariableSelection import EnetSelect
from ImportModule import cut_specs, importLuzCol
from PLSfunctions import PLSOptimizer
from ValidationUtils import cv_benchmark_model

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

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
        ("variable_selection", EnetSelect()),
    ]
)
# to perform variable selection y values are corrleated in the fit method
X_train_0 = pip_dev0.fit_transform(X_train, y_train)
X_test_0 = pip_dev0.transform(X_test)

data_en0 = {"X": X_train_0, "y": y_train, "X_test": X_test_0, "y_test": y_test}
# # %%
# pip_dev0["variable_selection"].plot(wave_number, X_0)
# # %%
# pip_dev0["variable_selection"].plot_feature_importance(wave_number)

# %%
def pls_crossval(X, y, n_comp, **kwargs):

    opt_comp = optimal_n_comp(X, y, n_comp)
    variance_explained(X, y, n_comp)

# %%
pls_crossval(**data_en0, n_comp=15)

# %%

model_0 = pls_regression(**data_en0, n_comp=3, plot=False)

cv_benchmark_model(**data_en0, model=model_0, y_unscaled=y, ref=ref, plot=True)


val_regression_plot(**data_en0, model = model_0)


# %%
pls_opt_0 = PLSOptimizer()

pls_opt_0.fit(X_train_0, y_train, max_comp=2)
# transformation pipeline
# %%

pip_dev1 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=1)),
        ("variable_selection", EnetSelect()),
    ]
)
# to perform variable selection y values are corrleated in the fit method
X_train_1 = pip_dev1.fit_transform(X_train, y_train)
X_test_1 = pip_dev1.transform(X_test)

data_en1 = {"X": X_train_1, "y": y_train, "X_test": X_test_1, "y_test": y_test}

X.shape

# %%


pls_crossval(**data_en1, n_comp=15)
model_1 = pls_regression(**data_en1, n_comp=2, plot=False)
cv_benchmark_model(**data_en1, model=model_1, y_unscaled=y, ref=ref)
val_regression_plot(**data_en0, model = model_1)

# %%
pls_opt_1 = PLSOptimizer()

pls_opt_1.fit(X_train_1, y_train, max_comp=2)
# %%
# transformation pipeline


pip_dev3 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=2)),
        ("variable_selection", EnetSelect()),
    ]
)
# to perform variable selection y values are corrleated in the fit method
X_train_2 = pip_dev3.fit_transform(X_train, y_train)
X_test_2 = pip_dev3.transform(X_test)

data_en2 = {"X": X_train_2, "y": y_train, "X_test": X_test_2, "y_test": y_test}
# %%
X.shape[1]
pls_crossval(**data_en2, n_comp=15)
model_2 = pls_regression(**data_en2, n_comp=4)
cv_benchmark_model(**data_en2, model=model_2, y_unscaled=y, ref=ref)

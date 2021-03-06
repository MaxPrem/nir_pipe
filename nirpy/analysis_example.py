# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from CrossValPLS import (extra_plot_mse, extra_plot_variance_explained, mse_minimum, pls_regression, pls_scores, variance_explained)
from ElasticNetVariableSelection import EnetSelect
from ImportModule import cut_specs, importLuzCol
from ValidationUtils import (cv_benchmark_model, print_cv_table, val_regression_plot)
from pathlib import Path
# %%
specs_path = Path("../calData_full.csv")
lab_patth = Path("../luzrawSpectra/labData.xlsx")

specs  = pd.read_csv(specs_path)
lab = pd.read_excel(lab_patth)
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
    ]
)

X_train_0 = pip_dev0.fit_transform(X_train, y_train)
X_test_0 = pip_dev0.transform(X_test)


data_en0 = {"X": X_train_0, "y": y_train, "X_test": X_test_0, "y_test": y_test}
# %%
# variable selection on whole data

# scale y
y_scaler = GlobalStandardScaler()
y_scaled = y_scaler.fit_transform(y)



pip_sel = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
        ("variable_selection", EnetSelect())
    ]
)

X_sel = pip_sel.fit_transform(X, y_scaled)

#pip_sel["variable_selection"].plot_feature_importance(wave_number)
#pip_sel["variable_selection"].plot(wave_number, X_train_0)


# %%
# to perform variable selection from regression with whole dataset

X_train_0_sel = pip_sel["variable_selection"].transform(X_train_0)
X_test_0_sel = pip_sel["variable_selection"].transform(X_test_0)

data_en_sel = {"X": X_train_0_sel, "y": y_train, "X_test": X_test_0_sel, "y_test": y_test}







# %%
# variance explained and MSECV for train and test set for each PLS component
var, comp = variance_explained(X_train_0, y_train, plot=False)
var_2, comp_2 = variance_explained(X_test_0, y_test, plot=False)
extra_plot_variance_explained(var, comp, var_2, comp_2)

mse, comp = mse_minimum(X_train_0, y_train, plot=False)
mase_2, comp_2 = mse_minimum(X_test_0, y_test, plot=False)
extra_plot_mse(mse, comp, mase_2, comp_2)
# %%
# variance explained and MSECV for train and test set for each component
var, comp = variance_explained(X_train_0_sel, y_train, plot=False)
var_2, comp_2 = variance_explained(X_test_0_sel, y_test, plot=False)
extra_plot_variance_explained(var, comp, var_2, comp_2)

mse, comp = mse_minimum(X_train_0, y_train, plot=False)
mase_2, comp_2 = mse_minimum(X_test_0, y_test, plot=False)
extra_plot_mse(mse, comp, mase_2, comp_2)


# %%

model_0 = pls_regression(**data_en0, n_comp=5, plot=False)
print_cv_table(**data_en0, model=model_0)


# %%

cv_benchmark_model(**data_en0, model=model_0, y_unscaled=y, ref=ref, plot=True)
#val_regression_plot(**data_en0, model=model_0)
#val_regression_plot(**data_en_sel, model=model_sel)

# %%

model_sel = pls_regression(**data_en_sel, n_comp=2, plot=False)
print_cv_table(**data_en_sel, model=model_sel)

cv_benchmark_model(**data_en_sel, y_unscaled=y, ref=ref, model=model_sel)
#%%

cv_benchmark_model(**data_en_sel, y_unscaled=y, ref=ref, model=model_en_sel)

# %%
from PLSfunctions import PLSOptimizer
pls_opt = PLSOptimizer()
pls_opt.fit(**data_en_sel, max_comp=10)

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from ElasticNetVariableSelection import EnetSelect
from ImportModule import cut_specs, importLuzCol

from PLSfunctions import PLSOptimizer

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
    X, y, test_size=0.1, random_state=42
)

# %%


dev_0 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler(with_std=False)),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
    ]
)

X_0 = dev_0.fit_transform(X)

# %%


dev_1 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler(with_std=False)),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=1)),
    ]
)

X_1 = dev_1.fit_transform(X)


# %%


dev_2 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler(with_std=False)),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=2)),
    ]
)

X_2 = dev_2.fit_transform(X)


# %%
# transformation pipeline

# scale y
y_scaler = GlobalStandardScaler(with_std=False)
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


pip_dev0 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler(with_std=False)),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
        ("variable_selection", EnetSelect())
    ]
)
# to perform variable selection y values are corrleated in the fit method
X_train_0 = pip_dev0.fit_transform(X_train, y_train)
X_test_0 = pip_dev0.transform(X_test)

data_en0 = {"X": X_train_0, "y": y_train, "X_test": X_test_0, "y_test": y_test}
# %%
pip_dev0["variable_selection"].plot(wave_number, X_0)
# %%
pip_dev0["variable_selection"].plot_feature_importance(wave_number)


# %%
pls_opt_0 = PLSOptimizer()

pls_opt_0.fit(X_train_0, y_train, max_comp=2)
# transformation pipeline
# %%

pip_dev1 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler(with_std=False)),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=1)),
        ("variable_selection", EnetSelect())
    ]
)
# to perform variable selection y values are corrleated in the fit method
X_train_1 = pip_dev1.fit_transform(X_train, y_train)
X_test_1 = pip_dev1.transform(X_test)

data_en1 = {"X": X_train_1, "y": y_train, "X_test": X_test_1, "y_test": y_test}

X.shape
# %%
pip_dev1["variable_selection"].plot_feature_importance(wave_number)
# %%
pip_dev1["variable_selection"].plot(wave_number, X_1)
# %%

# %%
pls_opt_1 = PLSOptimizer()

pls_opt_1.fit(X_train_1, y_train, max_comp=2)
# %%
# transformation pipeline


pip_dev3 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler(with_std=False)),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=2)),
        ("variable_selection", EnetSelect())
    ]
)
# to perform variable selection y values are corrleated in the fit method
X_train_2 = pip_dev3.fit_transform(X_train, y_train)
X_test_2 = pip_dev3.transform(X_test)

data_en1 = {"X": X_train_2, "y": y_train, "X_test": X_test_2, "y_test": y_test}
# %%
pip_dev3["variable_selection"].plot_feature_importance(wave_number)
# %%
pip_dev3["variable_selection"].plot(wave_number, X_2)

# %%
pls_opt_2 = PLSOptimizer()

pls_opt_2.fit(X_train_2, y_train, max_comp=2)

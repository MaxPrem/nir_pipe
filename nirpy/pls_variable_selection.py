# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from ImportModule import importLuzCol, plot_spec, cut_specs
from PLSfunctions import PLSOptimizer
from CrossValPLS import optimal_n_comp, pls_regression, pls_scores
from ValidationUtils import cv_benchmark_model

# %%
# impor data

# specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")  # full spectra
lab = pd.read_excel("/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx")

# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)


# %%
# assing spectra and refernce method


X, y, wave_number, feat_ref = importLuzCol(specs, lab, 4)

# split dataset in train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# %%

plot_spec(wave_number, X)

# %%

# scale y
y_scaler = GlobalStandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)



dev_0 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=1))
    ]
)

X_train_0 = dev_0.fit_transform(X_train)
X_test_0 = dev_0.fit_transform(X_test)

plot_spec(wave_number, X_train_0)
# %%
pls_opt = PLSOptimizer()
pls_opt.fit(X_train_0, y_train, max_comp=20)
pls_opt.plot(wave_number, X_train_0)
# %%
X_train_sel = pls_opt.transform(X_train_0)
X_test_sel = pls_opt.transform(X_test_0)


data_sel = {"X": X_train_sel, "y": y_train, "X_test": X_test_sel, "y_test": y_test}
# %%
def pls_crossval(X, y, n_comp, **kwargs):

    opt_comp = optimal_n_comp(X, y, n_comp)

    opt_model = pls_regression(X, y, opt_comp)

    pls_scores(X, y, opt_model)

    return opt_model

pls_crossval(X_train_sel, y_train, 10)
model = pls_regression(**data_sel, n_comp=2)

cv_benchmark_model(**data_sel, model = model,y_unscaled= y,  ref=feat_ref)

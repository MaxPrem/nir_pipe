# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from CrossValPLS import optimal_n_comp, pls_regression, pls_scores, variance_explained
from ElasticNetVariableSelection import EnetSelect
from ImportModule import cut_specs, importLuzCol
from ValidationUtils import print_regression_benchmark, print_cv_table, cv_benchmark_model, print_nir_metrics

# %%
# impor data

# specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")  # full spectra
lab = pd.read_excel("/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx")
# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)
# specs = cut_specs(specs, 4100, 5500)

# %%
lab
lab.set_index("Code").sort_index()
lab["Code"]
# assing spectra and refernce method


X, y, wl, ref = importLuzCol(specs, lab, 4)

# split dataset in train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# %%


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
        ("variable_selection", EnetSelect()),
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
    ]
)
# to perform variable selection y values are corrleated in the fit method
X_train_pip = variable_selection_pip.fit_transform(X_train, y_train)
X_test_pip = variable_selection_pip.transform(X_test)

data_pip = {"X": X_train_pip, "y": y_train, "X_test": X_test_pip, "y_test": y_test}
# %%
pipeline['variable_selection'].plot(wl, X)

# %% codecell

opt_comp = optimal_n_comp(**data_pip, n_comp=12, plot=True)



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

variance_explained(X_train_en, y_train, n_comp=20, plot=True)
variance_explained(X_train_pip, y_train, n_comp=20, plot=True)

# %%




# %%

model_en = pls_regression(**data_en, n_comp =3)


model_pip = pls_regression(**data_pip, n_comp =2)

# %%
 def print_regression_table(X, y, X_test, y_test, model, y_unscaled, ref):

     print_nir_metrics(X, y, X_test, y_test, model, y_unscaled, ref)
     print_regression_benchmark(X, y, X_test, y_test, model)
     print_cv_table(X, y, X_test, y_test, model)

print_regression_table(**data_en, model=model_en, y_unscaled=y, ref=ref)

# %%

 print_regression_table(**data_pip, model = model_pip, y_unscaled = y, ref=ref)

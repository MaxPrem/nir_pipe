# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from CrossValPLS import optimal_n_comp, pls_regression, pls_scores
from ElasticNetVariableSelection import EnetSelect, MultiTaskElasticNet
from ImportModule import cut_specs, importLuzCol
from ValidationUtils import benchmark_model, benchmark_table, cross_table

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


X, y, wl, ref = importLuzCol(specs, lab, 4)

# split dataset in train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)



data = {"X": X_train, "y": y_train, "X_test": X_test, "y_test": y_test}

data["y"]

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

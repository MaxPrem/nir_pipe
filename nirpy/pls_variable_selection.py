# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from ImportModule import importLuzCol, plot_spec

# %%
# impor data

# specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")  # full spectra
lab = pd.read_excel("/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx")
# input wavenumber to cut spectra
#specs = cut_specs(specs, 4100, 6500)
# specs = cut_specs(specs, 4100, 5500)

# %%
# assing spectra and refernce method


X, y, wave_number, feat_ref = importLuzCol(specs, lab, 4)

# split dataset in train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# %%

plot_spec(wave_number, X)

# %%


dev_0 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=0))
    ]
)

X_0 = dev_0.fit_transform(X)

plot_spec(wave_number, X_0)
# %%


dev_1 = Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=1))
    ]
)

X_1 = dev_1.fit_transform(X)

plot_spec(wave_number, X_1)
# %%


dev_2= Pipeline(
    [
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2, deriv=2))
    ]
)

X_2 = dev_2.fit_transform(X)

plot_spec(wave_number, X_2)


-wave_number.max()

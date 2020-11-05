# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter, Dataaugument
from CrossValPLS import optimal_n_comp, pls_regression, pls_scores, variance_explained
from ElasticNetVariableSelection import EnetSelect
from ImportModule import cut_specs, importLuzCol, plot_spec
from ValidationUtils import print_regression_benchmark, print_cv_table, cv_benchmark_model, print_nir_metrics

# %%
# impor data

# specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")  # full spectra
lab = pd.read_excel("/Users/maxprem/nirGit/nirpy/luzrawSpectra/labData.xlsx")
# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)


X, y, wave_number, ref = importLuzCol(specs, lab, 4)
# %%

x1 = X[0:9,:]

plot_spec(wave_number, x1)

# %%

aug_pipline = Pipeline([
	("scaleing_X", GlobalStandardScaler()),
	("dataaugmentation", Dataaugument()),
	("scatter_correction", EmscScaler())
	])



# to perform variable selection y values are corrleated in the fit met
x_aug = aug_pipline.fit_transform(x1)


plot_spec(wave_number, x_aug)

emsc = EmscScaler()
x1_emsc = emsc.fit_transform(x_aug)
x10_emsc = emsc.fit_transform(x1)

plot_spec(wave_number, x1_emsc)
plot_spec(wave_number, x10_emsc)

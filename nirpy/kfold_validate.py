import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#import

specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv') # full spectra
lab = pd.read_excel('./luzrawSpectra/labdata.xlsx')




from import_Module import importLuzCol, cut_specs

# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)
#specs = cut_specs(specs, 4100, 5500)



X, y, wl, ref = importLuzCol(specs, lab, 4)


from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from pls_utils import pls_opt_cv, pls_cv

# splitting dataset
"""to be continued with test set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#########################
# scaling and transformingfrom ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from sklearn.cross_decomposition import  PLSRegression
from pls_utils import PLSOptimizer, Outlier
from enet_var import Enet_Select

# scale y
y_scaler = GlobalStandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)


# pipeline = Pipeline([
#     ("scaleing_X", GlobalStandardScaler()),
#     ("scatter_correction", EmscScaler()),
#     ("smmothing", SavgolFilter(polyorder=2,deriv=0)),
#     ("variable_selection", Enet_Select())
# ])


pipeline = Pipeline([
	("scaleing_X", GlobalStandardScaler()),
	("scatter_correction", EmscScaler()),
	("smmothing", SavgolFilter(polyorder=2,deriv=0))
])




# transforming only the spectra

''''''
#X_train = pipeline.fit_transform(X_train)

X_train = pipeline.fit_transform(X_train)

X_test = pipeline.transform(X_test)


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate


from sklearn.metrics import mean_squared_error, r2_score,  make_scorer
from sklearn import metrics

from ChemUtils import huber

model = pls_opt_cv(X_train, y_train,29)


from validation_utils import create_score_list, cv_scores, cross_table, da_func_ncv

cross_table(X_train, y_train, X_test, y_test, model)

da_func_ncv(X_train, y_train, X_test, y_test, y, ref, model)
############################################
############################################
# from sklearn.cross_decomposition import PLSRegression
# scores = []
# residues = []
# pls = PLSRegression()
#
#
# cv = KFold(n_splits=5)
# for train_index, test_index in cv.split(X_train):
# 	print("Train Index: ", train_index, "\n")
# 	print("Test Index: ", test_index)
#
# 	X_train_kfold, X_test_kfold, y_train_kfold, y_test_kfold = X_train[train_index], X_train[test_index], y_train[train_index], y_train[test_index]
# 	pls.fit(X_train_kfold, y_train_kfold)
#
# 	# STECV
# 	p_pred_cv = pls.predict(X_train_kfold)
# 	residues.append(np.sum((p_pred_cv-y_train_kfold)**2))
#
# 	scores.append(pls.score(X_test_kfold, y_test_kfold))
# residues
# print(scores)
#
# SECV = []
# for residue in residues:
# 	SECV.append(np.sqrt(residue/(X_test_kfold.shape[0]-1)))
#
# SECV
# np.mean(SECV)
# np.std(SECV)
y
############################################
############################################
############################################



def calc_secv(X, y, model, n_splits=10):
	residues = []

	cv = KFold(n_splits=n_splits)
	for train_index, test_index in cv.split(X):
		#print("Train Index: ", train_index, "\n")
		#print("Test Index: ", test_index)

		X_train_kfold, X_test_kfold, y_train_kfold, y_test_kfold = X[train_index], X[test_index], y[train_index], y[test_index]
		# train model on splits
		model.fit(X_train_kfold, y_train_kfold)

		# SECV
		y_pred_cv = model.predict(X_train_kfold)
		residues.append(np.sum((y_pred_cv - y_train_kfold)**2))

		#y_pred_cv = model.predict(X_train_kfold)
		#residues.append(y_pred_cv - y_train_kfold)



	SECV = []
	for residue in residues:
		#print(residue)
		SECV.append(np.sqrt(residue/(y_pred_cv.shape[0]-1)))
		#print('resiude', residue)
		##print('xtestshape', y_pred_cv.shape)
		#print('SECV', SECV)
		#print(np.mean(SECV).round(2),'±',np.std(SECV).round(2))
	# print('X_train_kfold',X_train_kfold ,X_train_kfold.shape)
	# # print('SECV',SECV)
	print(np.mean(SECV).round(2),'±',np.std(SECV).round(2))



def cal_rpd(X, y, model):
	std = np.std(X)
	rpd = std/calc_sec(X, y, model)
	return rpd





############################
############################################
############################################
############################################
######      MultiModleScorer  ##############

#num_instances = 10

splits = 5
huber_score = make_scorer(huber, greater_is_better=True)
scoring ='r2'

models = []
models.append(('PLS', PLSRegression()))
models.append(('PLS3', PLSRegression(n_components=3)))
models.append(('PLS4', PLSRegression(n_components=4)))
# models.append(('RandomForestRegressor', RandomForestRegressor()))
# models.append(('LassoLars', LassoLars()))
#models.append(('PLS', ElasticNetCV(max_iter=10000)))
results = []
names = []

for name, model in models:
	#kfold = KFold(n_splits=splits)
	cv_results = cross_val_score(model, X_train, y_train, cv=5)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print('cv_results', cv_results)
	print(msg)


scores = cross_val_score(model, X_train, y_train, cv=5)
scores.mean()
# warum • 2
print("Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

############################################
############################################
######       Multiscorere   ################

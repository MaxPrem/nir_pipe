from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, cross_validate
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import  PLSRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f
import matplotlib.collections as collections



from sklearn.metrics import mean_squared_error, r2_score,  make_scorer

from sklearn.model_selection import  train_test_split

def pls_opt_cv(X, y, n_comp = 10, plot_components=True):
	"""Run PLS regression up to the number of components given, and finde the minimum mean squared error."""

	mse = []

	component = np.arange(1,n_comp)

	for i in component:
		pls = PLSRegression(n_components=i)

		# Cross-validation
		y_cv = cross_val_predict(pls, X, y, cv = 10)

		mse.append(cross_validate(pls, X, y, scoring=make_scorer(mean_squared_error), cv=10)['test_score'].mean())

		comp = 100*(i+1)/40
		# Trick to updata status on the same line
		stdout.write("\r%d%% completed" % comp)
		stdout.flush()
	stdout.write("\n")

	# Calculate and print the postion of of minimum in mse 'mean squared error'
	msemin = np.argmin(mse)
	print("Suggested number of components: ", msemin+1)
	stdout.write("\n")

	if plot_components is True:
		with plt.style.context(('ggplot')):
			plt.plot(component, np.array(mse), '-v', color = 'blue',
					  mfc='blue')
			plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10,
					  mfc='red')
			plt.xlabel('Number of PLS components')
			plt.ylabel('MSE CV')
			plt.title('PLS')
			plt.xlim(left=-1)
		plt.show()

	# Define PLS object with optimal number of components
	pls_opt = PLSRegression(n_components=msemin+1)

	# Fit to the entire dataset
	model = pls_opt.fit(X,y)
	y_c = pls_opt.predict(X)

	# Cross-VALIDATION
	y_cv = cross_val_predict(pls_opt, X, y, cv=10)

	# Calculate scores for calibration and cross-validation
	score_c = r2_score(y,y_c)
	score_cv = cross_val_score(model, X, y, cv=10)


	# Calculate mean squared error for calibration and cross validation
	mse_c = mean_squared_error(y,y_c)
	mse_cv = cross_val_score(model, X, y, cv=10, scoring= make_scorer(mean_squared_error))

	print('R2 calib: %5.3f' % score_c)
	print('R2 CV2: %5.3f' % score_cv.mean())
	print('MSE calib: %5.3f' % mse_c)
	print('MSE CV: %5.3f' % mse_cv.mean())


	## Plot regression and figures of merit
	# rangey = max(y) - min(y)
	# rangex = max(y_c) - min(y_c)
	#
	# # Fit a line to the CV vs Response
	#
	# z = np.polyfit(y, y_c, 1)
	# with plt.style.context(('ggplot')):
	# 	 fig, ax = plt.subplots(figsize=(9, 5))
	# 	 ax.scatter(y_c, y, c='red', edgecolors='k')
	# 	 #ax.scatter(y_cv, y, c='blue', edgecolors='k')
	# 	 #Plot the best fit line
	# 	 ax.plot(np.polyval(z,y), y, c='blue', linewidth=1)
	# 	 #Plot the ideal 1:1 linear
	# 	 ax.plot(y,y, color='green', linewidth=1)
	#
	# 	 plt.title('$R^{2}$ (CV): '+str(score_cv.mean().round(3)))
	# 	 plt.xlabel('Predicted')
	# 	 plt.ylabel('Measured')
	#
	# 	 plt.show()

	return model

def pls_min(X,y, n_comp):

	cv_optimised_comp(X,y,n_comp)

	plot_regression(y, y_c, scorecv)


def cv_optimised_comp(X, y, n_comp = 20):
	'''Finds minimal mean squared error by 10-fold crossvalidation
	up to a give number of components'''

	mse = []
	component = np.arange(1,n_comp)

	for i in component:
		pls = PLSRegression(n_components=i)

		# Cross-validation

		mse.append(cross_validate(pls, X, y, scoring=make_scorer(mean_squared_error), cv=10)['test_score'].mean())

		comp = 100*(i+1)/40
		# Trick to updata status on the same line
		stdout.write("\r%d%% completed" % comp)
		stdout.flush()
	stdout.write("\n")

	mse_min = np.argmin(mse)
	print("Suggested number of components: ", mse_min)
	stdout.write("\n")
	# plot mse for each component
	plot_mse(mse,component)

	return mse_min+1

def plot_mse(mse, component):
	mse_min = np.argmin(mse)
	print("Suggested number of components: ", mse_min)
	stdout.write("\n")
	# plot mse for each component
	with plt.style.context(('ggplot')):
		plt.plot(component, np.array(mse), '-v', color = 'blue',
				  mfc='blue')
		plt.plot(component[mse_min], np.array(mse)[mse_min], 'P', ms=10,
				  mfc='red')
		plt.xlabel('Number of PLS components')
		plt.ylabel('MSE CV')
		plt.title('PLS')
		plt.xlim(left=-1)
	plt.show()

cv_optimised_comp(X,y,10)

def plot_regression(y, y_c, score_cv):
	'''plots predicted vs meassured reference method'''

	rangey = max(y) - min(y)
	rangex = max(y_c) - min(y_c)

	z = np.polyfit(y, y_c, 1)
	with plt.style.context(('ggplot')):
		 fig, ax = plt.subplots(figsize=(9, 5))
		 ax.scatter(y_c, y, c='red', edgecolors='k')
		 #ax.scatter(y_cv, y, c='blue', edgecolors='k')
		 #Plot the best fit line
		 ax.plot(np.polyval(z,y), y, c='blue', linewidth=1)
		 #Plot the ideal 1:1 linear
		 ax.plot(y,y, color='green', linewidth=1)

		 plt.title('$R^{2}$ (CV): '+str(score_cv.mean().round(3)))
		 plt.xlabel('Predicted')
		 plt.ylabel('Measured')

		 plt.show()


specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv') # full spectra
lab = pd.read_excel('./luzrawSpectra/labdata.xlsx')




from ImportModule import importLuzCol, cut_specs

# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)
#specs = cut_specs(specs, 4100, 5500)



X, y, wl, ref = importLuzCol(specs, lab, 2)



from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from pls_utils import pls_cv

# splitting dataset
"""to be continued with test set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#########################
# scaling and transformingfrom ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from sklearn.cross_decomposition import  PLSRegression
from pls_utils import PLSOptimizer, Outlier
from enet_var import Enet_Select
from sklearn.pipeline import Pipeline
# scale y
y_scaler = GlobalStandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# needing unscaled


pipeline = Pipeline([
	("scaleing_X", GlobalStandardScaler()),
	("scatter_correction", EmscScaler()),
	("smmothing", SavgolFilter(polyorder=2,deriv=0)),
	#("variable_selection", Enet_Select())
])



pip = Pipeline([
	("scaleing_X", GlobalStandardScaler()),
	("scatter_correction", EmscScaler()),
	("smmothing", SavgolFilter(polyorder=2,deriv=0)),
])

X_train_pip = pip.fit_transform(X_train)
X_test_pip = pip.transform(X_test)



# transforming only the spectra

''''''
#X_train = pipeline.fit_transform(X_train)
X_train_sel = pipeline.fit_transform(X_train, y_train)
X_test_sel = pipeline.transform(X_test)


#######




sel_model = pls_opt_cv(X_train_sel, y_train, 4)
full_model = pls_opt_cv(X_train_pip, y_train, 4)





##############################################
##############################################
##############################################
##############################################
##############################################
# full_model = pls_opt_cv(X_train, y_train, 3)
y_train.shape

# full_model.get_params()['n_components']

from validation_utils import cross_table, da_func_ncv, calc_secv

calc_secv(X_train_pip, y_train, full_model)
# conf intervall
#MSECV
da_func_ncv(X_train_sel, y_train, X_test_sel, y_test, y, ref, sel_model)
# da_func_ncv(X_train_pls, y_train, X_test_pls, y_test, pls_model)

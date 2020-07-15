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
	"""Run PLS including a variable number of components, up to n_comp"""

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


	#Plot regression and figures of merit
	rangey = max(y) - min(y)
	rangex = max(y_c) - min(y_c)

	# Fit a line to the CV vs Response

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

	return model




specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv') # full spectra
lab = pd.read_excel('./luzrawSpectra/labdata.xlsx')




from import_Module import importLuzCol, cut_specs

# input wavenumber to cut spectra
specs = cut_specs(specs, 4100, 5500)
#specs = cut_specs(specs, 4100, 5500)



X, y, wl = importLuzCol(specs, lab, 2)



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
da_func_ncv(X_train_sel, y_train, X_test_sel, y_test, y, sel_model)
# da_func_ncv(X_train_pls, y_train, X_test_pls, y_test, pls_model)


da_func_ncv(X_train_pip, y_train, X_test_pip, y_test, full_model)
da_func_ncv(X_train_pip, y_train, X_test_pip, y_test, full_model3)

%load_ext autoreload
%autoreload 2

np.std(y)
y_train
da_func_ncv(X_train_pip, y_train, X_test_pip, y_test, full_model9)

# %% markdown
When calculating the SEP, it is critical that the constituent distribution be uniform and the wet chemistry be very accurate for the validation sample set. If these criteria are not met for validation sample sets, the calculated SEP may not have validity as an accurate indicator of overall calibration performance. To summarize, the SEP is the square root of the mean square for residuals for N − 1 degrees of freedom, where the residual equals actual minus predicted for samples outside the calibration set.

# %% codecell

def SEP(X, y, model):

	n_samples = X.shape[0]


	y_c = model.predict(X)

	res = np.sum((y_c - y_train)**2)
	sep= np.sqrt(res/(n_samples))
	print('Standard Error of Prediction:', sep)



def SEC(X, y, model):
	''' SEC will decrease with the number of wavelengths (independent variable
	NIRS Calibration Basics 145
	terms) used within an equation, indicating that increasing the number of terms will allow more variation within the data to be explained, or “fitted.”
	The SEC statistic is a useful estimate of the theoretical “best” accuracy obtainable for a specified set of wavelengths used to develop a calibration equation. T
	'''
	n_samples = X.shape[0]
	n_coef = model.get_params()['n_components']

	y_c = model.predict(X)

	res = np.sum((y_c - y_train)**2)
	sec = np.sqrt(res/abs(n_samples-n_coef-1))
	print('Standard Error Calibration:', sec)

from sklearn.model_selection import cross_validate, KFold

def SECV(X, y, model, cv=5):

   n_samples = X.shape[0]
   n_wave=X.shape[1]

   y_cv = cross_val_predict(model, X, y, cv=cv)
   # y_cv = cross_validate(model, X_test, y_test, cv=kfold)
   # return(y_cv)
   #
   # y_cv = KFold(n_splits=5)
   # prediction_cv = cross_val_predict(model, X, y, cv=cv)
   # return(y_cv)
   res = np.sum((y_cv - y)**2)

   res2 = (cross_validate(model, X, y, scoring=make_scorer(mean_squared_error), cv=cv)['test_score'])

   sec = np.sqrt(res/abs(n_samples-1))


   print('Standard Error Crossvalidation:', sec)
   print('res',res)

   print('msecv', np.sqrt(np.sum(res2*(n_samples-1))))
   print('rec2', res2)
   print('rec**2', res2**2)
   print('rec2sum', np.sum(res2**2))


SECV(X_train_pls, y_train, pls_model)


def SECV1(X, y, model, cv=None):

   if cv == None:
       cv = 5

   n_samples = X.shape[0]
   n_wave=X.shape[1]

   y_cv =model.predict(X, y)
   # y_cv = cross_validate(model, X_test, y_test, cv=kfold)
   # return(y_cv)
   #
   # y_cv = KFold(n_splits=5)
   # prediction_cv = cross_val_predict(model, X, y, cv=cv)
   # return(y_cv)
   res = np.sum((y_cv - y)**2)
   sec = np.sqrt(res/abs(n_samples-1))
   print('Standard Error Crossvalidation:', sec.mean())


def SECV2(X, y, model, cv=None):
	#### trying to use CV predichted

	if cv == None:
		cv = 5


	n_samples = X.shape[0]
	n_wave=X.shape[1]

	kfold = KFold(n_splits=cv)
	cv_results = cross_validate(model, X_train, y_train, cv=kfold,
	scoring=make_scorer(mean_squared_error))
	return cv_results
	#return cv_results
	# return(y_cv)
	res = np.sum(cv_results['test_score']*n_samples)
	sec = np.sqrt(res/abs(n_samples-1))
	print('Standard Error Crossvalidation:', sec.mean())


import numpy as np
from sklearn.model_selection import KFold


from sklearn.cross_decomposition import PLSRegression

pls= PLSRegression()
kf = KFold(n_splits=5)
kf.split(X, y)

X.shape
X[3,:].shape
y_c_list = []
def kfold(X, y, splits = 5):
	for train, test in kf.split(X):
		pls.fit(X[train], y[train])
		y_c = pls.predict(X[train])
		return y_c

		# print(X[train])
		# print(y[train])

	# model = pls.fit(X[train,:], y[train])
	# y_c = model.predict(X_test)

nl=kfold(X, y)
nl.shape

y-nl
train
X[train].shape
X.shape


hasattr(model, 'get_params')

x_cv = SECV(X_train_sel, y_train, model)
x_cv = SECV2(X_train_sel, y_train, model)

x_cv['test_score'].mean()*60

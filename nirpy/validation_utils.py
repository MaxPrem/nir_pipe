from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, cross_validate
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from scipy.stats import f
import matplotlib.collections as collections


#############################
######## NIR Metrics ########
#############################

def calc_secv(X, y, model, n_splits=10):
	'standard error cross validation'
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
	# print(np.mean(SECV).round(2),'±',np.std(SECV).round(2), 'cv:', n_splits)
	secv = np.mean(SECV).round(3)
	return secv


# classical NIR PARA
def calc_sep(X, y, model):
	'standard error prediction'

	n_samples = X.shape[0]


	y_c = model.predict(X)

	res = np.sum((y_c - y)**2)
	sep= np.sqrt(res/(n_samples-1))
	return sep

def calc_rpd(X_train, y_train, y, model):
	'relative prediction deviation - using unscaled y'
	std = np.std(y)
	rpd = std/calc_sec(X_train, y_train, model)
	return rpd


def calc_sec(X, y, model):
	''' SEC will decrease with the number of wavelengths (independent variable
	NIRS Calibration Basics 145
	terms) used within an equation, indicating that increasing the number of terms will allow more variation within the data to be explained, or “fitted.”
	The SEC statistic is a useful estimate of the theoretical “best” accuracy obtainable for a specified set of wavelengths used to develop a calibration equation. T
	'''
	n_samples = X.shape[0]
	n_coef = model.get_params()['n_components']

	y_c = model.predict(X)

	res = np.sum((y_c - y)**2)
	sec = np.sqrt(res/abs(n_samples-n_coef-1))
	return sec



##############################
#### CV Metric Functions #####
##############################

# Functions used to performe crossfalidation of metrics and scores
# list of crossvalidated scores
def create_score_list():
	score_list = []
	# add metrics to list
	score_list.append(('Variance expl.', make_scorer(metrics.explained_variance_score)))
	score_list.append(('R2', 'r2'))
	score_list.append(('Huber Loss', make_scorer(huber)))
#	score_list.append(('mse_abs' , make_scorer(metrics.mean_absolute_error)))
	score_list.append(('MSE',make_scorer(mean_squared_error)))
	return score_list
def create_rmse_list():
	score_list = []
	# add metrics to list

	score_list.append(('RMSE',make_scorer(mean_squared_error)))
	return score_list

# cross validation on trained model
def cv_scores(X, y, model, score_list, cv=5):
	results = []
	#results_model = []
	names = []
	results_mean = []
	results_std = []
	for score_name, score in score_list:

		cv_results = cross_validate(model, X, y, cv=cv, scoring=score)
		#results_model.append(cv_results)
		names.append(score_name)

		#msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		#print(score_name, cv_results['test_score'].mean().round(3),'±',cv_results['test_score'].std().round(3))

		results.append((cv_results['test_score'].mean().round(3),'±',cv_results['test_score'].std().round(3)))

		results_mean.append((cv_results['test_score'].mean().round(3)))
		results_std.append((cv_results['test_score'].std().round(3)))

	return results , names, results_mean, results_std

# output function
def cross_table(X_train, y_train, X_test, y_test, model):
	# unpacks dict from crossvalpredict with metrics and prits output table
	score_list = create_score_list()
	train_results, score_name, train_mean, train_std = cv_scores(X_train, y_train, model, score_list)

	test_results, score_name, test_mean, test_std = cv_scores(X_test, y_test, model, score_list)

	for name, item_a_m, item_a_s, item_b_m, item_b_s in zip(score_name, train_mean, train_std, test_mean, test_std):
		print('{: <5}\t\t\t{}{}{}\t{}{}{}'.format(name, item_a_m,' ± ', item_a_s,
		item_b_m,' ± ', item_b_s))

def rmse_cross_table(X_train, y_train, X_test, y_test, model):
	# unpacks dict from crossvalpredict with metrics and prits output table
	rmse_list = create_rmse_list()
	train_results, score_name, train_mean, train_std = cv_scores(X_train, y_train, model, rmse_list)

	test_results, score_name, test_mean, test_std = cv_scores(X_test, y_test, model, rmse_list)

	for name, train_mean, train_sd, test_mean, test_sd in zip(score_name, train_mean, train_std, test_mean, test_std):

		train_mean, train_sd, test_mean, test_sd = np.sqrt(np.square(train_mean)),np.sqrt(np.square(train_sd)), np.sqrt(np.square(test_mean)), np.sqrt(np.square(test_sd))

		print('{: <5}\t\t\t{}{}{}\t{}{}{}'.format(name, np.sqrt(train_mean).round(3),' ± ', np.sqrt(train_sd).round(3), np.sqrt(test_mean).round(3),' ± ', np.sqrt(test_sd).round(3)))



# score function (not cross validated)

def score_table(X, y, model):

	# y_calibration estimate
	y_c = model.predict(X)
	# cv estimate for plotting
	y_cv = cross_val_predict(model, X, y, cv=10)

	# Calculate scores for calibration and cross-validation
	hub = huber(y, y_c)
	mse_c = mean_squared_error(y, y_c)
	score_c = r2_score(y, y_c)


	return y_c, y_cv, hub, mse_c, score_c



def da_func_ncv(X_train, y_train, X_test, y_test, y, ref, model):
	'''Final Output Function for pls regression'''
	# get name of reference method for outputtable




	rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
	rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
	# hub = huber(y_train, model.predict(X_train))
	# hub_test = huber(y_test, model.predict(X_test))
	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
	# print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))
	print("\n")

	y_c, y_cv, hub, mse_c, score_c = score_table(X_train, y_train, model)


	y_c_test, y_cv_test, hub_test, mse_c_test, score_test = score_table(X_test, y_test, model)



	sec_train = calc_sec(X_train, y_train, model)
	#sep_train = calc_sep(X_train, y_train, model)
	#sec_test= SEC(X_test, y_test, model)
	sep_test= calc_sep(X_test, y_test, model)

	secv_train = calc_secv(X_test, y_test, model)

	rpd = calc_rpd(X_train, y_train,y, model)
	# pls model components
	n_comp = model.get_params()['n_components']
	print('Number of Principal Components:', n_comp)

	print('Number of training sampels:', y_train.shape[0])
	print('Number of test sampels:', y_test.shape[0])
	print('SD reference method ', ref,':', np.std(y))

	# nir metrics
	print('SEC \t%0.4F'%(sec_train.round(3)))
	print('SEP \t%0.4F'%(sep_test.round(3)))
	print('SECV \t%0.4F'%(secv_train.round(3)))
	print('RPD \t%0.4F'%(rpd.round(3)))

	# printing
	print('#### Valdation #### \t{}\t\t{}'.format('train', 'test'))

	print('MSE calib\t\t%0.4F\t\t%0.4F'%(mse_c , mse_c_test))
	print('RMSE calib\t\t%0.4F\t\t%0.4F'%(np.sqrt(mse_c) , np.sqrt(mse_c_test)))
	print("Huber\t\t\t%0.4F\t\t%0.4F"%(hub, hub_test))
	print('R2 calib\t\t%0.4F\t\t%0.4F'%(score_c , score_test))
	print('MSE calib\t\t%0.4F\t\t%0.4F'%(mse_c , mse_c_test))
	print('RMSE calib\t\t%0.4F\t\t%0.4F'%(np.sqrt(mse_c) , np.sqrt(mse_c_test)))
	print('conf 95%')
	print('kfold')
	#cross validation loop + unpack function
	print('#### CrossVal. #### \t{}\t\t{}'.format('train', 'test'))
	cross_table(X_train, y_train, X_test, y_test, model)
	rmse_cross_table(X_train, y_train, X_test, y_test, model)



	z = np.polyfit(y_train, y_cv, 1)
	with plt.style.context(('ggplot')):
		fig, ax = plt.subplots(figsize=(9,5))
		ax.scatter(y_cv, y_train, edgecolors='k')
		ax.scatter(y_cv_test, y_test, edgecolors='k')
		ax.plot(z[1]+z[0]*y_train, y_train)
		ax.plot(y_train,y_train)
		plt.title('$R^{2}$ (CV):' + str(cross_val_score(model, X_train, y_train, cv=10).mean().round(2)))
		plt.xlabel('Predicted')
		plt.ylabel('Measured')

		plt.show()




### if no pls components can be extracted....
def da_func2(X_train, y_train, X_test, y_test, model):
	rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
	rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
	# hub = huber(y_train, model.predict(X_train))
	# hub_test = huber(y_test, model.predict(X_test))
	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
	# print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))
	print("\n")

	#####################
	#  pls cv here#
	#####################
	# get n_comps from fitted model
	if hasattr(model, 'get_params')  == True:
		params = model.get_params()
		print('PLS model with:', params, 'components.')

	#n_comp = params["n_components"]
	# y_calibration estimate
	y_c = model.predict(X_train)
	# Cross-vali-predicted
	y_cv = cross_val_predict(model, X_train, y_train, cv=10)
	# Calculate scores for calibration and cross-validation
	scores_c = r2_score(y_train, y_c)
	scores_cv = r2_score(y_train, y_cv)

	scores_cv2 = cross_val_score(model, X_train, y_train)
	#sklearn cross_val_score
	# Calculate mean square error for calibration and cross VALIDATION
	mse_c = mean_squared_error(y_train, y_c)
	m_cv = mean_squared_error(y_train, y_cv)

	#####################
	#  pls Test cv here#
	#####################
	# get n_comps from fitted model

	#n_comp = params["n_components"]
	# y_calibration estimate
	y_c_test = model.predict(X_test)
	# Cross-validation
	y_cv_test = cross_val_predict(model, X_test, y_test, cv=10)
	# Calculate scores for calibration and cross-validation
	scores_c_test = r2_score(y_test, y_c_test)
	scores_cv_test = r2_score(y_test, y_cv_test)
	scores_cv_test2 = cross_val_score(model, X_test, y_test)
	# Calculate mean square error for calibration and cross VALIDATION
	mse_c_test = mean_squared_error(y_test, y_c_test)
	mse_cv_test = mean_squared_error(y_test, y_cv_test)
	#
	# rmse_c_test = np.sqrt((mean_squared_error(y_test, y_c_test)))
	# rmse_cv_test =np.sqrt((mean_squared_error(y_test, y_cv_test)))
	hub_2 = huber(y_train, y_c)
	hub_test_2 = huber(y_test, y_c_test)
	hub_train_cv = huber(y_train, y_cv)
	hub_test_cv = huber(y_test, y_cv_test)
	print ("Huber2 Train/Test\t%0.4F\t%0.4F"%(hub_2, hub_test_2))
	print ("HuberCV Train/Test\t%0.4F\t%0.4F"%(hub_train_cv, hub_test_cv))
	# calc_metrics(X_train, y_train, X_test, y_test, model)

	##################################
	print('##################################')
	##################################
	print('R2 calib Train/Test\t%0.4F\t%0.4F'%(scores_c , scores_c_test))
	print('R2 CV Train/Test\t%0.4F\t%0.4F'%(scores_cv , scores_cv_test))

	print('R2 CV2 Train',scores_cv2)
	print('R2 CV2mean Train',scores_cv2.mean())
	print('R2 CV2sd Train',scores_cv2.std())
	print('R2 CV2Test',scores_cv_test2)
	print('MSE calib Train/Test\t%0.4F\t%0.4F'%(mse_c , mse_c_test))
	print('MSE CV Train/Test\t%0.4F\t%0.4F'%(mse_cv , mse_cv_test))
	#´print('RMSE calib Train/Test\t%0.4F\t%0.4F'%(rmse_c , reprmse_c_test))
	#´print('RMSE CV Train/Test\t%0.4F\t%0.4F'%(rmse_cv , rmse_cv_test))


	z = np.polyfit(y_train, y_cv, 1)
	with plt.style.context(('ggplot')):
		fig, ax = plt.subplots(figsize=(9,5))
		ax.scatter(y_cv, y_train, edgecolors='k')
		ax.scatter(y_cv_test, y_test, edgecolors='k')
		ax.plot(z[1]+z[0]*y_train, y_train)
		ax.plot(y_train,y_train)
		plt.title('$R^{2}$ (CV):' + str(scores_cv))
		plt.xlabel('Predicted $^{\circ}$Brix')
		plt.ylabel('Measured $^{\circ}$Brix')

		plt.show()



def huber(y_true, y_pred, delta=1.0):
	y_true = y_true.reshape(-1,1)
	y_pred = y_pred.reshape(-1,1)
	return np.mean(delta**2*( (1+((y_true-y_pred)/delta)**2)**0.5 -1))


def huber_np(y_true, y_pred, delta=1.0):
	y_true = y_true.reshape(-1,1)
	y_pred = y_pred.reshape(-1,1)
	return np.mean(delta**2*( (1+(np.substract(y_true,y_pred)/delta)**2)**0.5 -1))

#select same varaiables on X_test
def benchmark(X_train,y_train,X_test, y_test, model):
	rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
	rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
	hub = huber(y_train, model.predict(X_train))
	hub_test = huber(y_test, model.predict(X_test))
	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
	print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))


def cv_benchmark(X_train,y_train,X_test, y_test, model):
	rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
	rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
	hub = huber(y_train, model.predict(X_train))
	hub_test = huber(y_test, model.predict(X_test))
	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
	print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))
	print("\n")

	#####################
	#  pls cv here#
	#####################
	# get n_comps from fitted model

	params = model.get_params()
	n_comp = params["n_components"]
	# y_calibration estimate
	y_c = model.predict(X_train)
	# Cross-vali-predicted
	y_cv = cross_val_predict(model, X_train, y_train, cv=10)
	# Calculate scores for calibration and cross-validation
	scores_c = r2_score(y_train, y_c)
	scores_cv = r2_score(y_train, y_cv)
	#sklearn cross_val_score
	# Calculate mean square error for calibration and cross VALIDATION
	mse_c = mean_squared_error(y_train, y_c)
	mse_cv = mean_squared_error(y_train, y_cv)

	#####################
	#  pls Test cv here#
	#####################
	# get n_comps from fitted model
	params = model.get_params()
	n_comp = params["n_components"]
	# y_calibration estimate
	y_c_test = model.predict(X_test)
	# Cross-validation
	y_cv_test = cross_val_predict(model, X_test, y_test, cv=10)
	# Calculate scores for calibration and cross-validation
	scores_c_test = r2_score(y_test, y_c_test)
	scores_cv_test = r2_score(y_test, y_cv_test)
	# Calculate mean square error for calibration and cross VALIDATION
	mse_c_test = mean_squared_error(y_test, y_c_test)
	mse_cv_test = mean_squared_error(y_test, y_cv_test)
	#
	# rmse_c_test = np.sqrt((mean_squared_error(y_test, y_c_test)))
	# rmse_cv_test =np.sqrt((mean_squared_error(y_test, y_cv_test)))


	# calc_metrics(X_train, y_train, X_test, y_test, model)

	##################################
	print('##################################')
	##################################
	print('R2 calib Train/Test\t%0.4F\t%0.4F'%(scores_c , scores_c_test))
	print('R2 CV Train/Test\t%0.4F\t%0.4F'%(scores_cv , scores_cv_test))
	print('MSE calib Train/Test\t%0.4F\t%0.4F'%(mse_c , mse_c_test))
	print('MSE CV Train/Test\t%0.4F\t%0.4F'%(mse_cv , mse_cv_test))
	#´print('RMSE calib Train/Test\t%0.4F\t%0.4F'%(rmse_c , reprmse_c_test))
	#´print('RMSE CV Train/Test\t%0.4F\t%0.4F'%(rmse_cv , rmse_cv_test))


	z = np.polyfit(y_train, y_cv, 1)
	with plt.style.context(('ggplot')):
		fig, ax = plt.subplots(figsize=(9,5))
		ax.scatter(y_cv, y_train, edgecolors='k')
		ax.scatter(y_cv_test, y_test, edgecolors='k')
		ax.plot(z[1]+z[0]*y_train, y_train)
		ax.plot(y_train,y_train)
		plt.title('$R^{2}$ (CV):' + str(scores_cv))
		plt.xlabel('Predicted $^{\circ}$Brix')
		plt.ylabel('Measured $^{\circ}$Brix')

		plt.show()


# validation
def tensor_benchmark(X_train,y_train,X_test, y_test, model):
	rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
	rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
	hub = huber(y_train, model.predict(X_train))
	hub_test = huber(y_test, model.predict(X_test))
	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
	print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))
	print("\n")

	#####################
	#  pls cv here#
	#####################
	# get n_comps from fitted model

	####€params = model.get_params()
	#####n_comp = params["n_components"]

	# y_calibration estimate
	y_c = model.predict(X_train)
	# Cross-validation
	y_cv = cross_val_predict(model, X_train, y_train, cv=10)
	# Calculate scores for calibration and cross-validation
	scores_c = r2_score(y_train, y_c)
	scores_cv = r2_score(y_train, y_cv)
	# Calculate mean square error for calibration and cross VALIDATION
	mse_c = mean_squared_error(y_train, y_c)
	mse_cv = mean_squared_error(y_train, y_cv)

	#####################
	#  pls Test cv here#
	#####################
	# get n_comps from fitted model

	###params = model.get_params()
	#####n_comp = params["n_components"]

	# y_calibration estimate
	y_c_test = model.predict(X_test)
	# Cross-validation
	y_cv_test = cross_val_predict(model, X_test, y_test, cv=10)
	# Calculate scores for calibration and cross-validation
	scores_c_test = r2_score(y_test, y_c_test)
	scores_cv_test = r2_score(y_test, y_cv_test)
	# Calculate mean square error for calibration and cross VALIDATION
	mse_c_test = mean_squared_error(y_test, y_c_test)
	mse_cv_test = mean_squared_error(y_test, y_cv_test)

	rmse_c_test = sqrt((mean_squared_error(y_test, y_c_test)))
	rmse_cv_test =sqrt((mean_squared_error(y_test, y_cv_test)))


	# calc_metrics(X_train, y_train, X_test, y_test, model)

	##################################
	print('##################################')
	print ("RMSE  Train/Test\t%0.4F\t%0.4F"%(rmse, rmse_test))
	print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))
	##################################
	print('R2 calib Train/Test\t%0.4F\t%0.4F'%(scores_c , scores_c_test))
	print('R2 CV Train/Test\t%0.4F\t%0.4F'%(scores_cv , scores_cv_test))
	print('MSE calib Train/Test\t%0.4F\t%0.4F'%(mse_c , mse_c_test))
	print('MSE CV Train/Test\t%0.4F\t%0.4F'%(mse_cv , mse_cv_test))
	#´print('RMSE calib Train/Test\t%0.4F\t%0.4F'%(rmse_c , reprmse_c_test))
	#´print('RMSE CV Train/Test\t%0.4F\t%0.4F'%(rmse_cv , rmse_cv_test))


	z = np.polyfit(y_train, y_cv, 1)
	with plt.style.context(('ggplot')):
		fig, ax = plt.subplots(figsize=(9,5))
		ax.scatter(y_cv, y_train, edgecolors='k')
		ax.scatter(y_cv_test, y_test, edgecolors='k')
		ax.plot(z[1]+z[0]*y_train, y_train)
		ax.plot(y_train,y_train)
		plt.title('$R^{2}$ (CV):' + str(scores_cv))
		plt.xlabel('Predicted $^{\circ}$Brix')
		plt.ylabel('Measured $^{\circ}$Brix')

		plt.show()


# validation


def scaled_benchmark(X_train,y_train,X_test, y_test, model):
	rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
	rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
	hub = huber(y_train, model.predict(X_train))
	hub_test = huber(y_test, model.predict(X_test))
	# transform
	hub = yscaler.inverse_transform(hub)
	hub_test = yscaler.inverse_transform(hub_test)
	rmse = yscaler.inverse_transform(rmse)
	rmse_test= yscaler.inverse_transform(rmse_test)
	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
	print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))



from sklearn.metrics import mean_squared_error

"""Rewrite Crossvalprdict """

def calc_train_error(X_train, y_train, model):
	'''returns in-sample error for already fit model.'''
	predictions = model.predict(X_train)
	mse = mean_squared_error(y_train, predictions)
	rmse = np.sqrt(mse)
	return mse

def calc_validation_error(X_test, y_test, model):
	'''returns out-of-sample error for already fit model.'''
	predictions = model.predict(X_test)
	mse = mean_squared_error(y_test, predictions)
	rmse = np.sqrt(mse)
	return mse

def calc_metrics(X_train, y_train, X_test, y_test, model):
	'''fits model and returns the RMSE for in-sample error and out-of-sample error'''
	#model.fit(X_train, y_train)
	train_error = calc_train_error(X_train, y_train, model)
	validation_error = calc_validation_error(X_test, y_test, model)
	print("Train Error:", train_error)
	print("Validation Error:", validation_error)

# calc_train_error(X_train, y_train, model)
# calc_validation_error(X_train, y_train, model)
# calc_metrics(X_train, y_train, X_test, y_test, model)

#args kwags
def benchmark_val(X_train,y_train,X_test, y_test, model, X_val = None, y_val = None):
	rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
	rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
	rmse_val = np.mean((y_val - model.predict(X_val).reshape(y_val.shape))**2)**0.5
	hub = huber(y_train, model.predict(X_train))
	hub_test = huber(y_test, model.predict(X_test))
	hub_val = huber(y_val, model.predict(X_val))
	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
	print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))
	print ("val\t%0.4F\t%0.4F"%(rmse_val, hub_val))

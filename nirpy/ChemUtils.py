#Spectral Utilities
#from __futur__ import print_function
import numpy as np
import scipy #TODO reimplement as Numpy only
from scipy import newaxis as nA
from scipy import linalg
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#Some metrics
def huber(y_true, y_pred, delta=1.0):
	y_true = y_true.reshape(-1,1)
	y_pred = y_pred.reshape(-1,1)
	return np.mean(delta**2*( (1+((y_true-y_pred)/delta)**2)**0.5 -1))


def benchmark(X_train,y_train,X_test, y_test, model):

	rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
	rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
	hub = huber(y_train, model.predict(X_train))
	hub_test = huber(y_test, model.predict(X_test))
	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
	print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))


#
# def benchmark(X_train,y_train,X_test, y_test, model, X_val = None, y_val = None):
#
# 	def huber(y_true, y_pred, delta=1.0):
# 		y_true = y_true.reshape(-1,1)
# 		y_pred = y_pred.reshape(-1,1)
# 		return np.mean(delta**2*( (1+((y_true-y_pred)/delta)**2)**0.5 -1))
#
#     rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
#     rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
#     hub = huber(y_train, model.predict(X_train))
#     hub_test = huber(y_test, model.predict(X_test))
#     print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
#     print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))


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

def spec_mean_centering(X):
	X_df = pd.DataFrame(X_train)
	return X_df.subtract(X_df.mean()).values

class SpectraMeanCenter():
	"""Applies mean centering to spectra"""
	def __init__(self):
		self.mean_spec = None
		self._fitted = False


	def fit(self, X):
		X_df = pd.DataFrame(X)
		self.mean_spec = X_df.mean()
		self._fitted = True

	def transform(self, X):
		X_df = pd.DataFrame(X)
		if self._fitted == True:
			return X_df.subtract(self.mean_spec).values
		else:
			print('Call .fit() first!')

	def fit_transform(X):
		X_df = pd.DataFrame(X_train)
		return X_df.subtract(X_df.mean()).values





class GlobalStandardScaler(object):
	"""Scales to unit standard deviation and mean centering using global mean and std of X, skleran like API"""
	def __init__(self,with_mean=True, with_std=True, normfact=1.0):
		self._with_mean = with_mean
		self._with_std = with_std
		self.std = None
		self.normfact=normfact
		self.mean = None
		self._fitted = False

	def fit(self,X, y = None):
		X = np.array(X)
		self.mean = X.mean()
		self.std = X.std()
		self._fitted = True

	def transform(self,X, y=None):
		if self._fitted:
			X = np.array(X)
			if self._with_mean:
				X=X-self.mean
			if self._with_std:
				X=X/(self.std*self.normfact)
			return X
		else:
			print("Scaler is not fitted")
			return

	def inverse_transform(self,X, y=None):
		if self._fitted:
			X = np.array(X)
			if self._with_std:
				X=X*self.std*self.normfact
			if self._with_mean:
				X=X+self.mean
			return X
		else:
			print("Scaler is not fitted")
			return

	def fit_transform(self,X, y=None):
		self.fit(X)
		return self.transform(X)




class SavgolFilter(FunctionTransformer):
	'''Fit transform Input with Savgol Alogorithm'''
	def __init__(self, window_length=13, polyorder=2, deriv=0):
		# set parameters
		self.window_length = window_length
		self.polyorder = polyorder
		self.deriv = deriv

	def fit(self, X, y=None):
		X_sg = savgol_filter(X, self.window_length, self.polyorder, self.deriv)
		return X_sg

	def transform(self, X ,y = None):
		return self.fit(X)

	def fit_transform(self, X, y=None):
		return self.fit(X)





class EmscScaler(object):
	def __init__(self,order=1):
		self.order = order
		self._mx = None

	def mlr(self,x,y):
		"""Multiple linear regression fit of the columns of matrix x
		(dependent variables) to constituent vector y (independent variables)

		order -     order of a smoothing polynomial, which can be included
					in the set of independent variables. If order is
					not specified, no background will be included.
		b -         fit coeffs
		f -         fit result (m x 1 column vector)
		r -         residual   (m x 1 column vector)
		"""

		if self.order > 0:
			s=scipy.ones((len(y),1))
			for j in range(self.order):
				s=scipy.concatenate((s,(scipy.arange(0,1+(1.0/(len(y)-1)),1.0/(len(y)-1))**j)[:,nA]),1)
			X=scipy.concatenate((x, s),1)
		else:
			X = x

		#calc fit b=fit coefficients
		b = scipy.dot(scipy.dot(scipy.linalg.pinv(scipy.dot(scipy.transpose(X),X)),scipy.transpose(X)),y)
		f = scipy.dot(X,b)
		r = y - f

		return b,f,r


	def inverse_transform(self, X, y=None):
		print("Warning: inverse transform not possible with Emsc")
		return X

	def fit(self, X, y=None):
		"""fit to X (get average spectrum), y is a passthrough for pipeline compatibility"""
		self._mx = scipy.mean(X,axis=0)[:,nA]

	def transform(self, X, y=None, copy=None):
		if type(self._mx) == type(None):
			print("EMSC not fit yet. run .fit method on reference spectra")
		else:
			#do fitting
			corr = scipy.zeros(X.shape)
			for i in range(len(X)):
				b,f,r = self.mlr(self._mx, X[i,:][:,nA])
				corr[i,:] = scipy.reshape((r/b[0,0]) + self._mx, (corr.shape[1],))
			return corr

	def fit_transform(self, X, y=None):
		self.fit(X)
		return self.transform(X)




class Dataaugument(object):

	'''Using dataaugmentation on input spectra to create slightly altered spectra to train NeuralNet'''

	def __init__(self, repeats=100, betashift = 0.05, slopeshift = 0.05, multishift = 0.05):
		self.repeats = repeats
		self.betashift = betashift
		self.slopeshift = slopeshift
		self.multishift = multishift

	def repeat(self, X, y = None):
		X = np.repeat(X, self.repeats, axis=0)
		return X

	def augment(self, X, y= None):
		self.betashift
		self.slopeshift
		self.multishift
		#Shift of baseline
		#calculate arrays
		beta = np.random.random(size=(X.shape[0],1))*2*self.betashift-self.betashift
		slope = np.random.random(size=(X.shape[0],1))*2*self.slopeshift-self.slopeshift + 1
		#Calculate relative position
		axis = np.array(range(X.shape[1]))/float(X.shape[1])
		#Calculate offset to be added
		offset = slope*(axis) + beta - axis - slope/2. + 0.5

		#Multiplicative
		multi = np.random.random(size=(X.shape[0],1))*2*self.multishift-self.multishift + 1

		X = multi*X + offset

		return X

	def fit(self, X, y = None):
		return self.repeat(X)
		# when passing test data to pipeline, we dont want it to be augmentet
	def transform(self, X, y=None):
		return X

	def fit_transform(self, X, y= None):
		X = self.fit(X)
		return self.augment(X)


if __name__ == "__main__":
	from sys import stdout
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from scipy.signal import savgol_filter
	from sklearn.cross_decomposition import PLSRegression
	from sklearn.model_selection import cross_val_predict
	from sklearn.metrics import mean_squared_error, r2_score
	from sklearn.pipeline import Pipeline
	from sklearn.model_selection import train_test_split


	#specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv') # cut spectra
	specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv') # full spectra
	lab = pd.read_excel('./luzrawSpectra/labdata.xlsx')

	from import_Module import importLuzCol

	X, y, wl = importLuzCol(specs, lab, 4)

	#from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter, MeanCenter



	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
	# scale y
	yscaler = GlobalStandardScaler()
	y_train = yscaler.fit_transform(y_train)
	y_test = yscaler.transform(y_test)
	y_train.mean()
	#np.histogram(y_train)
	y_test.mean()
	#Extensive Multiplicative Scattercorrection
	#plt.hist(y_train)
	#plt.hist(y_test)


	pipeline = Pipeline([
		("scaleing_X", GlobalStandardScaler()),
		("scatter_correction", EmscScaler()),
		("smmothing", SavgolFilter(deriv=0))
	])

	pipe_mean = Pipeline([
		("scatter_correction", EmscScaler()),
		("meancentering", SpectraMeanCenter()),
		("smmothing", SavgolFilter(deriv=0)),
	])



	X_train = pipeline.fit_transform(X_train)
	X_m = pipe_mean.fit_transform(X)

	meaner = MeanCenter()
	meaner.fit(X)
	meaner.transform(X)
	meaner.fit_transform(X)
	#
	#     from sklearn.pipeline import Pipeline
	#
	#     def create_pipe(pipeline):
	#         pipeline = Pipeline([
	#             ("scaleing_X", GlobalStandardScaler()),
	#             ("scatter_correction", EmscScaler()),
	#             ("smmothing", SavgolFilter())
	#             ])
	#         return pipeline
	#
	#     def create_aug_pipe(aug_pipeline):
	#         aug_pipeline = Pipeline([
	#             ("scaleing_X", GlobalStandardScaler()),
	#             ("dataaugmentation", Dataaugument())
	#             ("scatter_correction", EmscScaler()),
	#             ])
	#         return pipeline
	#
	#
	#
	#     specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv')
	#     lab = pd.read_excel('./luzrawSpectra/labdata.xlsx')
	#
	#     from import_luz import importLuz
	#     X, y_Xp, y_Xl, wl = importLuz(specs, lab)
	#
	#
	#     aug_pipline = Pipeline([
	#         ("scaleing_X", GlobalStandardScaler()),
	#         ("dataaug", Dataaugument()),
	#         ("scatter_correction", EmscScaler())
	#         ])
	#
	#
	#     X_pipe = aug_pipline.fit_transform(X)

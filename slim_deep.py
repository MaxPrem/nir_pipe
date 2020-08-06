# %% markdown
# First demo DeepLearning


# Set directory for model callbacks to be saved
# %%
import os
import time

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
import pkg_resources
import scipy.io as sio
import tensorflow as tf
#tensorflow
import tensorflow.keras as keras
import tensorflow.keras.losses
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import (cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
										ReduceLROnPlateau)
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten,
									 GaussianNoise, Reshape, SeparableConv1D)
from tensorflow.keras.models import Sequential, load_model, save_model
from nirpy.ChemUtils import (Dataaugument, EmscScaler, GlobalStandardScaler)
from nirpy.DeepUtils import CustomStopper, HuberLoss, MCDropout
from nirpy.ImportModule import cut_specs, importLuzCol
import tensorflow as tf

assert tf.__version__ >= "2.0"
print(tf.__version__)



# %%codecell
#specs = pd.read_csv('./nirpy/luzrawSpectra/nirMatrix.csv')
specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv') # full
lab = pd.read_excel('./nirpy/luzrawSpectra/labdata.xlsx')


specs = cut_specs(specs, 4100, 8000)

X, y, wl, ref = importLuzCol(specs, lab, 1)

# %%

X, X_val, y, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.35, random_state=42)



# %% codecell
#
# scale y
#from sklearn.preprocessing import RobustScaler
yscaler = GlobalStandardScaler()
y = yscaler.fit_transform(y)
y_test = yscaler.transform(y_test)
y_val = yscaler.transform(y_val)

y_test.shape
y_val.shape

y_aug = np.repeat(y, repeats=100, axis=0) #y is simply repeated

# y augmentation

# try to repat y with dataaug.fit()


aug_pipline = Pipeline([
	("scaleing_X", GlobalStandardScaler()),
	("dataaugmentation", Dataaugument()),
	("scatter_correction", EmscScaler())
	])
# Data augmentation is only applied when using fit_transfrom (on X_aug for training the model on more data)
# when plotting we prefere normaly fitted X without augmentation
X_aug = aug_pipline.fit_transform(X)

X_val = aug_pipline.transform(X_val)
X = aug_pipline.transform(X)
X_test = aug_pipline.transform(X_test)

X_aug.shape
X.shape
y.shape



data = {"X": X, "y": y, "X_test": X_test, "y_test": y_test, "X_val": X_val, "y_val": y_val}




# %%
###################################################
######## model initialization as funciton #########
###################################################
def create_model():

	# optimsed network shape of la
	DENSE = 128
	DROPOUT = 0.5
	C1_K  = 8   #Number of kernels/feature extractors for first layer
	C1_S  = 32  #Width of the convolutional mini networks
	C2_K  = 16
	C2_S  = 32

	#input
	input_dim = X_aug.shape[1]

    # activatoin function
	leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
	activation=leaky_relu
	kernel_initializer = "he_normal"


	model = keras.models.Sequential()

	model.add(GaussianNoise(0.05, input_shape=(input_dim,)))
	model.add(Reshape((input_dim, 1)))
	model.add(SeparableConv1D(C1_K, (C1_S),activation=activation, padding="same", kernel_initializer= kernel_initializer, use_bias=False, kernel_constraint=keras.constraints.max_norm(1.)))
	keras.layers.MaxPooling1D(pool_size=2),
	model.add(SeparableConv1D(C2_K, (C2_S), activation=activation, padding="same", kernel_initializer= kernel_initializer, use_bias=False, kernel_constraint=keras.constraints.max_norm(1.)))
	keras.layers.MaxPooling1D(pool_size=2),
	model.add(Flatten())
	model.add(MCDropout(DROPOUT))
	model.add(Dense(DENSE,activation=activation, kernel_constraint=keras.constraints.max_norm(1.)))
	model.add(MCDropout(DROPOUT))
	model.add(Dense(1, activation='linear', kernel_constraint=keras.constraints.max_norm(1.) ,use_bias=False))

	###########
	# sometimes model needs to be compiled outside of function
	model.compile(loss=HuberLoss(), optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999))
	return model

model = create_model()
#compile model
model.compile(loss=HuberLoss(), optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999))

model.summary()


# %%
checkpoint_name = "test_XA"
run_id_tensor_board = "run_%Y_%m_%d-%H_%M_%S_test_XA"

# tensorboard
def get_run_logdir():
	import time
	run_id = time.strftime(run_id_tensor_board)
	return os.path.join(root_logdir, run_id)

root_logdir = os.path.join(os.curdir, "my_logs")

run_logdir = get_run_logdir()
print(root_logdir)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
	print(entry_point.dist)

# %% cv predict

# from sklearn import metrics
# from ScoreUtils import huber_loss
# huber_score = metrics.make_scorer(huber_loss)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=create_model, epochs=45, batch_size=16, verbose=0)
# kfold = KFold(n_splits=5, shuffle=True)
#
# #results = cross_val_score(estimator, X_aug, y_aug, cv=kfold, scoring=huber_score )
# y_cv = cross_val_predict(estimator, X, y, cv=10)

#print("HuberScore: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



# %%
# define callbacks

rdlr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, monitor='val_loss', verbose=1)
checkpoint_cb = ModelCheckpoint(checkpoint_name, save_best_only=True, monitor='val_loss')
custom_stopper = CustomStopper(start_epoch = 60, monitor='val_loss', verbose=1, patience=16)

#early_stopping_cb = EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss')

# %% mark
# TRAIN MODEL

history = model.fit(X_aug, y_aug, epochs = 80, batch_size=16, validation_data=(X_test, y_test) , callbacks=[rdlr, custom_stopper, tensorboard_cb, checkpoint_cb])

# %%
# %%
"""improove early stopping """
#model.save("prot.h5")

#model = keras.models.load_model("./prot.h5", compile=False)


with plt.style.context("ggplot"):
	plt.plot(history.history['loss'], label='loss')
	plt.plot(history.history['val_loss'], label='val_loss')

	plt.yscale('log')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend()
	ax2 = plt.gca().twinx()
	ax2.plot(history.history['lr'], color='r')
	ax2.set_ylabel('lr',color='r')

	plt.legend()

# %%
from tensor_val_utils import *

def tensor_benchmark_model(
	X, y, X_test, y_test, model, ref, **kwargs
):
	"""Final Output Function for pls regression"""
	# get name of reference method for outputtable
	print_nir_metrics(X, y, X_test, y_test, model, ref, **kwargs)
	print("___Benchmarks___")
	print_tensor_benchmark(X, y, X_test, y_test, model, **kwargs)
	print("___MC dropout benchmarks___")
	print_tensor_benchmark(X, y, X_test, y_test, model, **kwargs, mc_dropout = True)


tensor_benchmark_model(**data, model = model, ref=ref)

# %%
mc_tensor_plot(**data, model = model)

# %%
"""improove early stopping """
#model.save("prot.h5")

#model = keras.models.load_model("./prot.h5", compile=False)


with plt.style.context("ggplot"):
	plt.plot(history.history['loss'], label='loss')
	plt.plot(history.history['val_loss'], label='val_loss')

	plt.yscale('log')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend()
	ax2 = plt.gca().twinx()
	ax2.plot(history.history['lr'], color='r')
	ax2.set_ylabel('lr',color='r')

	plt.legend()




tensor_benchmark_model(**data, model = model, ref=ref)


# %% markdown
# The "history" object returned by the training, contain the losses and learning rates from the training. The loss for the training and validation (here = Test) settles down when the lr is lowered. The model seems a bit overfit as the validation loss rises towards the end of training, maybe a higher dropout ratio could help.
# %% codecell

# loading a reconstructed model
# Loading best Checkpoint with custom loss

#reconstructed_model = keras.models.load_model('luz_XP', compile=False)
reconstructed_model = keras.models.load_model('test_XA', compile=False)

reconstructed_model.compile(loss=HuberLoss(), optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999))






mc_tensor_plot(**data, model = reconstructed_model, errorbar=True)



tensor_benchmark_model(**data, model = reconstructed_model, ref=ref)


# %% codecell


# %% markdown
# Saved output from benchmark fuction
#
# TM
# RMSE  Train/Test	0.87	0.91
# Huber Train/Test	0.2522	0.2767
#
#
# XA: Rohasche
# RMSE  Train/Test	0.75	0.60 # data is containing outlier, PLS OUtlier loop removed 9 samples
# Huber Train/Test	0.1734	0.1630
#
#
# RMSE  Train/Test	0.19	0.19
# Huber Train/Test	0.0215	0.0222
#
# # full spec - edges
# RMSE  Train/Test	0.21	0.23
# Huber Train/Test	0.0204	0.0296
#
# # XP part of the spec
# RMSE  Train/Test	0.98	0.24
# Huber Train/Test	0.3655	0.0202
#
#
# 'reconstructed_model' selu batch norm ...
# RMSE  Train/Test	0.12	0.15
# Huber Train/Test	0.0077	0.0108
#
# XP: Nadam 70 epochs
# RMSE  Train/Test	0.04	0.30
# Huber Train/Test	0.0009	0.0418
#
#
# XP: 100 epochs, use dataaug on validation_data
# and retrain for 2 epocks
# &seperabl   Conv
#
#
# RMSE  Train/Test	0.22	0.34
# Huber Train/Test	0.0228	0.0532
#
# XLP
#
# RMSE  Train/Test	0.54	0.74
# Huber Train/Test	0.1249	0.2278
# """
#
# """33 epochs
#
# RMSE  Train/Test	0.21	0.36
# Huber Train/Test	0.0214	0.0576

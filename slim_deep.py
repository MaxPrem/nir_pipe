# %% markdown
# First demo DeepLearning


# %%codecell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

#tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, Reshape, GaussianNoise, SeparableConv1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf

import tensorflow.keras.losses
from tensorflow.keras import backend as K

assert tf.__version__ >= "2.0"
print(tf.__version__)



# %%codecell
#specs = pd.read_csv('./nirpy/luzrawSpectra/nirMatrix.csv')
specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv') # full
lab = pd.read_excel('./nirpy/luzrawSpectra/labdata.xlsx')

from nirpy.import_Module import importLuzCol, cut_specs

specs = cut_specs(specs, 4100, 8000)

X, y, wl, ref = importLuzCol(specs, lab, 4)



# %%codecell
from sklearn.model_selection import train_test_split
X, X_val, y, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


from nirpy.ChemUtils import GlobalStandardScaler, Dataaugument, EmscScaler
from sklearn.pipeline import Pipeline

# %% codecell
#
# scale y
#from sklearn.preprocessing import RobustScaler
yscaler = GlobalStandardScaler()
y_train = yscaler.fit_transform(y_train)
y_test = yscaler.transform(y_test)
y_val = yscaler.transform(y_val)




# y augmentation
y_aug = np.repeat(y_train, repeats=100, axis=0) #y_train is simply repeated

# try to repat y with dataaug.fit()


aug_pipline = Pipeline([
	("scaleing_X", GlobalStandardScaler()),
	("dataaugmentation", Dataaugument()),
	("scatter_correction", EmscScaler())
	])
# Data augmentation is only applied when using fit_transfrom (on X_aug for training the model on more data)
# when plotting we prefere normaly fitted X_train without augmentation
X_aug = aug_pipline.fit_transform(X_train)

X_val = aug_pipline.transform(X_val)
X_train = aug_pipline.transform(X_train)
X_test = aug_pipline.transform(X_test)

X_aug.shape
X_train.shape
y_train.shape



# %% codecell

#Hyperparameters for the network
DENSE = 128
DROPOUT = 0.5
C1_K  = 8   #Number of kernels/feature extractors for first layer
C1_S  = 32  #Width of the convolutional mini networks
C2_K  = 16
C2_S  = 32

#input
input_dim = X_aug.shape[1]

'''leakyrelu'''

leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
activation=leaky_relu
#activation='relu'
kernel_initializer = "he_normal"

'''selu'''
#For SELU activation, just set activation="selu" and kernel_initial izer="lecun_normal" when creating a layer:

activation = 'selu'
kernel_initializer = 'lecun_normal'

#example: layer = keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")

from nirpy.deep_utils import MCDropout, HuberLoss, CustomStopper



model = keras.models.Sequential([
	GaussianNoise(0.05, input_shape=(input_dim,)),
	Reshape((input_dim, 1)),
	SeparableConv1D(C1_K, (C1_S), padding="same", kernel_initializer= kernel_initializer, use_bias=False, kernel_constraint=keras.constraints.max_norm(1.)),
	keras.layers.Activation(activation),
	SeparableConv1D(C2_K, (C2_S), padding="same", kernel_initializer= kernel_initializer, use_bias=False, kernel_constraint=keras.constraints.max_norm(1.)),
	keras.layers.Activation(activation),
	Flatten(),
	MCDropout(DROPOUT),
	Dense(DENSE, kernel_constraint=keras.constraints.max_norm(1.)),
	keras.layers.Activation(activation),
	MCDropout(DROPOUT),
	Dense(1, activation='linear', kernel_constraint=keras.constraints.max_norm(1.) ,use_bias=False)
])


########################################################################################################################
###########################################     cross_validate   #######################################################
########################################################################################################################

from sklearn.model_selection import cross_val_score

###################################################
######## model initialization as funciton #########
###################################################


def create_model():

    DENSE = 128
    DROPOUT = 0.5
    C1_K  = 8   #Number of kernels/feature extractors for first layer
    C1_S  = 32  #Width of the convolutional mini networks
    C2_K  = 16
    C2_S  = 32

    #input
    input_dim = X_aug.shape[1]

    '''leakyrelu'''

    leaky_relu = keras.layers.LeakyReLU(alpha=0.2)
    activation=leaky_relu
    #activation='relu'
    kernel_initializer = "he_normal"

    '''selu'''
    #For SELU activation, just set activation="selu" and kernel_initial izer="lecun_normal" when creating a layer:

    activation = 'selu'
    kernel_initializer = 'lecun_normal'

    model = keras.models.Sequential()

    model.add(GaussianNoise(0.05, input_shape=(input_dim,)))
    model.add(Reshape((input_dim, 1)))
    model.add(SeparableConv1D(C1_K, (C1_S), padding="same", kernel_initializer= kernel_initializer, use_bias=False, kernel_constraint=keras.constraints.max_norm(1.)))
    model.add(keras.layers.Activation(activation))
    model.add(SeparableConv1D(C2_K, (C2_S), padding="same", kernel_initializer= kernel_initializer, use_bias=False, kernel_constraint=keras.constraints.max_norm(1.)))
    model.add(keras.layers.Activation(activation))
    model.add(Flatten())
    model.add(MCDropout(DROPOUT))
    model.add(Dense(DENSE, kernel_constraint=keras.constraints.max_norm(1.)))
    model.add(keras.layers.Activation(activation))
    model.add(MCDropout(DROPOUT))
    model.add(Dense(1, activation='linear', kernel_constraint=keras.constraints.max_norm(1.) ,use_bias=False))

    ###########

    #model.compile(loss=HuberLoss(), optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999))
    return model


#compile model
model.compile(loss=HuberLoss(), optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999))


#####################################################
############cv-predict/score TF######################
#####################################################

#huber_score = metrics.make_scorer(huber)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=create_model, epochs=45, batch_size=16, verbose=0)
# kfold = KFold(n_splits=5, shuffle=True)
#
# results = cross_val_score(estimator, X_aug, y_aug, cv=kfold, scoring=huber_score )
# results
# print("HuberScore: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


###############
#compile model#
###############

# model.compile(loss=HuberLoss(), optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999))

############
#set run id#
############

checkpoint_name = "Xl"
run_id_tensor_board = "run_%Y_%m_%d-%H_%M_%S_prot_XL"


############
#**********#
############

# Set directory for model callbacks to be saved
import os
import time
import pkg_resources

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
X_aug.shape


model.summary()
######################
# defineing Callbacks#
######################

rdlr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, monitor='val_loss', verbose=1)
checkpoint_cb = ModelCheckpoint(checkpoint_name, save_best_only=True, monitor='val_loss')
custom_stopper = CustomStopper(start_epoch = 40, monitor='val_loss', verbose=1, patience=16)

#early_stopping_cb = EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss')



##############
# TRAIN MODEL#
##############

history = model.fit(X_aug, y_aug, epochs = 70, batch_size=16, validation_data=(X_test, y_test) , callbacks=[rdlr, custom_stopper, tensorboard_cb, checkpoint_cb])
#
# with keras.backend.learning_phase_scope(1): # force training mode = dropout on
#     y_probas = np.stack([model.predict(X_test)
#     for sample in range(100)]):
# y_proba = y_probas.mean(axis=0)


# Load the TensorBoard notebook extension
# %load_ext tensorboard
# %tensorboard -logdir=./my_logs
#
#
# import pkg_resources
#
# for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
#     print(entry_point.dist)
#

"""improove early stopping """
#model.save("dataaug_luz.h5")

#model = keras.models.load_model("./dataaug_luz.h5"

# %%codecell
import pandas as pd
with plt.style.context("ggplot"):
	pd.DataFrame(history.history).plot()
	plt.grid(True)
	#plt.gca().set_ylim(-0.2, 1) # set the vertical range to [0-1] plt.show()


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

from nirpy.ChemUtils import huber, benchmark

benchmark(X_train, y_train, X_test, y_test, model)



'''add a blue test set line???'''
with plt.style.context("ggplot"):
	plt.scatter(y_train, model.predict(X_train))
	plt.scatter(y_test, model.predict(X_test))

	plt.xlabel('measured')
	plt.ylabel('predicted')
	#plt.plot(y_train, y_train) # Y = PredY line
	plt.plot([-2,3],[-2,3])
	#plt.plot(y_test, y_test) # Y = PredY line

from nirpy.validation_utils import tensor_benchmark
# %% markdown
# The "history" object returned by the training, contain the losses and learning rates from the training. The loss for the training and validation (here = Test) settles down when the lr is lowered. The model seems a bit overfit as the validation loss rises towards the end of training, maybe a higher dropout ratio could help.
# %% codecell
X_train.shape
X_test.shape


cv_bench(X_train, y_train, X_test, y_test, model)


#################################
##loading a reconstructed model##
#################################

'''Loading best Checkpoint with custom loss'''

#reconstructed_model = keras.models.load_model('luz_XP', compile=False)
reconstructed_model = keras.models.load_model(checkpoint_name, compile=False)

reconstructed_model.compile(loss=HuberLoss(), optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999))


'''reconstructed_model'''
#benchmark(X_train, y_train, X_test, y_test, reconstructed_model)
from nirpy.validation_utils import da_func

da_func2(X_train, y_train, X_test, y_test, model)

from sklearn.model_selection import cross_val_predict

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



'''Train reconstructed model'''

history = reconstructed_model.fit(X_aug, y_aug, epochs = 50, batch_size=16, validation_data=(X_test, y_test), callbacks=[rdlr, checkpoint_cb, custom_stopper, tensorboard_cb])

# oberseving similiar effekt to double cross validation
'''add a blue test set line???'''
with plt.style.context("ggplot"):
	plt.scatter(y_train, model.predict(X_train))
	plt.scatter(y_test, model.predict(X_test))
	plt.scatter(y_val, model.predict(X_val))
	#plt.scatter(y, reconstructed_model.predict(X))
	plt.xlabel('measured')
	plt.ylabel('predicted')
	#plt.plot(y_train, y_train) # Y = PredY line
	plt.plot([-2,3],[-2,3])
	# plt.plot(y_test, y_test) # Y = PredY line



'''add a blue test set line???'''
with plt.style.context("ggplot"):
	plt.scatter(y_train, reconstructed_model.predict(X_train))
	plt.scatter(y_test, reconstructed_model.predict(X_test))
	#plt.scatter(y_val, model.predict(X_val))
	#plt.scatter(y, reconstructed_model.predict(X))
	plt.xlabel('protein_measured')
	plt.ylabel('protein_predicted')
	# plt.plot(y_test, y_test) # Y = PredY line
	plt.plot([-2,3],[-2,3])




# %% codecell
from nirpy.ChemUtils import benchmark, scaled_benchmark, huber

from keras.wrappers.scikit_learn import KerasRegressor


# %% codecell

benchmark(X_train, y_train, X_test, y_test, model, X_val, y_val)


'''reconstructed_model'''
benchmark(X_train, y_train, X_test, y_test, reconstructed_model, X_val, y_val)





SEP(X_train, y_train, model)

SEC(X_train, y_train, model)

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

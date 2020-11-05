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
from sklearn import metrics
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     train_test_split, KFold)
from sklearn.pipeline import Pipeline
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten,
                                     GaussianNoise, Reshape, SeparableConv1D)
from tensorflow.keras.models import Sequential, load_model, save_model

from nirpy.ChemUtils import Dataaugument, EmscScaler, GlobalStandardScaler
from nirpy.DeepUtils import CustomStopper, HuberLoss, MCDropout
from nirpy.ImportModule import cut_specs, importLuzCol
from nirpy.ValidationUtils import (benchmark, cv_benchmark_model,
                                   print_nir_metrics, val_regression_plot)
from nirpy.ScoreUtils import huber_loss
from TensorValUtils import mc_benchmarks

assert tf.__version__ >= "2.0"
print(tf.__version__)
print(keras.__version__)

# %%


checkpoint_name = "XP"
run_id_tensor_board = "run_%Y_%m_%d-%H_%M_%S_prot_XP"

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


# %%

'''Loading best Checkpoint with custom loss'''

#reconstructed_model = keras.models.load_model('luz_XP', compile=False)
reconstructed_model = keras.models.load_model(checkpoint_name, compile=False)

reconstructed_model.compile(loss=HuberLoss(), optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999))



###################################################
##########cv-predict/score TF######################
###################################################

##############################################
# %%

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

    model.compile(loss=HuberLoss(), optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999))
    return model
# %%
 mc_benchmarks()



# %%
#
# huber_score = metrics.make_scorer(huber_loss)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=create_model, epochs=45, batch_size=16, verbose=0)
# kfold = KFold(n_splits=5, shuffle=True)
#
# results = cross_val_score(estimator, X_aug, y_aug, cv=kfold, scoring=huber_score )
# results
# print("HuberScore: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#


# compile model#


# model.compile(loss=HuberLoss(), optimizer =


# %%

# trying mc DROPOUT

 # force training mode = dropout on  # force training mode = dropout on
with keras.backend.learning_phase_scope(1):
    y_probas = np.stack([model.predict(X_test)
        for sample in range(100)])
    y_proba = y_probas.mean(axis=0)
X_test.shape
y_test
y_proba.T
y_proba.shape

y_proba.mean()
y_proba.std()


ci = 0.95
lower_lim = np.quantile(y_proba, 0.5-ci/2, axis=1)
upper_lim = np.quantile(y_proba, 0.5+ci/2, axis=1)

lower_lim
upper_lim

lower_lim==upper_lim

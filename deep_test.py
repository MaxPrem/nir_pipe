
# %% markdown
# First demo DeepLearning


## Set directory for model callbacks to be saved
# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import scipy.io as sio
import tensorflow as tf
# tensorflow
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.losses
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import metrics
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import (cross_val_predict, cross_val_score,
									 cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
										ReduceLROnPlateau)
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten,
									 GaussianNoise, Reshape, SeparableConv1D)
from tensorflow.keras.models import Sequential, load_model, save_model

from nirpy.ChemUtils import (Dataaugument, EmscScaler, GlobalStandardScaler, huber)
from nirpy.DeepUtils import CustomStopper, HuberLoss, MCDropout
from nirpy.ImportModule import cut_specs, importLuzCol
from nirpy.ScoreUtils import huber_loss, root_mean_squared_error, standard_error_calibration, relative_prediction_deviation, standard_error_prediction, standard_error_cross_validation

assert tf.__version__ >= "2.0"
print(tf.__version__)


# %%codecell
# specs = pd.read_csv('./nirpy/luzrawSpectra/nirMatrix.csv')
specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")  # full
lab = pd.read_excel("./luzrawSpectra/labdata.xlsx")


specs = cut_specs(specs, 4100, 8000)

X, y, wl, ref = importLuzCol(specs, lab, 6)


X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# %% codecell
#
# scale y
# from sklearn.preprocessing import RobustScaler
yscaler = GlobalStandardScaler()
y = yscaler.fit_transform(y)
y_test = yscaler.transform(y_test)
y_val = yscaler.transform(y_val)


# y augmentation
y_aug = np.repeat(y, repeats=100, axis=0)  # y is simply repeated

# try to repat y with dataaug.fit()


aug_pipline = Pipeline(
	[
		("scaleing_X", GlobalStandardScaler()),
		("dataaugmentation", Dataaugument()),
		("scatter_correction", EmscScaler()),
	]
)
# Data augmentation is only applied when using fit_transfrom (on X_aug for training the model on more data)
# when plotting we prefere normaly fitted X without augmentation
X_aug = aug_pipline.fit_transform(X)

X_val = aug_pipline.transform(X_val)
X = aug_pipline.transform(X)
X_test = aug_pipline.transform(X_test)

X_aug.shape
X.shape
y.shape


data_val = {
	"X": X,
	"y": y,
	"X_test": X_test,
	"y_test": y_test,
	"X_val": X_val,
	"y_val": y_val,
}


data = {"X": X, "y": y, "X_test": X_test, "y_test": y_test}


# %%

checkpoint_name = "test_XB"
run_id_tensor_board = "run_%Y_%m_%d-%H_%M_%S_prot_XB"

# tensorboard
def get_run_logdir():
	import time

	run_id = time.strftime(run_id_tensor_board)
	return os.path.join(root_logdir, run_id)


root_logdir = os.path.join(os.curdir, "my_logs")

run_logdir = get_run_logdir()
print(root_logdir)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


for entry_point in pkg_resources.iter_entry_points("tensorboard_plugins"):
	print(entry_point.dist)
X_aug.shape


# %%
# Loading best Checkpoint with custom loss

# reconstructed_model = keras.models.load_model('luz_XP', compile=False)
reconstructed_model = keras.models.load_model('test_XB', compile=False)

reconstructed_model.compile(
	loss=HuberLoss(),
	optimizer=keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999),
)

model = reconstructed_model

# %%
from TensorValUtils import *

tensor_benchmark_model(X,y, X_test, y_test, model= model, ref=ref)
# %%
print_tensor_benchmark(**data_val, model=model)



print_mc_tensor_benchmark(**data_val, model=model)



def func_print():
	print("{:^30}".format("Scores and Metrics MC-Dropout"))


func_print()

# %%

def tensor_print(X, y, X_test, y_test, model, ref,  X_val=None, y_val=None, **kwargs):
	print_nir_metrics(X, y, X_test, y_test, model=model, ref=ref, **kwargs)
	print("\t\t\t{:^30}\t\t\t\t\t\t".format("____________Scores and Metrics MC-Dropout____________"))
	print_tensor_benchmark(X, y, X_test, y_test, model, X_val=None, y_val=None, **kwargs)


tensor_print(**data_val, model = model, ref=ref)
# %%
mc_tensor_plot(**data_val, model = model)

# %%

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


tensor_benchmark_model(**data_val, model = model, ref=ref)
# %%

# %%

# %% markdown

# We’ve looked at how we can use Dropout as a way to estimate of model uncertainty at prediction time. This technique is formally known as MC Dropout, and was developed by Yarin Gal, while he was completing his PhD at Cambridge.
# This approach circumvents the computational bottlenecks associated with having to train an ensemble of Neural Networks in order to estimate predictive uncertainty.

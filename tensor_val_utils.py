import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
	explained_variance_score,
	make_scorer,
	mean_squared_error,
	r2_score,
)
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

from nirpy.ScoreUtils import (
	huber_loss,
	relative_prediction_deviation,
	root_mean_squared_error,
	standard_error_calibration,
	standard_error_cross_validation,
	standard_error_prediction,
)


def tensor_benchmark_model(
	X, y, X_test, y_test, model, ref, **kwargs
):
	"""Final Output Function for CNN regression"""
	# get name of reference method for outputtable
	print_nir_metrics(X, y, X_test, y_test, model, ref, **kwargs)
	print("___Benchmarks___")
	print_tensor_benchmark(X, y, X_test, y_test, model, **kwargs)
	print("___MC dropout benchmarks___")
	print_tensor_benchmark(X, y, X_test, y_test, model, **kwargs, mc_dropout = True)


def mc_benchmarks(X, y, model, **kwargs):
	"""calculates regression metrics and scores"""
	hub = []
	mse = []
	score = []
	var = []

	for _ in range(100):


				hub.append(huber_loss(y, model.predict(X)))
				mse.append(mean_squared_error(y, model.predict(X)))
				score.append(r2_score(y, model.predict(X)))
				var.append(explained_variance_score(y, model.predict(X)))

	return hub, mse, score, var


#### Reg Metric Functions ####

def calculate_regression_benchmarks(X, y, model, mc_dropout_estimates=True):
	"""calculates regression metrics and scores"""
	if mc_dropout_estimates == False:
		# y_predalibration estimate
		#with keras.backend.learning_phase_scope(0):
		y_pred = model.predict(X)
	# cv estimate for plotting
	else:
		#with keras.backend.learning_phase_scope(1):

		y_pred = predict_mc_dropout(X, model)
		y_pred = y_pred["mean"].values

	# Calculate scores for calibration and cross-validation
	hub = huber_loss(y, y_pred)
	mse = mean_squared_error(y, y_pred)
	score = r2_score(y, y_pred)
	var = explained_variance_score(y, y_pred)

	return hub, mse, score, var




def print_tensor_benchmark(
	X, y, X_test, y_test, model, X_val=None, y_val=None, **kwargs
):
	"""prints regression score and metrics, takes validation data as optional argument"""

	hub, mse, score, var = calculate_regression_benchmarks(X, y, model)

	hub_test, mse_test, score_test, var_test = calculate_regression_benchmarks(
		X_test, y_test, model
	)

	if X_val is not None:
		hub_val, mse_val, score_val, var_val = calculate_regression_benchmarks(
			X_val, y_val, model
		)
		print("{:<9}\t{:^5}\t\t{:^5}\t\t{:^5}".format("Scores", "train", "test", "val"))
		print(
			"{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format("Var. expl.", var, var_test, var_val)
		)
		print("{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format("R2", score, score_test, score_val))
		print("{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format("MSE", mse, mse_test, mse_val))
		print(
			"{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format(
				"RMSE", np.sqrt(mse), np.sqrt(mse_test), np.sqrt(mse_val)
			)
		)
		print("{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format("Huber", hub, hub_test, hub_val))

	else:
		# print statements without validation data
		print("{:<9}\t{:^5}\t\t{:^5}".format("Scores", "train", "test"))
		print("{:<9}\t{:.3}\t\t{:.3}".format("Var", var, var_test))
		print("{:<9}\t{:.3}\t\t{:.3}".format("R2", score, score_test))
		print("{:<9}\t{:.3}\t\t{:.3}".format("MSE", mse, mse_test))
		print("{:<9}\t{:.3}\t\t{:.3}".format("RMSE", np.sqrt(mse), np.sqrt(mse_test)))
		print("{:<9}\t{:.3}\t\t{:.3}".format("Huber", hub, hub_test))


# def mc_calculate_regression_benchmarks(X, y, model):
#     """calculates regression metrics using monte carlo dropout estimations"""
#
#     y_mc_pred = predict_mc_dropout(X, model)
#     # cv estimate for plotting
#
#     # Calculate scores for calibration and cross-validation
#     hub = huber_loss(y, y_mc_pred["mean"].values)
#     mse = mean_squared_error(y, y_mc_pred["mean"])
#     score = r2_score(y, y_mc_pred["mean"])
#     var = explained_variance_score(y, y_mc_pred["mean"])
#
#     return hub, mse, score, var


# def print_mc_tensor_benchmark(
#     X, y, X_test, y_test, model, X_val=None, y_val=None, **kwargs
# ):
#     """prints regression score and metrics using monte carlo dropout predictions, takes validation data as optional input"""
#
#     hub, mse, score, var = mc_calculate_regression_benchmarks(X, y, model)
#
#     hub_test, mse_test, score_test, var_test = mc_calculate_regression_benchmarks(
#         X_test, y_test, model
#     )
#
#     if X_val is not None:
#         hub_val, mse_val, score_val, var_val = mc_calculate_regression_benchmarks(
#             X_val, y_val, model
#         )
#         # printing
#         print(
#             "{:<9}\t{:^5}\t\t{:^5}\t\t{:^5}".format("MC Scores", "train", "test", "val")
#         )
#         print(
#             "{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format(
#                 "MC Var. expl.", var, var_test, var_val
#             )
#         )
#         print(
#             "{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format(
#                 "MC R2", score, score_test, score_val
#             )
#         )
#         print("{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format("MC MSE", mse, mse_test, mse_val))
#         print(
#             "{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format(
#                 "MC RMSE", np.sqrt(mse), np.sqrt(mse_test), np.sqrt(mse_val)
#             )
#         )
#         print(
#             "{:<9}\t{:.3}\t\t{:.3}\t\t{:.3}".format("MC Huber", hub, hub_test, hub_val)
#         )
#
#     else:
#         print("{:<9}\t{:^5}\t\t{:^5}".format("MC Scores", "train", "test"))
#         print("{:<9}\t{:.3}\t\t{:.3}".format("MC Var. expl.", var, var_test))
#         print("{:<9}\t{:.3}\t\t{:.3}".format("MC R2", score, score_test))
#         print("{:<9}\t{:.3}\t\t{:.3}".format("MC MSE", mse, mse_test))
#         print(
#             "{:<9}\t{:.3}\t\t{:.3}".format("MC RMSE", np.sqrt(mse), np.sqrt(mse_test))
#         )
#         print("{:<9}\t{:.3}\t\t{:.3}".format("MC Huber", hub, hub_test))


def print_nir_metrics(
	X, y, X_test, y_test, model, ref, X_val=None, y_val=None, **kwargs
):
	"""prints summary nir calibration"""

	sep_test = standard_error_prediction(X_test, y_test, model)

	# pls model meta_data
	print("*** Summary Reference Method:", ref, "***")
	print("Number of wavelengths:", X.shape[1])
	print("Number of training sampels:", y.shape[0])
	print("Number of test sampels:", y_test.shape[0])
	if y_val is not None:
		print("Number of validation sampels:", y_val.shape[0])
	# nir metrics
	print("*** NIR Metrics ***")
	print("Standard Error Prediction (SEP): \t{:.3}".format(sep_test))


def tensor_benchmark_model(
	X, y, X_test, y_test, model, ref, X_val=None, y_val=None, **kwargs
):
	"""Final Output Function for pls regression"""
	# get name of reference method for outputtable
	print_nir_metrics(
		X, y, X_test, y_test, model, ref, X_val=None, y_val=None, **kwargs
	)
	print_tensor_benchmark(
		X, y, X_test, y_test, model, X_val=None, y_val=None, **kwargs
	)
	print_mc_tensor_benchmark(X, y, X_test, y_test, model)
	# print crosstable
	# print_cv_table(X, y, X_test, y_test, model)


def val_tensor_plot(X, y, X_test, y_test, model, X_val=None, y_val=None, **kwargs):
	"""regression plot showing train,  test, validation data"""
	with keras.backend.learning_phase_scope(0):
		y_pred = model.predict(X)
		z = np.polyfit(y, y_pred, 1)
		with plt.style.context(("ggplot")):
			fig, ax = plt.subplots(figsize=(9, 5))
			ax.scatter(y, model.predict(X), edgecolors="k", label="train")
			ax.scatter(y_test, model.predict(X_test), edgecolors="k", label="test")
			if X_val is not None:
				ax.scatter(y_val, model.predict(X_val), edgecolors="k", label="val")

			# ax.plot(z[1] + z[0] * y, y)
			ax.plot(y, y, color="blue")
			plt.title("R:" + str(r2_score(y, y_pred).mean().round(2)))
			plt.xlabel("Predicted")
			plt.ylabel("Measured")
			ax.legend()

			plt.show()


def mc_tensor_plot(
	X, y, X_test, y_test, model, X_val=None, y_val=None, errorbar=True, **kwargs
):
	"""scatter plot showing monte carlo estimates for train & test data"""
	mc_dropout_estimates = predict_mc_dropout(X, model)
	mc_dropout_test = predict_mc_dropout(X_test, model)

	if errorbar == True:
		with plt.style.context(("ggplot")):
			fig, ax = plt.subplots(figsize=(9, 5))
				# scatterplots + errobars
			ax.errorbar(
				y,
				mc_dropout_estimates["mean"],
				yerr=mc_dropout_estimates["std"],
				fmt="o",
				mfc="red",
				color="k",
				capthick=2,
				elinewidth=2,
				zorder=10,
				label="train",
			)

			ax.errorbar(
				y_test,
				mc_dropout_test["mean"],
				yerr=mc_dropout_test["std"],
				fmt="o",
				mfc="#1f77b4",
				color="k",
				capthick=2,
				elinewidth=2,
				zorder=10,
				label="test",
			)

			if X_val is not None:
				# validation_data
				mc_dropout_val = predict_mc_dropout(X_val, model)
				ax.errorbar(
					y_val,
					mc_dropout_val["mean"],
					yerr=mc_dropout_val["std"],
					fmt="o",
					mfc="grey",
					color="k",
					capthick=2,
					elinewidth=2,
					zorder=10,
					label="val",
				)
				ax.plot(y, y, color="blue")
				z = np.polyfit(mc_dropout_estimates["mean"], y, 1)
				plt.plot(z[1] + z[0] * y, y, color="red")
				plt.title("$R^{2}$:" + str(r2_score(y, mc_dropout_estimates["mean"]).mean().round(2)))
				plt.xlabel("Predicted")
				plt.ylabel("Measured")
				ax.legend()
				plt.show()

	else:
		# scatterplots
		with plt.style.context(("ggplot")):
			fig, ax = plt.subplots(figsize=(9, 5))
			ax.scatter(y, mc_dropout_estimates["mean"], edgecolors="k", label="train")
			ax.scatter(y_test, mc_dropout_test["mean"], edgecolors="k", label="test")
			if X_val is not None:
				# validation data
				mc_dropout_val = predict_mc_dropout(X_val, model)
				ax.scatter(y_val, mc_dropout_val["mean"], edgecolors="k", label="val")


			ax.plot(y, y, color="blue")
			z = np.polyfit(mc_dropout_estimates["mean"], y, 1)
			plt.plot(z[1] + z[0] * y, y, color="red")
			plt.title("$R^{2}$:" + str(r2_score(y, mc_dropout_estimates["mean"]).mean().round(2)))
			plt.xlabel("Predicted")
			plt.ylabel("Measured")
			ax.legend()
			plt.show()


def predict_mc_dropout(X, model, n_predictions = 200):
	"""performs monte carlo dropout on weights to estimate prediciton uncertainty"""
	predictions = []
	# sample_size = X.shape[0]

	for t in range(n_predictions):
		predictions.append(model.predict(X))
	prediction_df = pd.DataFrame()
	pred_array = np.array(predictions)
	prediction_df["mean"] = pred_array.mean(axis=0).reshape(-1,)
	prediction_df["std"] = pred_array.std(axis=0).reshape(-1,)
	return prediction_df

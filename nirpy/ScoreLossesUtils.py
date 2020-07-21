import numpy as np
from sklearn import metrics
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, cross_val_predict, cross_val_score,
                                     cross_validate)

#############################
######## NIR Metrics ########
#############################


def standard_error_cross_validation(X, y, model, n_splits=10):
    "calculates  standard error of cross_validation by performing cross validation with a given number of train/test splits"
    residues = []

    # create indices for cross-validation
    cv = KFold(n_splits=n_splits)
    for train_index, test_index in cv.split(X):
        X_train_kfold, X_test_kfold, y_train_kfold, y_test_kfold = (
            X[train_index],
            X[test_index],
            y[train_index],
            y[test_index],
        )
        # train model on splits
        model.fit(X_train_kfold, y_train_kfold)

        # secv
        y_pred_kfold = model.predict(X_train_kfold)
        # residues of true vs cv_predicted value
        residues.append(np.sum((y_train_kfold - y_pred_kfold) ** 2))

    # standard error for each split
    secv = []
    for residue in residues:
        secv.append(np.sqrt(residue / (y_pred_kfold.shape[0] - 1)))

    secv_mean = np.mean(secv)
    secv_std = np.std(secv)
    return secv_mean, secv_std


# classical NIR PARA
def standard_error_prediction(X, y, model):
    """calculated standard error of prediction"""
    n_samples = X.shape[0]

    y_pred = model.predict(X)

    res = np.sum((y_pred - y) ** 2)
    sep = np.sqrt(res / (n_samples - 1))
    return sep


def relative_prediction_deviation(X, y, y_unscaled, model):
    """relative prediction deviation - using unscaled y"""
    std_y = np.std(y_unscaled)
    rpd = std_y / standard_error_calibration(X, y, model)
    return rpd


def standard_error_calibration(X, y, model):
    """ SEC will decrease with the number of wavelengths (independent variable
	NIRS Calibration Basics 145
	terms) used within an equation, indicating that increasing the number of terms will allow more variation within the data to be explained, or “fitted.”
	The SEC statistic is a useful estimate of the theoretical “best” accuracy obtainable for a specified set of wavelengths used to develop a calibration equation. T
	"""
    n_samples = X.shape[0]
    n_coef = model.get_params()["n_components"]

    y_pred = model.predict(X)

    res = np.sum((y_pred - y) ** 2)
    sec = np.sqrt(res / abs(n_samples - n_coef - 1))
    return sec

def root_mean_squared_error(y_true, y_pred):
	"""calculates root mean squared error"""
	return np.sqrt(mean_squared_error(y_true, y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
	'''calculates huber loss, a robust estimator for regression'''
	y_true = y_true.reshape(-1,1)
	y_pred = y_pred.reshape(-1,1)
	return np.mean(delta**2*( (1+((y_true-y_pred)/delta)**2)**0.5 -1))

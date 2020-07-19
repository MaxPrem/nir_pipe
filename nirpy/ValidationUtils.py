import matplotlib.pyplot as plt
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

    secv_mean = np.mean(secv).round(3)
    return secv_mean


# classical NIR PARA
def standard_error_prediction(X, y, model):
    """calculated standard error of prediction"""
    n_samples = X.shape[0]

    y_pred = model.predict(X)

    res = np.sum((y_pred - y) ** 2)
    sep = np.sqrt(res / (n_samples - 1))
    return sep


def relative_prediction_deviation(X_train, y_train, y, model):
    """relative prediction deviation - using unscaled y"""
    std = np.std(y)
    rpd = std / standard_error_calibration(X_train, y_train, model)
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


##############################
#### Benchmark#####
##############################


def benchmark(X_train, y_train, X_test, y_test, model):
    rmse = (
            np.mean((y_train - model.predict(X_train).reshape(y_train.shape)) ** 2) ** 0.5
    )
    rmse_test = (
            np.mean((y_test - model.predict(X_test).reshape(y_test.shape)) ** 2) ** 0.5
    )
    hub = huber(y_train, model.predict(X_train))
    hub_test = huber(y_test, model.predict(X_test))
    print("RMSE  Train/Test\t%0.2F\t%0.2F" % (rmse, rmse_test))
    print("Huber Train/Test\t%0.4F\t%0.4F" % (hub, hub_test))


##############################
#### CV Metric Functions #####
##############################
def root_mean_squared_error(y_true, y_pred):
    """calculates root mean squared error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Functions used to performe crossfalidation of metrics and scores
# list of crossvalidated scores
def create_score_list():
    score_list = []
    # add metrics to list
    score_list.append(("Variance expl.", make_scorer(metrics.explained_variance_score)))
    score_list.append(("R2", "r2"))
    score_list.append(("Huber Loss", make_scorer(huber)))
    score_list.append(("MSE", make_scorer(mean_squared_error)))
    score_list.append(("RMSE", make_scorer(root_mean_squared_error)))
    return score_list


# cross validation on trained model
def cv_scores(X, y, model, score_list, cv=5):
    "performs crossvalidation on list of metrics and scores"
    results = []
    names = []
    results_mean = []
    results_std = []
    for score_name, score in score_list:
        cv_results = cross_validate(model, X, y, cv=cv, scoring=score)
        # results_model.append(cv_results)
        names.append(score_name)

        # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        # print(score_name, cv_results['test_score'].mean().round(3),'±',cv_results['test_score'].std().round(3))

        results.append(
            (
                cv_results["test_score"].mean().round(3),
                "±",
                cv_results["test_score"].std().round(3),
            )
        )

        results_mean.append((cv_results["test_score"].mean().round(3)))
        results_std.append((cv_results["test_score"].std().round(3)))

    return results, names, results_mean, results_std
# output function

def cross_table(X_train, y_train, X_test, y_test, model):
    """unpacks dict from crossvalpredict with metrics and prits output table"""
    score_list = create_score_list()
    train_results, score_name, train_mean, train_std = cv_scores(
        X_train, y_train, model, score_list
    )

    test_results, score_name, test_mean, test_std = cv_scores(
        X_test, y_test, model, score_list
    )

    for name, item_a_m, item_a_s, item_b_m, item_b_s in zip(
            score_name, train_mean, train_std, test_mean, test_std
    ):
        print(
            "{:<15}{:<4}{:^2}{:<5}{:>10}{:^2}{:>3}".format(
                name, item_a_m, " ± ", item_a_s, item_b_m, " ± ", item_b_s
            )
        )

def cal_scores(X, y, model, score_list):
    "performs crossvalidation on list of metrics and scores"
    results = []
    names = []
    results_mean = []
    for score_name, score in score_list:
        cv_results = cross_validate(model, X, y, cv=1, scoring=score)
        # results_model.append(cv_results)
        names.append(score_name)

        # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        # print(score_name, cv_results['test_score'].mean().round(3),'±',cv_results['test_score'].std().round(3))

        results.append(
            (
                cv_results["test_score"].mean().round(3),
                "±",
            )
        )

        results_mean.append((cv_results["test_score"].mean().round(3)))

    return results, names, results_mean


# output function
def benchmark_table(X_train, y_train, X_test, y_test, model):
    """unpacks dict from crossvalpredict with metrics and prits output table"""
    score_list = create_score_list()
    train_results, score_name, train_mean= cal_scores(
        X_train, y_train, model, score_list
    )

    test_results, score_name, test_mean= cv_scores(
        X_test, y_test, model, score_list
    )

    for name, item_a_m, item_b_m in zip(
            score_name, train_mean, test_mean
    ):
        print(
            "{:<15}{:<4}{:>10}".format(
                name, item_a_m, item_b_m,
            )
        )


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


def benchmark_model(X_train, y_train, X_test, y_test, y, ref, model):
    """Final Output Function for pls regression"""
    # get name of reference method for outputtable

    y_c, y_cv, hub, mse_c, score_c = score_table(X_train, y_train, model)

    y_c_test, y_cv_test, hub_test, mse_c_test, score_test = score_table(
        X_test, y_test, model
    )

    sec_train = standard_error_calibration(X_train, y_train, model)

    sep_test = standard_error_prediction(X_test, y_test, model)

    secv_train = standard_error_cross_validation(X_test, y_test, model)

    rpd = relative_prediction_deviation(X_train, y_train, y, model)
    # pls model components

    n_comp = model.get_params()["n_components"]
    print("Number of Principal Components:", n_comp)

    print("Number of training sampels:", y_train.shape[0])
    print("Number of test sampels:", y_test.shape[0])
    print("SD reference method{}:{:.3} ".format(ref,np.std(y)))

    # nir metrics
    print("SEC \t{:.3}".format(sec_train))
    print("SEP \t{:.3}".format(sep_test))
    print("SECV \t{:.3}".format(secv_train))
    print("RPD \t{:.3}".format(rpd))

    # printing
    print("#### Valdation #### \t{}\t\t{:.3}".format("train", "test"))

    print("MSE calib\t\t{:.3}\t\t{:.3}".format(mse_c, mse_c_test))
    print("RMSE calib\t\t{:.3}\t\t{:.3}".format(np.sqrt(mse_c), np.sqrt(mse_c_test)))
    print("Huber\t\t\t{:.3}\t\t{:.3}".format(hub, hub_test))
    print("R2 calib\t\t{:.3}\t\t{:.3}".format(score_c, score_test))
    print("MSE calib\t\t{:.3}\t\t{:.3}".format(mse_c, mse_c_test))
    print("RMSE calib\t\t{:.3}\t\t{:.3}".format(np.sqrt(mse_c), np.sqrt(mse_c_test)))
    print("conf 95%")
    print("kfold")

    # cross validation loop + unpack function
    #print("#### CrossVal. #### \t{}\t\t{}".format("train", "test"))
    #cross_table(X_train, y_train, X_test, y_test, model)

    z = np.polyfit(y_train, y_cv, 1)
    with plt.style.context(("ggplot")):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_cv, y_train, edgecolors="k")
        ax.scatter(y_cv_test, y_test, edgecolors="k")
        ax.plot(z[1] + z[0] * y_train, y_train)
        ax.plot(y_train, y_train)
        plt.title(
            "$R^{2}$ (CV):"
            + str(cross_val_score(model, X_train, y_train, cv=10).mean().round(2))
        )
        plt.xlabel("Predicted")
        plt.ylabel("Measured")

        plt.show()


def huber(y_true, y_pred, delta=1.0):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    return np.mean(delta ** 2 * ((1 + ((y_true - y_pred) / delta) ** 2) ** 0.5 - 1))


def huber_np(y_true, y_pred, delta=1.0):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    return np.mean(
        delta ** 2 * ((1 + (np.substract(y_true, y_pred) / delta) ** 2) ** 0.5 - 1)
    )


# select same varaiables on X_test
def benchmark(X_train, y_train, X_test, y_test, model):
    rmse = (
            np.mean((y_train - model.predict(X_train).reshape(y_train.shape)) ** 2) ** 0.5
    )
    rmse_test = (
            np.mean((y_test - model.predict(X_test).reshape(y_test.shape)) ** 2) ** 0.5
    )
    hub = huber(y_train, model.predict(X_train))
    hub_test = huber(y_test, model.predict(X_test))
    print("RMSE  Train/Test\t{:.4}\t{:.4}".format((rmse, rmse_test)))
    print("Huber Train/Test\t{:.4}\t{:.4}".format((hub, hub_test)))

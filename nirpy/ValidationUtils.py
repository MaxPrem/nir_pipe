import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     cross_validate)

from ScoreLossesUtils import (huber_loss, relative_prediction_deviation,
                              root_mean_squared_error,
                              standard_error_calibration,
                              standard_error_cross_validation,
                              standard_error_prediction)


##############################
#### CV Metric Functions #####
##############################
# Functions used to performe crossfalidation of metrics and scores
# list of crossvalidated scores
def _create_score_list():
    """List with scores and metrics for _calculate_cv_scores function"""
    score_list = []
    # add metrics to list
    score_list.append(
        ("Variance expl.:", make_scorer(metrics.explained_variance_score))
    )
    score_list.append(("R2:", make_scorer(r2_score)))
    score_list.append(("Huber Loss:", make_scorer(huber_loss)))
    score_list.append(("MSE:", make_scorer(mean_squared_error)))
    score_list.append(("RMSE:", make_scorer(root_mean_squared_error)))
    return score_list


# cross validation on trained model
def _calculate_cv_scores(X, y, model, score_list, cv=10):
    "stores crossvalidation results in a list"
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
                cv_results["test_score"].mean().round(5),
                "±",
                cv_results["test_score"].std().round(5),
            )
        )

        results_mean.append((cv_results["test_score"].mean().round(5)))
        results_std.append((cv_results["test_score"].std().round(5)))

    return results, names, results_mean, results_std


# output function


def print_cv_table(X, y, X_test, y_test, model, **kwargs):
    """unpacks results and names from _calculate_cv_scores"""
    score_list = _create_score_list()
    # cv_predict list of scores and store
    # list definition in _calculate_cv_scores function
    train_results, score_name, train_mean, train_std = _calculate_cv_scores(
        X, y, model, score_list
    )

    test_results, score_name, test_mean, test_std = _calculate_cv_scores(
        X_test, y_test, model, score_list
    )
    # print header for table
    print("Mean score and the 95`%` confidence interval from 10 fold CV")
    print(
        "{:^15}{:<4}{:^2}{:<5}{:>10}{:^2}{:>3}".format(
            "Scores", " ", " train ", " ", " ", "test", " "
        )
    )
    # print table
    for name, mean_1, std_1, mean_2, std_2 in zip(
        score_name, train_mean, train_std, test_mean, test_std
    ):
        print(
            "{:<15}{:<4}{:^2}{:<5}{:>10}{:^2}{:>3}".format(
                name, mean_1, " ± ", std_1 * 2, mean_2, " ± ", std_2 * 2
            )
        )


# score function (not cross validated)


def calculate_regression_benchmarks(X, y, model):
    # y_calibration estimate
    y_c = model.predict(X)
    # cv estimate for plotting
    y_cv = cross_val_predict(model, X, y, cv=10)

    # Calculate scores for calibration and cross-validation
    hub = huber_loss(y, y_c)
    mse_c = mean_squared_error(y, y_c)
    score_c = r2_score(y, y_c)

    return y_c, y_cv, hub, mse_c, score_c


def print_regression_benchmark(X, y, X_test, y_test, model):
    """prints regression score and metrics"""
    y_pred, y_cv, hub, mse_c, score_c = calculate_regression_benchmarks(X, y, model)

    (
        y_c_test,
        y_cv_test,
        hub_test,
        mse_c_test,
        score_test,
    ) = calculate_regression_benchmarks(X_test, y_test, model)
    # printing
    print("{:^10}\t{}\t\t{}".format("Validation", "train", "test"))
    print("R2 calib\t\t{:.3}\t\t{:.3}".format(score_c, score_test))
    print("MSE calib\t\t{:.3}\t\t{:.3}".format(mse_c, mse_c_test))
    print("RMSE calib\t\t{:.3}\t\t{:.3}".format(np.sqrt(mse_c), np.sqrt(mse_c_test)))
    print("Huber\t\t\t{:.3}\t\t{:.3}".format(hub, hub_test))


def print_nir_metrics(X, y, X_test, y_test, model, y_unscaled, ref, **kwargs):
    """calculates and prints relevant nir metrics"""
    sec_train = standard_error_calibration(X, y, model)

    sep_test = standard_error_prediction(X_test, y_test, model)
    secv_mean, secv_std = standard_error_cross_validation(X_test, y_test, model)

    rpd = relative_prediction_deviation(X, y, y_unscaled, model)
    # pls model meta_data
    n_comp = model.get_params()["n_components"]
    print("Summary Reference Method:", ref)
    print("Number of Principal Components:", n_comp)
    print("Number of training sampels:", y.shape[0])
    print("Number of test sampels:", y_test.shape[0])
    # nir metrics
    print("NIR Metrics")
    print("Standard Error of Calibration (SEC): \t{:.3}".format(sec_train))
    print("Standard error of Prediction (SEP): \t{:.3}".format(sep_test))
    print(
        "Standard error of cross-validatoin (SECV): \t{:.3}± {:.3}".format(
            secv_mean, secv_std * 2
        )
    )
    print("RPD \t{:.3}".format(rpd))


def cv_benchmark_model(X, y, X_test, y_test, model, y_unscaled, ref, **kwargs):
    """Final Output Function for pls regression"""
    # get name of reference method for outputtable
    print_nir_metrics(X, y, X_test, y_test, model, y_unscaled, ref)
    print_regression_benchmark(X, y, X_test, y_test, model, y_unscaled, ref)
    # print crosstable
    print_cv_table(X, y, X_test, y_test, model)


def val_regression_plot(X, y, X_test, y_test, model, **kwargs):
    """regression plot showing cross-val estimates for train & test data"""
    y_cv = cross_val_predict(model, X, y)
    y_cv_test = cross_val_predict(model, X_test, y_test)
    z = np.polyfit(y, y_cv, 1)
    with plt.style.context(("ggplot")):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_cv, y, edgecolors="k")
        ax.scatter(y_cv_test, y_test, edgecolors="k")
        ax.plot(z[1] + z[0] * y, y)
        ax.plot(y, y)
        plt.title(
            "$R^{2}$ (CV):" + str(cross_val_score(model, X, y, cv=10).mean().round(2))
        )
        plt.xlabel("Predicted")
        plt.ylabel("Measured")

        plt.show()


# select same varaiables on X_test
def benchmark(X, y, X_test, y_test, model):
    rmse = np.mean((y - model.predict(X).reshape(y.shape)) ** 2) ** 0.5
    rmse_test = (
        np.mean((y_test - model.predict(X_test).reshape(y_test.shape)) ** 2) ** 0.5
    )
    hub = huber_loss(y, model.predict(X))
    hub_test = huber_loss(y_test, model.predict(X_test))
    print("RMSE  Train/Test\t{:.4}\t{:.4}".format(rmse, rmse_test))
    print("Huber Train/Test\t{:.4}\t{:.4}".format(hub, hub_test))

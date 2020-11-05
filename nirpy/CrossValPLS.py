from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (explained_variance_score, make_scorer,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     cross_validate)


def pls_crossval(X, y, n_comp, **kwargs):

    opt_comp = optimal_n_comp(X, y, n_comp)

    opt_model = pls_regression(X, y, opt_comp)

    pls_scores(X, y, opt_model)




def _plot_variance_explained(var_explained, component, *args):
    """Plots MSE for each component, to show component with minimal error"""
    opt_n_comp = np.argmin(var_explained) - 1
    stdout.write("\n")
    # plot var_explained for each component
    with plt.style.context(("ggplot")):
        plt.plot(component, np.array(var_explained), "-v", color="blue", mfc="blue")
        plt.plot(
            component[opt_n_comp], np.array(var_explained)[opt_n_comp], "P", ms=10, mfc="red"
        )
        plt.xlabel("Number of PLS components")
        plt.ylabel("Variance explained %")
        plt.title("PLS")
        plt.xlim(0-1)

    plt.show()

# rewrite variance expalined to always take train test, and set train to test ... if test is none
def extra_plot_variance_explained(var_explained, component,var_explained_2, component_2,  *args):
    """Plots train & test MSE for each component, to show component with minimal error"""
    opt_n_comp = np.argmin(var_explained) - 1
    stdout.write("\n")
    # plot var_explained for each component
    with plt.style.context(("ggplot")):
        ax = plt.subplot(111)
        plt.plot(component, np.array(var_explained), "-v", color="red", mfc="red", label="train")
        plt.plot(
            component[opt_n_comp], np.array(var_explained)[opt_n_comp], "P", ms=10, mfc="red"
        )
        plt.plot(component_2, np.array(var_explained_2), "-v", color="blue", mfc="blue", label="test")
        plt.plot(
            component_2[opt_n_comp], np.array(var_explained_2)[opt_n_comp], "P", ms=10, mfc="red"
        )
        plt.xlabel("Number of PLS components")
        plt.ylabel("Variance explained %")
        plt.title("PLS")
        plt.xlim(0-1)
        ax.legend()
        plt.show()

def variance_explained(X, y, n_comp=15, plot=True):
    """Finds number of components max variance explained"""

    var_explained = []
    component = np.arange(1, n_comp)

    # compute var_explained for each component
    for i in component:
        pls = PLSRegression(n_components=i)

        # Cross-validation

        var_explained.append(
            cross_validate(pls, X, y, scoring=make_scorer(explained_variance_score), cv=10)[
                "test_score"
            ].mean()
        )

    # one component before mse minimum
    opt_n_comp = np.argmax(var_explained)
    stdout.write("\n")
    print(var_explained)

    if plot == True:
        # plot mse for each component
        _plot_variance_explained(var_explained, component)

    return var_explained, component





def mse_minimum(X, y, n_comp=15, plot=True, **kwargs):
    """Finds number of components with a minimal MSE CV on test set regression"""

    mse = []
    component = np.arange(1, n_comp)

    # compute mse for each component
    for i in component:
        pls = PLSRegression(n_components=i)

        # Cross-validation

        mse.append(
            cross_validate(pls, X, y, scoring=make_scorer(mean_squared_error), cv=10)[
                "test_score"
            ].mean()
        )

    #     # generate % value to print update
    #     completed = 100 * (i + 1) / 40
    #     # Trick to updata status on the same line
    #     stdout.write("\r{:.3} completed".format(completed))
    #     stdout.flush()
    # stdout.write("\n")

    # one component before mse minimum
    opt_n_comp = np.argmin(mse) - 1

    stdout.write("\n")

    if plot == True:
        # plot mse for each component
        _plot_mse(mse, component)

    return mse, component


def optimal_n_comp(X, y, n_comp=15, plot=True, **kwargs):
    """Finds number of components with a minimal MSE CV on test set regression"""

    mse = []
    component = np.arange(1, n_comp)

    # compute mse for each component
    for i in component:
        pls = PLSRegression(n_components=i)

        # Cross-validation

        mse.append(
            cross_validate(pls, X, y, scoring=make_scorer(mean_squared_error), cv=10)[
                "test_score"
            ].mean()
        )

    opt_n_comp = np.argmin(mse) - 1

    stdout.write("\n")

    return opt_n_comp


def _plot_mse(mse, component):
    """Plots MSE for each component, to show component with minimal error"""
    opt_n_comp = np.argmin(mse) - 1
    stdout.write("\n")
    # plot mse for each component
    with plt.style.context(("ggplot")):
        plt.plot(component, np.array(mse), "-v", color="blue", mfc="blue")
        plt.plot(
            component[opt_n_comp], np.array(mse)[opt_n_comp], "P", ms=10, mfc="red"
        )
        plt.xlabel("Number of PLS components")
        plt.ylabel("MSE CV")
        plt.title("PLS")
        plt.xlim(left=-1)
    plt.show()


def extra_plot_mse(mse, component, mse_2, component_2):
    """Plots MSE for each component, to show component with minimal error"""
    opt_n_comp = np.argmin(mse) - 1
    stdout.write("\n")
    # plot mse for each component
    with plt.style.context(("ggplot")):
        ax = plt.subplot(111)
        plt.plot(component, np.array(mse), "-v", color="red", mfc="red", label="train")
        plt.plot(
            component[opt_n_comp], np.array(mse)[opt_n_comp], "P", ms=10, mfc="red"
        )
        plt.plot(component_2, np.array(mse_2), "-v", color="blue", mfc="blue", label="test")
        plt.plot(
            component_2[opt_n_comp], np.array(mse_2)[opt_n_comp], "P", ms=10, mfc="red"
        )
        plt.xlabel("Number of PLS components")
        plt.ylabel("MSE CV")
        plt.title("PLS")
        plt.xlim(0-1)
        ax.legend()
    plt.show()


def _plot_regression(y, y_pred):
    """plots predicted vs meassured wtih crossvalidated R2"""
    z = np.polyfit(y, y_pred, 1)
    with plt.style.context(("ggplot")):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_pred, y, c="red", edgecolors="k")
        # Plot the best fit line
        ax.plot(np.polyval(z, y), y, c="blue", linewidth=1)
        # Plot the ideal 1:1 linear
        ax.plot(y, y, color="green", linewidth=1)

        plt.title("R2: {:.3}".format(r2_score(y, y_pred)))
        plt.xlabel("Predicted")
        plt.ylabel("Measured")

        plt.show()


def pls_regression(X, y, n_comp, plot=True, **kwargs):
    """Define PLS object with optimal number of components"""
    pls_opt = PLSRegression(n_components=n_comp)

    # Fit to dataset
    pls_opt.fit(X, y)

    if plot == True:
        _plot_regression(y, pls_opt.predict(X))

    return pls_opt


def pls_scores(X, y, pls_opt, **kwar):
    """prints regression score and metric"""
    y_pred = pls_opt.predict(X)

    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_pred)
    score_cv = cross_val_score(pls_opt, X, y, cv=10)

    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_pred)
    mse_cv = cross_val_score(
        pls_opt, X, y, cv=10, scoring=make_scorer(mean_squared_error)
    )

    print("R2 calib: {:.3}".format(score_c))
    print("R2 CV2: {:.3}".format(score_cv.mean()))
    print("MSE calib: {:.3}".format(mse_c))
    print("MSE CV: {:.3}".format(mse_cv.mean()))

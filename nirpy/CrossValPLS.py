from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate


def pls_crossval(X, y, n_comp):

    opt_comp = optimal_n_comp(X, y, n_comp)

    opt_model = pls_regression(X, y, opt_comp)

    pls_scores(X, y, opt_model)


def optimal_n_comp(X, y, n_comp=20, plot=True):
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

        # generate % value to print update
        completed = 100 * (i + 1) / 40
        # Trick to updata status on the same line
        stdout.write("\r{:.3} completed".format(completed))
        stdout.flush()
    stdout.write("\n")

    # component with lowest mse
    opt_n_comp = np.argmin(mse)
    print("Suggested number of components: ", opt_n_comp)
    stdout.write("\n")

    if plot == True:
        # plot mse for each component
        _plot_mse(mse, component)

    return opt_n_comp


def _plot_mse(mse, component):
    """Plots MSE for each component, to show component with minimal error"""
    opt_n_comp = np.argmin(mse)
    print("Suggested number of components: ", opt_n_comp)
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


def _plot_regression(y, y_pred, score_cv):
    """plots predicted vs meassured wtih crossvalidated R2"""
    z = np.polyfit(y, y_pred, 1)
    with plt.style.context(("ggplot")):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_pred, y, c="red", edgecolors="k")
        # ax.scatter(y_pred_cv, y, c='blue', edgecolors='k')
        # Plot the best fit line
        ax.plot(np.polyval(z, y), y, c="blue", linewidth=1)
        # Plot the ideal 1:1 linear
        ax.plot(y, y, color="green", linewidth=1)

        plt.title("$R^{2}$ (CV): " + str(score_cv.mean().round(3)))
        plt.xlabel("Predicted")
        plt.ylabel("Measured")

        plt.show()


def pls_regression(X, y, n_comp, plot=True):
    """Define PLS object with optimal number of components"""
    pls_opt = PLSRegression(n_components=n_comp)

    # Fit to dataset
    pls_opt.fit(X, y)

    return pls_opt


def pls_scores(X, y, pls_opt):
    """prints regression score and metric"""
    y_pred = pls_opt.predict(X)

    # Cross-VALIDATION estiamte for plotting
    y_pred_cv = cross_val_predict(pls_opt, X, y, cv=10)

    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_pred)
    score_cv = cross_val_score(pls_opt, X, y, cv=10)

    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_pred)
    mse_cv = cross_val_score(
        pls_opt, X, y, cv=10, scoring=make_scorer(mean_squared_error)
    )

    print("R2 calib: {}".format(score_c))
    print("R2 CV2: {}".format(score_cv.mean()))
    print("MSE calib: {}".format(mse_c))
    print("MSE CV: {}".format(mse_cv.mean()))

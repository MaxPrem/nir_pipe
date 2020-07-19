from sys import stdout

import matplotlib.collections as collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import f
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

#from scipy.signal import savgol_filter



class Outlier(PLSRegression):
    """Removes Outlier by performing PLS regression on spectra and the lab reference method. and compareing Q-Residuals and """

    def __init__(self, conf=0.95, n_comps=5):

        self.conf = conf
        self.n_comps = n_comps

        # self.fit = None
        self.X = None
        # self.X1 = None
        # self.X2 = None

        self.T = None
        self.P = None
        self.Err = None
        self.Q = None
        self.Tsq = None
        self.Tsq_conf = None
        self.Q_conf = None

        self._fitted = False
        self._removed = False
        self.X_in = None
        self.y_in = None

    def fit(self, X, y):
        self.X = X

        pls = PLSRegression(n_components=self.n_comps)
        # Fit data
        pls.fit(self.X, y)

        # Get X scores
        self.T = pls.x_scores_
        # Get X loadings
        self.P = pls.x_loadings_
        # Calculate error array
        self.Err = self.X - np.dot(self.T, self.P.T)

        # Calculate Q-residuals (sum over the rows of the error array)
        self.Q = np.sum(self.Err ** 2, axis=1)
        # Calculate Hotelling's T-squared (note that data are normalised by default)
        self.Tsq = np.sum((pls.x_scores_ / np.std(pls.x_scores_, axis=0)) ** 2, axis=1)

        # set the confidence level
        # conf = self.conf
        # Calculate confidence level for T-squared from the ppf of the F distribution
        self.Tsq_conf = (
            f.ppf(q=self.conf, dfn=self.n_comps, dfd=self.X.shape[0])
            * self.n_comps
            * (self.X.shape[0] - 1)
            / (self.X.shape[0] - self.n_comps)
        )
        # Estimate the confidence level for the Q-residuals
        i = np.max(self.Q) + 1
        while 1 - np.sum(self.Q > i) / np.sum(self.Q > 0) > self.conf:
            i -= 1
        self.Q_conf = i

        self._fitted = True

    def plot(self, X, y):
        if self._fitted == True:
            if self._removed == True:
                self.fit(X=self.X_in, y=self.y_in)

            import matplotlib.pyplot as plt

            ax = plt.figure(figsize=(8, 4.5))
            with plt.style.context(("ggplot")):
                plt.plot(self.Tsq, self.Q, "o")
                plt.plot(
                    [self.Tsq_conf, self.Tsq_conf], [plt.axis()[2], plt.axis()[3]], "--"
                )
                plt.plot(
                    [plt.axis()[0], plt.axis()[1]], [self.Q_conf, self.Q_conf], "--"
                )
                plt.xlabel("Hotelling's T-squared")
                plt.ylabel("Q residuals")
                plt.show()
        else:
            print("Call Fit!")

    def transform(self, X, y, max_outliers=10):

        if self._fitted == True:

            # plscomp = n_comp

            rms_dist = np.flip(np.argsort(np.sqrt(self.Q ** 2 + self.Tsq ** 2)), axis=0)

            # Sort calibration spectgra accroding to descending RMS distance

            Xc = self.X[rms_dist, :]
            Yc = y[rms_dist]

            # Discard one outlier at a time up to the value max_outliers
            # and calculate the mse cross-validation of the PLS model
            # max_outliers = 20

            # Define empty mse array
            mse = np.zeros(max_outliers)

            for j in range(max_outliers):
                pls = PLSRegression(n_components=self.n_comps)
                pls.fit(Xc[j:, :], Yc[j:])
                y_cv = cross_val_predict(pls, Xc[j:, :], Yc[j:], cv=16)

                mse[j] = mean_squared_error(Yc[j:], y_cv)
                msemin = np.where(mse == np.min(mse[np.nonzero(mse)]))[0][0]
                print(msemin)
            # Find  the postion of the minimum in the mse (excluding the zeros)

            msemin = np.where(mse == np.min(mse[np.nonzero(mse)]))[0][0]
            msemin
            print("Removed ", msemin, " outlier.")

            self.X_in = Xc[msemin:, :]
            self.y_in = Yc[msemin:]
            self._removed = True

            return self.X_in, self.y_in
        else:
            print("Call .fit() first!")

    def fit_transform(self, X, y, max_outliers=10):
        self.fit(X, y)
        return self.transform(X, y)


class PLSOptimizer(object):
    def __init__(self):

        self.n_comp = None
        self.X = None
        self.y = None
        self.wl = None

        self.opt_Xc = None
        self.opt_ncomp = None
        self.wav = None
        self.sorted_ind = None

    def fit(self, X, y, max_comp=2):

        # self.wl = wl
        self.X = X
        self.y = y
        self.wl = np.arange(0, X.shape[1])

        # max_comp = self.n_comp

        mse = np.zeros((max_comp, self.X.shape[1]))
        # Loop over the nimber of PLS componets
        for i in range(max_comp):
            # Regression with specified number of components,
            # using full spectrum
            pls1 = PLSRegression(n_components=i + 1)
            pls1.fit(self.X, self.y)
            # Indices of sort spectra accroding to ascending absolute
            # value of PLS coefficients
            #  these are the regression coefficients that quantify the strength of the association between each wavelength and the response
            sorted_ind = np.argsort(np.abs(pls1.coef_[:, 0]))
            # Sort spectra accordingly
            Xc = self.X[:, sorted_ind]
            # Discard one wavelength at a time of the sorted spectra,
            # regress, and calculate the MSE cross-vaidation
            for j in range(Xc.shape[1] - (i + 1)):
                pls2 = PLSRegression(n_components=i + 1)
                pls2.fit(Xc[:, j:], self.y)

                y_cv = cross_val_predict(pls2, Xc[:, j:], self.y, cv=5)
                mse[i, j] = mean_squared_error(self.y, y_cv)

            comp = 100 * (i + 1) / (max_comp)
            stdout.write("\r%d%% completed" % comp)
            stdout.flush()
        stdout.write("\n")
        #### was intendet and not exceuted...

        # Calculate and print the position of minimum in MSE
        mseminx, mseminy = np.where(mse == np.min(mse[np.nonzero(mse)]))

        # print(f'mseminx: {0}, mseminy: {1}'.format(mseminx, mseminy))

        print("Optimised number of PLS components: ", mseminx[0] + 1)
        print("Wavelengths to be discarded", mseminy[0])
        print("Optimised MSEP", mse[mseminx, mseminy][0])
        stdout.write("\n")
        # plt.imshow(mse, interpolation=None)
        # plt.show()
        # Calculate PLS with optimal components and export values
        pls = PLSRegression(n_components=mseminx[0] + 1)
        pls.fit(self.X, self.y)

        sorted_ind = np.argsort(np.abs(pls.coef_[:, 0]))

        Xc = self.X[:, sorted_ind]

        opt_Xc = Xc[:, mseminy[0] :]

        self.opt_Xc = opt_Xc
        self.opt_ncomp = mseminx[0] + 1
        self.wav = mseminy[0]
        self.sorted_ind = sorted_ind

        # return(Xc[:,mseminy[0]:],mseminx[0]+1,mseminy[0], sorted_ind)

    def transform(self, X, y=None):
        """transfrom test data"""
        # sort
        Xc = X[:, self.sorted_ind]
        opt_Xc = Xc[:, self.wav :]
        # remove

        return opt_Xc, y

    def plot(self):
        #
        # if wl == None:
        #     wl = np.arange(0,X.shape[1])
        # self.wl = wl
        # import matplotlib.collections as collections

        # Plot spectra with superimpose selected bands
        ix = np.in1d(self.wl.ravel(), self.wl[self.sorted_ind][: self.wav])
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(8, 9))
            with plt.style.context(("ggplot")):
                ax.plot(self.wl, self.X.T)
                plt.ylabel("First derivative absorbance spectra")
                plt.xlabel("Wavelength (nm)")

                collection = collections.BrokenBarHCollection.span_where(
                    self.wl,
                    ymin=-1,
                    ymax=1,
                    where=ix == True,
                    facecolor="red",
                    alpha=0.3,
                )
                ax.add_collection(collection)

                plt.show()

    def get_params(self):

        return self.opt_Xc, self.opt_ncomp, self.wav, self.sorted_ind

    def predict(self, X_test=None, y_test=None, n_comp=None):
        if n_comp == None:
            n_comp = self.opt_ncomp

        # Run PLS with suggested number of components
        pls = PLSRegression(n_components=n_comp)
        model = pls.fit(self.opt_Xc, self.y)
        y_c = pls.predict(self.opt_Xc)
        # Cross-validation
        y_cv = cross_val_predict(pls, self.opt_Xc, self.y, cv=10)
        # Calculate scores for calibration and cross-validation
        scores_c = r2_score(self.y, y_c)
        scores_cv = r2_score(self.y, y_cv)
        # Calculate mean square error for calibration and cross VALIDATION
        mse_c = mean_squared_error(self.y, y_c)
        mse_cv = mean_squared_error(self.y, y_cv)
        print("Number of Principal Components:", n_comp)
        print("R2 calib: %5.3f" % scores_c)
        print("R2 CV: %5.3f" % scores_cv)
        print("MSE calib: %5.3f" % mse_c)
        print("MSE CV: %5.3f" % mse_cv)

        # Plot Regression
        z = np.polyfit(self.y, y_cv, 1)
        with plt.style.context(("ggplot")):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(y_cv, self.y, c="red", edgecolors="k")
            ax.plot(z[1] + z[0] * self.y, self.y, c="blue", linewidth=1)
            ax.plot(self.y, self.y, color="green", linewidth=1)
            plt.title("$R^{2}$ (CV):" + str(scores_cv))
            plt.xlabel("Predicted $^{\circ}$")
            plt.ylabel("Measured $^{\circ}$")

            # if X_test is not None and y_test is not None:
            #     ax.scatter(y_test,model.predict(X_test), c='blue', edgecolors='k')

            plt.show()

            return model

def pls_cv(X, y, n_comp=2):

    """ This function works by running PLS regression with a given number of components and returns plots of crossvalidatoin and PLScomponents minimum. It can optionally be run with test data for crossvalidatoin. """

    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=n_comp)
    model = pls.fit(X, y)
    y_c = pls.predict(X)
    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)
    # Calculate scores for calibration and cross-validation
    scores_c = r2_score(y, y_c)
    scores_cv = r2_score(y, y_cv)
    # Calculate mean square error for calibration and cross VALIDATION
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    print("Number of Principal Components:", n_comp)
    print("R2 calib: %5.3f" % scores_c)
    print("R2 CV: %5.3f" % scores_cv)
    print("MSE calib: %5.3f" % mse_c)
    print("MSE CV: %5.3f" % mse_cv)

    # Plot Regression
    z = np.polyfit(y, y_cv, 1)
    with plt.style.context(("ggplot")):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_cv, y, c="red", edgecolors="k")
        ax.plot(z[1] + z[0] * y, y, c="blue", linewidth=1)
        ax.plot(y, y, color="green", linewidth=1)
        plt.title("$R^{2}$ (CV):" + str(scores_cv))
        plt.xlabel("Predicted $^{\circ}$")
        plt.ylabel("Measured $^{\circ}$")

        # if X_test is not None and y_test is not None:
        #     ax.scatter(y_test,model.predict(X_test), c='blue', edgecolors='k')
        plt.show()

    return model


def pls_opt_cv(X, y, n_comp=10, plot_components=True):
    """Run PLS regression up to the number of components given, and finde the minimum mean squared error."""

    mse = []

    component = np.arange(1, n_comp)

    for i in component:
        pls = PLSRegression(n_components=i)

        # Cross-validation
        y_pred_cv = cross_val_predict(pls, X, y, cv=10)

        mse.append(
            cross_validate(pls, X, y, scoring=make_scorer(mean_squared_error), cv=10)[
                "test_score"
            ].mean()
        )

        comp = 100 * (i + 1) / 40
        # Trick to updata status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the postion of of minimum in mse 'mean squared error'
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin + 1)
    stdout.write("\n")

    if plot_components is True:
        with plt.style.context(("ggplot")):
            plt.plot(component, np.array(mse), "-v", color="blue", mfc="blue")
            plt.plot(component[msemin], np.array(mse)[msemin], "P", ms=10, mfc="red")
            plt.xlabel("Number of PLS components")
            plt.ylabel("MSE CV")
            plt.title("PLS")
            plt.xlim(left=-1)
        plt.show()

    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=msemin + 1)

    # Fit to the entire dataset
    model = pls_opt.fit(X, y)
    y_predicted = pls_opt.predict(X)

    # Cross-VALIDATION
    y_pred_cv = cross_val_predict(pls_opt, X, y, cv=10)

    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_predicted)
    score_cv = cross_val_score(model, X, y, cv=10)

    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_predicted)
    mse_cv = cross_val_score(
        model, X, y, cv=10, scoring=make_scorer(mean_squared_error)
    )

    print("R2 calib: %5.3f" % score_c)
    print("R2 CV2: %5.3f" % score_cv.mean())
    print("MSE calib: %5.3f" % mse_c)
    print("MSE CV: %5.3f" % mse_cv.mean())

    # Plot regression and figures of merit
    rangey = max(y) - min(y)
    rangex = max(y_predicted) - min(y_predicted)

    # Fit a line to the CV vs Response

    z = np.polyfit(y, y_predicted, 1)
    with plt.style.context(('ggplot')):
    	 fig, ax = plt.subplots(figsize=(9, 5))
    	 ax.scatter(y_predicted, y, c='red', edgecolors='k')
    	 #ax.scatter(y_pred_cv, y, c='blue', edgecolors='k')
    	 #Plot the best fit line
    	 ax.plot(np.polyval(z,y), y, c='blue', linewidth=1)
    	 #Plot the ideal 1:1 linear
    	 ax.plot(y,y, color='green', linewidth=1)

    	 plt.title('$R^{2}$ (CV): '+str(score_cv.mean().round(3)))
    	 plt.xlabel('Predicted')
    	 plt.ylabel('Measured')

    	 plt.show()

    return model


if __name__ == "__main__":

    import os

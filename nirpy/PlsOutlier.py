from sys import stdout
from scipy.stats import f
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (explained_variance_score, make_scorer,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (KFold, cross_val_predict, cross_val_score,
                                     cross_validate)

class Outlier(object):
    """Removes Outlier by performing PLS regression on spectra and the lab reference method. and compareing Q-Residuals and """

    def __init__(self, conf=0.95):

        self.conf = conf
        self.n_comp = None

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

    def fit(self, X, y, n_comp=5):
        self.X = X
        self.n_comp = n_comp

        pls = PLSRegression(n_components=self.n_comp)
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
            f.ppf(q=self.conf, dfn=self.n_comp, dfd=self.X.shape[0])
            * self.n_comp
            * (self.X.shape[0] - 1)
            / (self.X.shape[0] - self.n_comp)
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
                pls = PLSRegression(n_components=self.n_comp)
                pls.fit(Xc[j:, :], Yc[j:])
                #y_cv = cross_val_predict(pls, Xc[j:, :], Yc[j:], cv=10)
                #y_cv = cv_predict(pls,Xc[j:, :], Yc[j:], cv=10)
                mse[j] = cross_validate(pls,Xc[j:, :], Yc[j:], cv=10, scoring=make_scorer(mean_squared_error))['test_score'].mean()
                #mse[j] = mean_squared_error(Yc[j:], y_cv)
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

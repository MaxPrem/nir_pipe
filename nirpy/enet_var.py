from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin



class Enet_Select(TransformerMixin, RegressorMixin, BaseEstimator):
    """ .fit() performs crosscvalidation on ElasticNet model
     .transform() performs variable selection from sparse model
    """

    def __init__(self, max_iter=10000000, cv=5):
        # params opt enet
        self.l1_ratio = None
        self.alpha = None
        self.max_iter = max_iter
        self.cv = cv

        self.calibration = None
        # from transform
        self.feature_importance = None
        self.sel_feats = None
        self.n_selected_features = None

        # selected wl
        self.sorted_ind = None
        self.n_rem = None

        # opt vars
        self.X_sel = None
        # self.y_sel = None
        # self.wl_sel = None

    def fit(self, X, y):

        cv_model = ElasticNetCV(
            l1_ratio=[0.01, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 1],
            eps=0.001,
            n_alphas=100,
            fit_intercept=True,
            normalize=True,
            precompute="auto",
            max_iter=self.max_iter,
            tol=0.0001,
            cv=self.cv,
            copy_X=True,
            verbose=0,
            n_jobs=-1,
            positive=False,
            random_state=None,
            selection="cyclic",
        )

        cv_model.fit(X, y)

        print("Optimal l1_ratio: %.3f" % cv_model.l1_ratio_)
        print("Number of iterations %d" % cv_model.n_iter_)

        self.l1_ratio = cv_model.l1_ratio_
        self.alpha = cv_model.alpha_
        self.max_iter = cv_model.n_iter_

        #####

        model = ElasticNet(
            l1_ratio=self.l1_ratio,
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_intercept=True,
            normalize=True,
        )
        model.fit(X, y)
        # save model
        self.calibration = model

        print("R2:", r2_score(y, model.predict(X)))
        print("R2CV:", cross_val_predict(model, X, y, cv=10).mean())

        self.feature_importance = pd.Series(data=np.abs(model.coef_))

        self.n_selected_features = (self.feature_importance > 0).sum()
        print(
            "{0:d} features selected, reduction of {1:2.2f}%".format(
                self.n_selected_features,
                (1 - self.n_selected_features / len(self.feature_importance)) * 100,
            )
        )

        self.feature_importance.sort_values().tail(600).plot(
            kind="bar", figsize=(18, 6)
        )

        self.sel_feats = (
            self.feature_importance.sort_values()
            .tail(self.n_selected_features)
            .index.astype(str)
        )

        # replace with transformfeats
        # self.X_sel, self.y_sel, self.wl_sel = sel_wavs(specs, lab, self.sel_feats)

        # return self.X_sel, self.y_sel, self.wl_sel

        # feature_importance.sort_values()
        self.sorted_ind = self.feature_importance.argsort()
        # number of removed wavs
        self.n_rem = X.shape[1] - self.n_selected_features

        return self

    def transform(self, X):

        Xc = X[:, self.sorted_ind]

        self.X_sel = Xc[:, self.n_rem :]

        return self.X_sel

    def plot(self, wl, X):

        ix = np.in1d(wl.ravel(), wl[self.sorted_ind][: self.n_rem])

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(8, 9))
            with plt.style.context(("ggplot")):
                ax.plot(wl, X.T)
                plt.ylabel("First derivative absorbance spectra")
                plt.xlabel("Wavelength (nm)")

                collection = collections.BrokenBarHCollection.span_where(
                    wl,
                    ymin=X.min(),
                    ymax=X.max(),
                    where=ix == True,
                    facecolor="red",
                    alpha=0.3,
                )
                ax.add_collection(collection)

                plt.show()

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_model(self, X, y):
        model = ElasticNet(
            l1_ratio=self.l1_ratio,
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_intercept=True,
            normalize=True,
        )
        model.fit(X, y)
        return model

    def get_cal(self):
        return self.calibration

    def predict(self, X, y):

        # part form pls_cv

        # enet = ElasticNet(l1_ratio=self.l1_ratio, alpha = self.alpha, max_iter= self.max_iter, fit_intercept=True, normalize = True)
        # model = enet.fit(X,y)

        enet = self.get_model(X, y)

        y_c = enet.predict(X)
        # Cross-validation
        y_cv = cross_val_predict(enet, X, y, cv=10)
        # Calculate scores for calibration and cross-validation
        scores_c = r2_score(y, y_c)
        scores_cv = r2_score(y, y_cv)
        # Calculate mean square error for calibration and cross VALIDATION
        mse_c = mean_squared_error(y, y_c)
        mse_cv = mean_squared_error(y, y_cv)
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
            plt.xlabel("Predicted $^{\circ}$Brix")
            plt.ylabel("Measured $^{\circ}$Brix")

            # if X_test is not None and y_test is not None:
            #     ax.scatter(y_test,model.predict(X_test), c='blue', edgecolors='k')

            plt.show()

    def get_feats(self):
        return self.sel_feats, self.feature_importance, self.n_selected_features

    def get_vars(self):
        return self.X_sel, self.y_sel, self.wl_sel


def enet_cv(X, y, X_test=None, y_test=None):

    """ This function works by running PLS regression with a given number of components and returns plots of crossvalidatoin and PLScomponents minimum. """
    cv_model = ElasticNetCV(
        l1_ratio=[0.01, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 1],
        eps=0.001,
        n_alphas=100,
        fit_intercept=True,
        normalize=True,
        precompute="auto",
        max_iter=20000,
        tol=0.0001,
        cv=5,
        copy_X=True,
        verbose=0,
        n_jobs=-1,
        positive=False,
        random_state=None,
        selection="cyclic",
    )

    cv_model.fit(X, y)

    # part form pls_cv
    enet = ElasticNet(
        l1_ratio=cv_model.l1_ratio_,
        alpha=cv_model.alpha_,
        max_iter=cv_model.n_iter_,
        fit_intercept=True,
        normalize=True,
    )
    model = enet.fit(X, y)
    y_c = enet.predict(X)
    # Cross-validation
    y_cv = cross_val_predict(enet, X, y, cv=10)
    # Calculate scores for calibration and cross-validation
    scores_c = r2_score(y, y_c)
    scores_cv = r2_score(y, y_cv)
    # Calculate mean square error for calibration and cross VALIDATION
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
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
        plt.xlabel("Predicted $^{\circ}$Brix")
        plt.ylabel("Measured $^{\circ}$Brix")

        # if X_test is not None and y_test is not None:
        #     ax.scatter(y_test,model.predict(X_test), c='blue', edgecolors='k')

        plt.show()

    return model


if __name__ == "__main__":

    from sklearn.pipeline import Pipeline
    from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
    from sklearn.model_selection import train_test_split



    specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")
    lab = pd.read_excel("./luzrawSpectra/labdata.xlsx")

    from ImportModule import importLuzCol, cut_specs

    specs = cut_specs(specs, 4100, 5500)

    X, y, wl = importLuzCol(specs, lab, 4)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scale y
    yscaler = GlobalStandardScaler()
    """"""
    # y = yscaler.fit_transform(y)

    y_train = yscaler.fit_transform(y_train)
    y_test = yscaler.transform(y_test)

    pipe_sel = Pipeline(
        [
            ("scaleing_X", GlobalStandardScaler()),
            ("scatter_correction", EmscScaler()),
            ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
            ("variable_sel", Enet_Select(max_iter=100000)),
        ]
    )

    pipeline = Pipeline(
        [
            ("scaleing_X", GlobalStandardScaler()),
            ("scatter_correction", EmscScaler()),
            ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
        ]
    )
    # _ = plt.plot(wl,X.T)

    X_train_sel = pipe_sel.fit_transform(X_train, y_train)

import matplotlib.collections as collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import (ElasticNet, ElasticNetCV)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict


class EnetSelect(TransformerMixin, RegressorMixin):
    """ .fit() performs crosscvalidation on ElasticNet model
     .transform() performs variable selection from sparse model
    """

    def __init__(self, max_iter=10000000, cv=5):
        # params opt enet
        self.l1_ratio = None
        self.alpha = None
        self.max_iter = max_iter
        self.cv = cv

        # from transform
        self.feature_importance = None
        self.sel_feats = None
        self.n_selected_features = None
        self.wave_number_selected = None

        # selected wave_number
        self.sorted_ind = None
        self.n_rem = None
        self.wave_number = None

        # opt vars
        self.X_sel = None
        # self.y_sel = None
        self.wave_number_sel = None

    def fit(self, X, y):
        """cv-optimizes the l1 ratio and trains a model with found parameters to perform variable selection """
        # find optimal l1 ratio prior to variable selection
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

        print("Optimal l1_ratio: {}".format(cv_model.l1_ratio_))
        print("Number of iterations {}".format(cv_model.n_iter_))
        # save optimzed parameters
        self.l1_ratio = cv_model.l1_ratio_
        self.alpha = cv_model.alpha_
        self.max_iter = cv_model.n_iter_

        # train model on optimzed parameters
        model = ElasticNet(
            l1_ratio=self.l1_ratio,
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_intercept=True,
            normalize=True,
        )
        model.fit(X, y)

        print("R2:", r2_score(y, model.predict(X)))

        self.feature_importance = pd.Series(data=np.abs(model.coef_))

        self.n_selected_features = (self.feature_importance > 0).sum()


        # number of features selected print messaage
        print(
            "{0:d} features selected, reduction of {1:2.2f}%".format(
                self.n_selected_features,
                (1 - self.n_selected_features / len(self.feature_importance)) * 100,
            )
        )


        # define selected features
        self.sel_feats = (
            self.feature_importance.sort_values()
            .tail(self.n_selected_features)
            .index.astype(str)
        )
        # sorted features
        self.sorted_ind = self.feature_importance.argsort()
        # number of removed wavs
        self.n_rem = X.shape[1] - self.n_selected_features

        return self

    def transform(self, X):
        """returns spectra with selected wavenumbers only"""
        Xc = X[:, self.sorted_ind]

        self.X_sel = Xc[:, self.n_rem :]

        return self.X_sel
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

    def predict(self, X, y):
        model = self.get_model(X, y)
        self.plot_enet_regression(X, y, model)

    def get_feats(self):
        return self.sel_feats, self.feature_importance, self.n_selected_features

    def get_vars(self):
        return self.X_sel, self.y_sel, self.wave_number_sel

    def plot(self, wave_number, X):
        """shows selected variables as white bands on a given spectra"""
        # position selected variables
        self.wave_number = wave_number
        self.wave_number_selected = np.in1d(
            self.wave_number.ravel(), self.wave_number[self.sorted_ind][: self.n_rem]
        )

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(8, 6))
            with plt.style.context(("ggplot")):
                plt.plot(self.wave_number, X.T)
                ax.set_xlim(self.wave_number.max(), self.wave_number.min())  # decreasing
                ax.set_xlabel("Wavenumber (cm-1)")
                ax.set_ylabel("Absorbance spectra")
                ax.grid(True)

                collection = collections.BrokenBarHCollection.span_where(
                    self.wave_number,
                    ymin=X.min(),
                    ymax=X.max(),
                    where=self.wave_number_selected == True,
                    facecolor="red",
                    alpha=0.3,
                )
                ax.add_collection(collection)

                plt.show()
        return self


    def plot_feature_importance(self, wave_number, n_features=50):
        sel_feats, feature_importance, n_selected_features = self.get_feats()


        """bar plot with sorted wavenumbers up to highest correlation with response variable"""
        print(
            "{0:d} features selected, reduction of {1:2.2f}%".format(
                n_selected_features,
                (1 - n_selected_features / len(feature_importance)) * 100,
            )
        )
        #ind_arr = np.array(sel_feats, dtype=int)

        feature_df = pd.Series(data = feature_importance.values, index=wave_number)
        feature_df.sort_values().tail(30).plot(kind="bar")


    def plot_enet_regression(self, X, y):
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
            fig, ax = plt.subplots()
            ax.scatter(y_cv, y, c="red", edgecolors="k")
            ax.plot(z[1] + z[0] * y, y, c="blue", linewidth=1)
            ax.plot(y, y, color="green", linewidth=1)
            plt.title("$R^{2}$ (CV):" + str(scores_cv))
            plt.xlabel("Predicted $^{\circ}$Brix")
            plt.ylabel("Measured $^{\circ}$Brix")

            # if X_test is not None and y_test is not None:
            #     ax.scatter(y_test,model.predict(X_test), c='blue', edgecolors='k')

            plt.show()



# if __name__ == "__main__":
#
#     from sklearn.pipeline import Pipeline
#     from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
#     from sklearn.model_selection import train_test_split
#
#
#
#     specs = pd.read_csv("/Users/maxprem/nirPy/calData_full.csv")
#     lab = pd.read_excel("./luzrawSpectra/labdata.xlsx")
#
#     from ImportModule import importLuzCol, cut_specs
#
#     specs = cut_specs(specs, 4100, 5500)
#
#     X, y, wave_number = importLuzCol(specs, lab, 4)
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#
#     # scale y
#     yscaler = GlobalStandardScaler()
#     """"""
#     # y = yscaler.fit_transform(y)
#
#     y_train = yscaler.fit_transform(y_train)
#     y_test = yscaler.transform(y_test)
#
#     pipe_sel = Pipeline(
#         [
#             ("scaleing_X", GlobalStandardScaler()),
#             ("scatter_correction", EmscScaler()),
#             ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
#             ("variable_sel", Enet_Select(max_iter=100000)),
#         ]
#     )
#
#     pipeline = Pipeline(
#         [
#             ("scaleing_X", GlobalStandardScaler()),
#             ("scatter_correction", EmscScaler()),
#             ("smmothing", SavgolFilter(polyorder=2, deriv=0)),
#         ]
#     )
#     # _ = plt.plot(wave_number,X.T)
#
#     X_train_sel = pipe_sel.fit_transform(X_train, y_train)

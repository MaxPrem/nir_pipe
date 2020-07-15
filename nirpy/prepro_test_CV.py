from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

# from sys import stdout
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



#import

specs = pd.read_csv('./luzrawSpectra/nirMatrix.csv')
lab = pd.read_excel('./luzrawSpectra/labdata.xlsx')


from import_Module import importLuzCol

X, y, wl = importLuzCol(specs, lab, 4)

X.shape

y.shape

# splitting dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.shape
y_train.shape
X_test.shape
y_test.shape

# scaling and transforming
from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
from sklearn.pipeline import Pipeline

# scale y
yscaler = GlobalStandardScaler()
y_scaled = yscaler.fit_transform(y)

# Extensive Multiplicative Scattercorrection

pipeline = Pipeline([
    ("scaleing_X", GlobalStandardScaler()),
    ("scatter_correction", EmscScaler()),
    ("smmothing", SavgolFilter())
])

X = pipeline.fit_transform(X)



with plt.style.context(('ggplot')):
    plt.plot(wl, X_pipe.T)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('D2 Absorbance')
    plt.show()




import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance, MinCovDet

from sklearn.decomposition import PCA

pca = PCA(n_components=4)
model = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
model.get_params()

# fit a MCD robust estimator to data
robust_cov = MinCovDet().fit(X)
# fit a MLE estimator to data
emp_cov = EmpiricalCovariance().fit(X)
print('Estimated covariance matrix:\n'
      'MCD (Robust):\n{}\n'
      'MLE:\n{}'.format(robust_cov.covariance_, emp_cov.covariance_))

# %%
# To better visualize the difference, we plot contours of the
# Mahalanobis distances calculated by both methods. Notice that the robust
# MCD based Mahalanobis distances fit the inlier black points much better,
# whereas the MLE based distances are more influenced by the outlier
# red points.pltgg
with plt.style.context("ggplot"):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot data set
    inlier_plot = ax.scatter(X[:, 0], X[:, 1],
                             color='black', label='inliers')
    # outlier_plot = ax.scatter(X[:, 0][-n_outliers:], X[:, 1][-n_outliers:],
    #                           color='red', label='outliers')
    ax.set_xlim(ax.get_xlim()[0], 10.)
    ax.set_title("Mahalanobis distances of a contaminated data set")

    # Create meshgrid of feature 1 and feature 2 values
    xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
                         np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
    zz = np.c_[xx.ravel(), yy.ravel()]
    # Calculate the MLE based Mahalanobis distances of the meshgrid
    mahal_emp_cov = emp_cov.mahalanobis(zz)
    mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
    emp_cov_contour = plt.contour(xx, yy, np.sqrt(mahal_emp_cov),
                                  cmap=plt.cm.PuBu_r, linestyles='dashed')
    # Calculate the MCD based Mahalanobis distances
    mahal_robust_cov = robust_cov.mahalanobis(zz)
    mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
    robust_contour = ax.contour(xx, yy, np.sqrt(mahal_robust_cov),
                                cmap=plt.cm.YlOrBr_r, linestyles='dotted')

    # Add legend
    ax.legend([emp_cov_contour.collections[1], robust_contour.collections[1],
              inlier_plot, outlier_plot],
              ['MLE dist', 'MCD dist', 'inliers', 'outliers'],
              loc="upper right", borderaxespad=0)

    plt.show()

# %%
# Finally, we highlight the ability of MCD based Mahalanobis distances to
# distinguish outliers. We take the cubic root of the Mahalanobis distances,
# yielding approximately normal distributions (as suggested by Wilson and
# Hilferty [2]_), then plot the values of inlier and outlier samples with
# boxplots. The distribution of outlier samples is more separated from the
# distribution of inlier samples for robust MCD based Mahalanobis distances.
with plt.style.context("ggplot"):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=.6)

    # Calculate cubic root of MLE Mahalanobis distances for samples
    emp_mahal = emp_cov.mahalanobis(X - np.mean(X, 0)) ** (0.33)
    # Plot boxplots
    ax1.boxplot([emp_mahal[:-n_outliers], emp_mahal[-n_outliers:]], widths=.25)
    # Plot individual samples
    ax1.plot(np.full(n_samples - n_outliers, 1.26), emp_mahal[:-n_outliers],
             '+k', markeredgewidth=1)
    ax1.plot(np.full(n_outliers, 2.26), emp_mahal[-n_outliers:],
             '+k', markeredgewidth=1)
    ax1.axes.set_xticklabels(('inliers', 'outliers'), size=15)
    ax1.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
    ax1.set_title("Using non-robust estimates\n(Maximum Likelihood)")

    # Calculate cubic root of MCD Mahalanobis distances for samples
    robust_mahal = robust_cov.mahalanobis(X - robust_cov.location_) ** (0.33)
    # Plot boxplots
    ax2.boxplot([robust_mahal[:-n_outliers], robust_mahal[-n_outliers:]],
                widths=.25)
    # Plot individual samples
    ax2.plot(np.full(n_samples - n_outliers, 1.26), robust_mahal[:-n_outliers],
             '+k', markeredgewidth=1)
    ax2.plot(np.full(n_outliers, 2.26), robust_mahal[-n_outliers:],
             '+k', markeredgewidth=1)
    ax2.axes.set_xticklabels(('inliers', 'outliers'), size=15)
    ax2.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
    ax2.set_title("Using robust estimates\n(Minimum Covariance Determinant)")

    plt.show()



from pls_utils import simple_pls_cv, pls_variable_selection, optimize_pls_cv

optimize_pls_cv(X_train, y_train, 5)


#opt_Xc, ncomp, wav, sorted_ind = pls_variable_selection(X_sg, y, 5)

simple_pls_cv(opt_Xc, y, ncomp)
optimize_pls_cv(opt_Xc, y, ncomp)
scaled_benchmark(X_train, y_train, X_test, y_test, model)

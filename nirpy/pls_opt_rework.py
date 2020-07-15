from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f
import matplotlib.collections as collections





class PLSOptimizer(object):


    def __init__(self):

        self.n_comp = None


        self.opt_Xc = None
        self.opt_ncomp = None
        self.wav = None
        self.sorted_ind = None


    def fit(self, X, y, max_comp = 2):
        '''To minimze the MSE a set of variables is selected by PLS Regression. The outer Loop tests pls regression with up to max_comp and sorts the wavs.  For each calibration the inner loop discards wavelengths starting with the lowest correlation


        This function works by running PLS regression with a given number of components,
        then filtering out one regression coefficient at a time up to the maximum number
        allowed.
        All data are stored in the 2D array mse. At the end of the double loop we search for
        the global minimum of mse excluding the zeros.

        '''

        #max_comp = self.n_comp

        mse = np.zeros((max_comp, X.shape[1]))
        # Loop over the nimber of PLS componets
        for i in range(max_comp):
            #Regression with specified number of components,
            #using full spectrum
            pls1 = PLSRegression(n_components=i+1)
            pls1.fit(X,y)
            # Indices of sort spectra accroding to ascending absolute
            # value of PLS coefficients
            #  these are the regression coefficients that quantify the strength of the association between each wavelength and the response
            sorted_ind = np.argsort(np.abs(pls1.coef_[:,0]))
            # Sort spectra accordingly
            Xc = X[:,sorted_ind]
            # Discard one wavelength at a time of the sorted spectra,
            # regress, and calculate the MSE cross-vaidation
            for j in range(Xc.shape[1]-(i+1)):
                pls2 = PLSRegression(n_components=i+1)
                pls2.fit(Xc[:, j:], y)

                y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv =5)
                mse[i,j] = mean_squared_error(y, y_cv)

            comp = 100*(i+1)/(max_comp)
            stdout.write("\r%d%% completed" % comp)
            stdout.flush()
        stdout.write("\n")
        #### was intendet and not exceuted...

        # Calculate and print the position of minimum in MSE
        mseminx,mseminy = np.where(mse==np.min(mse[np.nonzero(mse)]))

        #print(f'mseminx: {0}, mseminy: {1}'.format(mseminx, mseminy))

        print("Optimised number of PLS components: ", mseminx[0]+1)
        print("Wavelengths to be discarded", mseminy[0])
        print("Optimised MSEP", mse[mseminx, mseminy][0])
        stdout.write("\n")
        # plt.imshow(mse, interpolation=None)
        # plt.show()
        #Calculate PLS with optimal components and export values
        pls = PLSRegression(n_components=mseminx[0]+1)
        pls.fit(X,y)

        sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))


        Xc = X[:,sorted_ind]



        opt_Xc = Xc[:,mseminy[0]:]

        self.opt_Xc = opt_Xc
        self.opt_ncomp = mseminx[0]+1
        self.wav = mseminy[0]
        self.sorted_ind = sorted_ind

        return self

    def transform(self, X):
        '''Discards wavelengths with low correlation to the response variable'''

        Xc = X[:,self.sorted_ind]
        return Xc[:,self.wav:]

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X,y)

    def get_cal(self, X, y, n_comp=None):
        if n_comp == None:
            n_comp = self.opt_ncomp
        X = self.transform(X)
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X,y)
        return pls

    def plot(self, wl , X):


        # Plot spectra with superimpose selected bands
        ix = np.in1d(wl.ravel(), wl[self.sorted_ind][:self.wav])
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(8,9))
            with plt.style.context(('ggplot')):
                ax.plot(wl, X.T)
                plt.ylabel('First derivative absorbance spectra')
                plt.xlabel('Wavelength (nm)')

                collection = collections.BrokenBarHCollection.span_where(
                wl, ymin=-1, ymax=1, where=ix == True, facecolor='red', alpha=0.3)
                ax.add_collection(collection)

                plt.show()

    def get_params(self):

        return self.opt_Xc, self.opt_ncomp, self.wav, self.sorted_ind

    def predict(self, X, y, n_comp=None):
        if n_comp == None:
            n_comp = self.opt_ncomp

        self.opt_Xc = self.transform(X)

        # Run PLS with suggested number of components
        pls = PLSRegression(n_components=n_comp)
        model = pls.fit(self.opt_Xc, y)
        y_c = pls.predict(self.opt_Xc)
        # Cross-validation
        y_cv = cross_val_predict(pls, self.opt_Xc, y, cv=10)
        # Calculate scores for calibration and cross-validation
        scores_c = r2_score(y, y_c)
        scores_cv = r2_score(y, y_cv)
        # Calculate mean square error for calibration and cross VALIDATION
        mse_c = mean_squared_error(y, y_c)
        mse_cv = mean_squared_error(y, y_cv)
        print('Number of Principal Components:', n_comp)
        print('R2 calib: %5.3f' % scores_c)
        print('R2 CV: %5.3f' % scores_cv)
        print('MSE calib: %5.3f' % mse_c)
        print('MSE CV: %5.3f' % mse_cv)

        # Plot Regression
        z = np.polyfit(y, y_cv, 1)
        with plt.style.context(('ggplot')):
            fig, ax = plt.subplots(figsize=(9,5))
            ax.scatter(y_cv, y, c='red', edgecolors='k')
            ax.plot(z[1]+z[0]*y, y, c ='blue', linewidth=1)
            ax.plot(y, y, color='green', linewidth = 1)
            plt.title('$R^{2}$ (CV):' + str(scores_cv))
            plt.xlabel('Predicted $^{\circ}$Brix')
            plt.ylabel('Measured $^{\circ}$Brix')

            # if X_test is not None and y_test is not None:
            #     ax.scatter(y_test,model.predict(X_test), c='blue', edgecolors='k')


            plt.show()

            return model


if __name__ == '__main__':


    from sklearn.pipeline import Pipeline
    from ChemUtils import EmscScaler, GlobalStandardScaler, SavgolFilter
    from sklearn.model_selection import train_test_split

    import pandas as pd
    specs = pd.read_csv('/Users/maxprem/nirPy/calData_full.csv')
    lab = pd.read_excel('./luzrawSpectra/labdata.xlsx')



    from import_Module import importLuzCol, cut_specs

    specs = cut_specs(specs, 4100, 5500)

    X, y, wl = importLuzCol(specs, lab, 4)



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # scale y
    yscaler = GlobalStandardScaler()
    ''''''
    #y = yscaler.fit_transform(y)

    y_train = yscaler.fit_transform(y_train)
    y_test = yscaler.transform(y_test)




    pipe_sel = Pipeline([
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2,deriv=0)),
        ("variable_selection", PLSOptimizer()),
    ])

    pipeline = Pipeline([
        ("scaleing_X", GlobalStandardScaler()),
        ("scatter_correction", EmscScaler()),
        ("smmothing", SavgolFilter(polyorder=2,deriv=0))
    ])
    #_ = plt.plot(wl,X.T)



    X_test_sel = pipe_sel.fit_transform(X_test, y_test)
    X_test_sel.shape

    X_train_pre = pipeline.fit_transform(X_train)
    X_test_pre = pipeline.transform(X_test)

    pls_opt = PLSOptimizer()
    pls_opt.fit(X_train,y_train, 4)
    pls_opt.transform(X_train,y_train)


    from pls_utils import  pls_variable_selection

    pls_variable_selection(X_train, y_train, 4)


    model = pipe_sel['variable_selection'].get_cal()
    model.predict(X_train_pre)
    X_train_sel.shape

    benchmark(X_train, y_train, X_test_pre, y_test, model)

    from ChemUtils import benchmark, huber
    from pls_utils import pls_opt_cv, pls_cv
    from ChemUtils import huber, benchmark

    pls_sel = pls_cv(X_train_sel, y_train, 2)
    pls_pre = pls_cv(X_train_pre, y_train, 2)

    benchmark(X_train_sel, y_train, X_test_sel, y_test, pls_sel)

    benchmark(X_train_pre, y_train, X_test_pre, y_test, pls_pre)

    def huber(y_true, y_pred, delta=1.0):
    	y_true = y_true.reshape(-1,1)
    	y_pred = y_pred.reshape(-1,1)
    	return np.mean(delta**2*( (1+((y_true-y_pred)/delta)**2)**0.5 -1))


    def benchmark(X_train,y_train,X_test, y_test, model):

    	rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape))**2)**0.5
    	rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape))**2)**0.5
    	hub = huber(y_train, model.predict(X_train))
    	hub_test = huber(y_test, model.predict(X_test))
        # # y-calibration
        # y_c = pls.predict(X)
        # # y-Cross-validation
        # y_cv = cross_val_predict(pls, X, y, cv=10)
        # # Calculate scores for calibration and cross-validation
        # scores_c = r2_score(y, y_c)
        # scores_cv = r2_score(y, y_cv)
        # # Calculate mean square error for calibration and cross VALIDATION
        # mse_c = mean_squared_error(y, y_c)
        # mse_cv = mean_squared_error(y, y_cv)

    	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
    	print ("RMSE  Train/Test\t%0.2F\t%0.2F"%(rmse, rmse_test))
    	print ("Huber Train/Test\t%0.4F\t%0.4F"%(hub, hub_test))

    benchmark(X_train, y_train, X_test, y_test, model)

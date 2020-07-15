'''old benchmark function collection. still used in deepnet validation'''
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


def huber(y_true, y_pred, delta=1.0):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    return np.mean(delta ** 2 * ((1 + ((y_true - y_pred) / delta) ** 2) ** 0.5 - 1))


# select same varaiables on X_test
def benchmark(X_train, y_train, X_test, y_test, model):
    rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape)) ** 2) ** 0.5
    rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape)) ** 2) ** 0.5
    hub = huber(y_train, model.predict(X_train))
    hub_test = huber(y_test, model.predict(X_test))
    print("RMSE  Train/Test\t%0.2F\t%0.2F" % (rmse, rmse_test))
    print("Huber Train/Test\t%0.4F\t%0.4F" % (hub, hub_test))


def get_metrics(X_train, y_train, model):
    # get n_components
    params = model.get_params()
    n_comp = params["n_components"]
    y_c = model.predict(X_train)
    # Cross-vali-predicted
    y_cv = cross_val_predict(model, X_train, y_train, cv=10)

    scores_c = r2_score(y_train, y_c)
    scores_cv = r2_score(y_train, y_cv)

    # Calculate mean square error for calibration and cross VALIDATION
    mse_c = mean_squared_error(y_train, y_c)
    mse_cv = mean_squared_error(y_train, y_cv)

    return n_comp, scores_c, scores_cv, mse_c, mse_cv


def cv_benchmark(X_train, y_train, X_test, y_test, model):
    rmse = np.mean((y_train - model.predict(X_train).reshape(y_train.shape)) ** 2) ** 0.5
    rmse_test = np.mean((y_test - model.predict(X_test).reshape(y_test.shape)) ** 2) ** 0.5
    hub = huber(y_train, model.predict(X_train))
    hub_test = huber(y_test, model.predict(X_test))
    print("RMSE  Train/Test\t%0.2F\t%0.2F" % (rmse, rmse_test))
    print("Huber Train/Test\t%0.4F\t%0.4F" % (hub, hub_test))
    get_metrics(X_train, y_train, model)

    n_comp, scores_c, scores_cv, mse_c, mse_cv = get_metrics(X_train, y_train, model)



    #####################
    #  pls cv here#
    #####################
    # get n_comps from fitted moden_comp, score_c, score_cv, mse_c, mse_cv \
    n_comp, scores_c, scores_cv, mse_c, mse_cv = get_metrics(X_train, y_train, model)

    mse_cv = get_metrics(X_train, y_train, model)


# n_comp_test, score_c_test, score_cv_test, mse_c_test, mse_cv_test = get_metrics(X_test, y_test, model)

# #####################
# #  pls Test cv here#
# #####################
# # get n_comps from fitted model
# params = model.get_params()
# n_comp = params["n_components"]
# # y_calibration estimate
# y_c_test = model.predict(X_test)
# # Cross-validation
# y_cv_test = cross_val_predict(model, X_test, y_test, cv=10)
# # Calculate scores for calibration and cross-validation
# scores_c_test = r2_score(y_test, y_c_test)
# scores_cv_test = r2_score(y_test, y_cv_test)
# # Calculate mean square error for calibration and cross VALIDATION
# mse_c_test = mean_squared_error(y_test, y_c_test)
# mse_cv_test = mean_squared_error(y_test, y_cv_test)
#
# rmse_c_test = np.sqrt((mean_squared_error(y_test, y_c_test)))
# rmse_cv_test =np.sqrt((mean_squared_error(y_test, y_cv_test)))


# calc_metrics(X_train, y_train, X_test, y_test, model)
#
# ##################################
# print('##################################')
# ##################################
# print('R2 calib Train/Test\t%0.4F\t%0.4F' % (scores_c, scores_c_test))
# print('R2 CV Train/Test\t%0.4F\t%0.4F' % (scores_cv, scores_cv_test))
# print('MSE calib Train/Test\t%0.4F\t%0.4F' % (mse_c, mse_c_test))
# print('MSE CV Train/Test\t%0.4F\t%0.4F' % (mse_cv, mse_cv_test))
# # ´print('RMSE calib Train/Test\t%0.4F\t%0.4F'%(rmse_c , reprmse_c_test))
# # ´print('RMSE CV Train/Test\t%0.4F\t%0.4F'%(rmse_cv , rmse_cv_test))
#
#
# z = np.polyfit(y_train, y_cv, 1)
# with plt.style.context(('ggplot')):
#     fig, ax = plt.subplots(figsize=(9, 5))
#     ax.scatter(y_cv, y_train, edgecolors='k')
#     ax.scatter(y_cv_test, y_test, edgecolors='k')
#     ax.plot(z[1] + z[0] * y_train, y_train)
#     ax.plot(y_train, y_train)
#     plt.title('$R^{2}$ (CV):' + str(scores_cv))
#     plt.xlabel('Predicted $^{\circ}$Brix')
#     plt.ylabel('Measured $^{\circ}$Brix')
#
#     plt.show()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

MODEL_FEATURES = 'narmax_poly2_PROVA'
FILE_NAME = MODEL_FEATURES + '.pkl'
FILE_NAME_ERROR = MODEL_FEATURES + '_error_' + '.pkl'
TRAIN = True
TRAIN_STOP = 10

data = pd.read_pickle('dataset_error_force_del_norm_1705.pkl')
dataframe = pd.DataFrame(data)
X = dataframe["x"]
Y = dataframe["y"]

train_len = int(0.8 * len(X.values[0]["time"]))
test_len = int(0.2 * len(X.values[0]["time"]))

xdata = np.empty((len(X.values), len(X.values[0]["data"])))
ydata = np.empty((len(Y.values), len(Y.values[0]["data"])))
for i in range(0, len(X.values)):
    xdata[i, :] = X.values[i]["data"]
    ydata[i, :] = Y.values[i]["data"]

X_train, X_tmp, Y_train, Y_tmp = train_test_split(np.transpose(xdata), np.transpose(ydata), test_size=0.2, shuffle=False)
X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, shuffle=False)
# x_id = X_train
# y_id = Y_train
# x_val = X_test
# y_val = Y_test

if TRAIN:
    basis_function = Polynomial(degree=5)

    model = FROLS(
        order_selection=True,
        n_info_values=50,
        extended_least_squares=False,
        ylag=2, xlag=2,
        info_criteria='aic',
        estimator='least_squares',
        basis_function=basis_function,
        model_type='NARMAX'
    )

    narmax = []
    errors = []
    for i in range(0, TRAIN_STOP):
        x_id = X_train[:, i].reshape(-1, 1)
        y_id = Y_train[:, i].reshape(-1, 1)
        x_val = X_val[:, i].reshape(-1, 1)
        y_val = Y_val[:, i].reshape(-1, 1)
        narmax.append(model.fit(X=x_id, y=y_id))
        y_hat_val = model.predict(X=x_val, y=y_val)
        err_i = root_relative_squared_error(y_val, y_hat_val)
        errors.append(err_i)
        if i % 10 == 0:
            print("progress: ", i)

    pickle.dump(narmax, open(FILE_NAME, 'wb'))
    pickle.dump(errors, open(FILE_NAME_ERROR,'wb'))
if not TRAIN:
    narmax = pickle.load(open(FILE_NAME, 'rb'))
    errors = pickle.load(open(FILE_NAME_ERROR, 'rb'))
# r = pd.DataFrame(
#     results(
#         model.final_model, model.theta, model.err,
#         model.n_terms, err_precision=8, dtype='sci'
#         ),
#     columns=['Regressors', 'Parameters', 'ERR'])


chosen_idx = np.argmin(np.array(errors))
chosen_model = narmax[chosen_idx]
print("chosen model is number: ", chosen_idx)

rand_idx = np.random.randint(low=0, high=TRAIN_STOP)
x_test = X_test[:, rand_idx].reshape(-1, 1)
y_test = Y_test[:, rand_idx].reshape(-1, 1)
x_plt = np.transpose(xdata)[:, rand_idx].reshape(-1, 1)
y_plt = np.transpose(ydata)[:, rand_idx].reshape(-1, 1)
y_hat_test = chosen_model.predict(X=x_test, y=y_test)
y_hat =  chosen_model.predict(X=x_plt, y=y_plt)
rrse = root_relative_squared_error(y_test, y_hat_test)

time = np.linspace(start=0, stop=60, num=7500)
plt.figure(1)
plt.plot(time, y_plt)
plt.plot(time, y_hat)
plt.xlabel('time')
plt.ylabel('force')
plt.legend(['truth', 'prediction'])
plt.grid()

# plot_results(y=y_test, yhat = y_hat, n=1000)
ee = compute_residues_autocorrelation(y_test, y_hat_test)
plot_residues_correlation(data=ee, title="Residues Autocorrelation", ylabel="$e^2$")
x1e = compute_cross_correlation(y_test, y_hat_test, x_test[:, 0])
plot_residues_correlation(data=x1e, title="Residues Cross-correlation", ylabel="$x_1e$")


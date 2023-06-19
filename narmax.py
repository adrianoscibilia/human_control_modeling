import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Fourier
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

MODEL_FEATURES = 'narmax_test'
FILE_NAME = MODEL_FEATURES + '.pkl'
TRAIN = False
TRAIN_STOP = 12
EPOCHS = 400
LOG_INTERVAL = 5

data = pd.read_pickle('dataset_error_force_del_raw_1506.pkl')
dataframe = pd.DataFrame(data)
X = dataframe["x"]
Y = dataframe["y"]

train_len = int(0.8 * len(X.values[0]["data"]))
test_len = int(0.2 * len(X.values[0]["data"]))

xdata = np.empty((len(X.values), len(X.values[0]["data"])))
ydata = np.empty((len(Y.values), len(Y.values[0]["data"])))
for i in range(0, len(X.values)):
    xdata[i, :] = X.values[i]["data"]
    ydata[i, :] = Y.values[i]["data"]

X_train, X_tmp, Y_train, Y_tmp = train_test_split(np.transpose(xdata), np.transpose(ydata), test_size=0.2, shuffle=False)
# X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, shuffle=False)
X_test = X_tmp
Y_test = Y_tmp

if TRAIN:
    basis_function = Polynomial(degree=4)

    xlag = [1, 1, 1, 1]
    ylag = [1, 125, 625, 1250]

    model = FROLS(
        order_selection=True,
        n_info_values=30,
        extended_least_squares=False,
        mu=0.001,
        ylag=ylag, xlag=xlag,
        info_criteria='aic',
        estimator='least_squares',
        basis_function=basis_function,
        model_type='NARMAX'
    )

    idx_list = np.arange(0, TRAIN_STOP-1).tolist()

    for i in range(0, TRAIN_STOP):
        # if len(idx_list) == 0:
            # break
        # idx = np.random.choice(idx_list)
        x_id = X_train[:, i].reshape(-1, 1)
        y_id = Y_train[:, i].reshape(-1, 1)
        # x_val = X_val[:, i].reshape(-1, 1)
        # y_val = Y_val[:, i].reshape(-1, 1)
        model = model.fit(X=x_id, y=y_id)

        # y_hat_val = model.predict(X=x_val, y=y_val)
        # err_i = root_relative_squared_error(y_val, y_hat_val)
        # errors.append(err_i)
        # idx_list.remove(idx)
        if i % LOG_INTERVAL == 0:
            print("progress: ", i)

    pickle.dump(model, open(FILE_NAME, 'wb'))
    # pickle.dump(errors, open(FILE_NAME_ERROR,'wb'))
if not TRAIN:
    model = pickle.load(open(FILE_NAME, 'rb'))
    # errors = pickle.load(open(FILE_NAME_ERROR, 'rb'))
r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=8, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])


# chosen_idx = np.argmin(np.array(errors))
chosen_model = model  # [TRAIN_STOP-1]
# print("chosen model is number: ", chosen_idx)

rand_idx = np.random.randint(low=0, high=TRAIN_STOP-1)
print("test idx: ", rand_idx)
x_test = X_test[:, rand_idx].reshape(-1, 1)
y_test = Y_test[:, rand_idx].reshape(-1, 1)
x_plt = np.transpose(xdata)[:, rand_idx].reshape(-1, 1)
y_plt = np.transpose(ydata)[:, rand_idx].reshape(-1, 1)
y_hat_test = chosen_model.predict(X=x_test, y=y_test)
y_hat = chosen_model.predict(X=x_plt, y=y_plt)
rrse = root_relative_squared_error(y_test, y_hat_test)

time = np.linspace(start=0, stop=60, num=7500)
plt.figure(1)
plt.plot(time, y_plt)
plt.plot(time, y_hat)
plt.xlabel('time')
plt.ylabel('force')
plt.legend(['truth', 'prediction'])
plt.grid()

rand_idx = np.random.randint(low=0, high=TRAIN_STOP-1)
print("test idx: ", rand_idx)
x_test = X_test[:, rand_idx].reshape(-1, 1)
y_test = Y_test[:, rand_idx].reshape(-1, 1)
x_plt = np.transpose(xdata)[:, rand_idx].reshape(-1, 1)
y_plt = np.transpose(ydata)[:, rand_idx].reshape(-1, 1)
y_hat_test = chosen_model.predict(X=x_test, y=y_test)
y_hat = chosen_model.predict(X=x_plt, y=y_plt)
rrse = root_relative_squared_error(y_test, y_hat_test)

time = np.linspace(start=0, stop=60, num=7500)
plt.figure(2)
plt.plot(time, y_plt)
plt.plot(time, y_hat)
plt.xlabel('time')
plt.ylabel('force')
plt.legend(['truth', 'prediction'])
plt.grid()

# plot_results(y=y_test, yhat = y_hat, n=1000)
# ee = compute_residues_autocorrelation(y_test, y_hat_test)
# plot_residues_correlation(data=ee, title="Residues Autocorrelation", ylabel="$e^2$")
# x1e = compute_cross_correlation(y_test, y_hat_test, x_test[:, 0])
# plot_residues_correlation(data=x1e, title="Residues Cross-correlation", ylabel="$x_1e$")

plt.show()


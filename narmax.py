import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

data = sio.loadmat('err_force_delsecsint_norm_iddata.mat')
dataframe = pd.DataFrame(data)
X = dataframe["x"]
Y = dataframe["y"]
train_len = int(0.8 * len(X.values[0].time))
test_len = int(0.2 * len(X.values[0].time))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

basis_function = Polynomial(degree=2)

x1lag = list(range(1, 10))
x2lag = list(range(1, 10))

model = FROLS(
    order_selection=True,
    n_info_values=39,
    extended_least_squares=False,
    ylag=20, xlag=[x1lag, x2lag],
    info_criteria='bic',
    estimator='least_squares',
    basis_function=basis_function
)



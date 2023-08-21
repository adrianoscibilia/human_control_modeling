import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils.data as Data
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from scipy import signal


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN = False
PREDICT = False
SUB_SAMPLING = False
MODE = 1

NET_FEATURES = 'PROVA'
FILE_NAME = NET_FEATURES + '.pkl'
fig_dir = '/home/adriano/Pictures/NN_model_Figures/'


class MLP(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, 2 * H)
        self.linear5 = torch.nn.Linear(2 * H, 4 * H)
        self.linear6 = torch.nn.Linear(4 * H, 2 * H)
        self.linear7 = torch.nn.Linear(2 * H, H)
        self.linear8 = torch.nn.Linear(H, H)
        self.linear9 = torch.nn.Linear(H, H)
        self.linear10 = torch.nn.Linear(H, D_out)
        self.to(DEVICE)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        x = self.linear4(F.relu(x))
        x = self.linear5(F.relu(x))
        x = self.linear6(F.relu(x))
        x = self.linear7(F.relu(x))
        x = self.linear8(F.relu(x))
        x = self.linear9(F.relu(x))
        y_pred = self.linear10(torch.tanh(x))
        return y_pred


def peak_to_peak(input_array):
    peaks, _ = signal.find_peaks(input_array, height=0.1, distance=150)

    peaks_amp = []
    for p in peaks:
        peaks_amp.append(input_array[p])

    time_diff = []
    for t_peak in range(1, len(peaks)):
        time_diff.append((peaks[t_peak] - peaks[t_peak - 1]) * 0.008)

    amplitudes = np.array(peaks_amp)
    times = np.array(time_diff)
    return peaks, amplitudes, times


def train(model, epochs, n_skip_elems, X_train, y_train, X_val, y_val, log_interval, l_rate):
    model.to(DEVICE)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    train_losses = []
    test_losses = []
    test_scores = []
    best_so_far = np.inf
    counter = 0

    n_skip_elems = n_skip_elems
    noise = np.random.normal(noise_mean, noise_std, size=X_train.shape[0])
    noise_val = np.random.normal(noise_mean, noise_std, size=X_val.shape[0])

    for epoch in range(epochs):

        epoch_val_loss = 0
        train_loss = []

        # PRE-TRAIN
        train_idx = np.random.randint(low=(n_k + n_b)*n_skip_elems, high=X_train.shape[0])
        invals = []

        for k in range(1, n_a): invals.append(y_train[train_idx - k * n_skip_elems, i])
        for k in range(n_k, n_k + n_b): invals.append(X_train[train_idx - k * n_skip_elems, i] + noise[train_idx - k * n_skip_elems])
        # for k in range(n_c): invals.append(noise[train_idx - k * n_skip_elems])

        narmax_input_train = np.array(invals).reshape(-1, 1).transpose()
        narmax_output_train = y_train[train_idx, i].reshape(-1, 1)
        train_dataset = Data.TensorDataset(torch.from_numpy(narmax_input_train).float(),
                                          torch.from_numpy(narmax_output_train).float())

        val_idx = np.random.randint(low=(n_k + n_b)*n_skip_elems, high=X_val.shape[0])
        invals = []

        for k in range(1, n_a): invals.append(y_val[val_idx - k * n_skip_elems, i])
        for k in range(n_k, n_k + n_b): invals.append(X_val[val_idx - k * n_skip_elems, i] + noise_val[val_idx - k * n_skip_elems])
        # for k in range(n_c): invals.append(noise_val[val_idx - k * n_skip_elems])

        narmax_input_val = np.array(invals).reshape(-1, 1).transpose()
        narmax_output_val = y_val[val_idx, i].reshape(-1, 1)
        val_dataset = Data.TensorDataset(torch.from_numpy(narmax_input_val).float(),
                                        torch.from_numpy(narmax_output_val).float())

        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=False)
        val_loader = Data.DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            y_train_pred = model(batch_x)
            loss = criterion(y_train_pred, batch_y)
            train_loss.append(loss.item())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train_loss = np.sum(train_loss) / len(train_loader)
        train_losses.append(epoch_train_loss)
        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for step, (val_x, val_y) in enumerate(val_loader):
                val_x, val_y = val_x.to(DEVICE), val_y.to(DEVICE)
                val_pred = model(val_x)
                loss_val = criterion(val_pred, val_y)
                epoch_val_loss = (epoch_val_loss + loss_val) / len(val_loader)
                labels = torch.argmax(val_y, dim=1).view(-1, 1)
                predicted = torch.argmax(val_pred.data, 1).view(-1, 1)
                y_true.extend(labels.cpu().detach().numpy().tolist())
                y_pred.extend(predicted.cpu().detach().numpy().tolist())
            test_losses.append(epoch_val_loss.cpu().detach().numpy().tolist())
        print("Iteration: ", epoch, " Loss: ", epoch_train_loss, " Validation loss: ", epoch_val_loss)
        # test_score = classification_report(y_true, y_pred, zero_division=0, output_dict=False)
        test_scores.append(classification_report(y_true, y_pred, zero_division=0, output_dict=True))

        model.train()

        if (epoch + 1) % log_interval == 0:
            clear_output(wait=True)

            # plot training loss
            fig, ax = plt.subplots()
            ax.plot(train_losses, label='Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()

            # plot testing loss
            fig, ax = plt.subplots()
            ax.plot(test_losses, label='Testing Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()

            plt.show()


dataframe = pd.read_pickle('mlp_tests/datasets/dataset100_norm.pkl')

X = dataframe["x"]
Y = dataframe["y"]

data_x = np.empty((len(X.values), len(X.values[0]['data'])))
data_y = np.empty((len(Y.values), len(Y.values[0]['data']))) 
for i in range(0, len(X.values)):
    data_x[i, :] = X.values[i]['data']  # .flatten()
    data_y[i, :] = Y.values[i]['data']  # .flatten()

time = np.linspace(start=0, stop=60, num=data_x.shape[1])
xdata = np.transpose(data_x)
ydata = np.transpose(data_y)

random_pk_test = np.random.randint(low=0, high=ydata.shape[1])
peak_idxs, peak_vals, peak_lags = peak_to_peak(ydata[:, random_pk_test])

# create training and validation set
seed = 113
X_train, X_tmp, y_train, y_tmp = train_test_split(xdata, ydata, test_size=0.5, shuffle=False)
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)

# define NARMAX parameters
n_b = 10
n_a = n_c = 11
n_k = 38
noise_mean = 0
noise_std = 0.4127

narmax_len = n_a + n_b - 1

D_in, H, D_out = narmax_len, narmax_len, 1
model = MLP(D_in, H, D_out)

# load scalers
x_scalers = pickle.load(open('mlp_tests/x_minmax_scalers_100_0707.pkl', 'rb'))
y_scalers = pickle.load(open('mlp_tests/y_minmax_scalers_100_0707.pkl', 'rb'))

if TRAIN:
    epochs = 50000
    batch = narmax_len

    train(model, epochs, 1, X_train, y_train, X_val, y_val, epochs, 1e-4)

    # save model
    pickle.dump(model, open(FILE_NAME, 'wb'))

if PREDICT:
    SUB_SAMPLING = False
    n_skip_elems = 1

    # load model
    model = pickle.load(open(FILE_NAME, 'rb'))

    # See model predictions
    test_start_idx = 0
    pred_start_idx = (n_k + n_b)
    # test_start_idx = X_train.shape[0] + X_val.shape[0]
    # pred_start_idx = X_train.shape[0] + X_val.shape[0] + n_k + n_b
    rand_subj_idx = np.random.randint(low=0, high=xdata.shape[1])
    noise = np.random.normal(noise_mean, noise_std, size=xdata.shape[0])

    dataframe_raw = pd.read_pickle('mlp_tests/datasets/dataset100_raw.pkl')
    X_raw = dataframe_raw["x"]
    Y_raw = dataframe_raw["y"]
    data_x_raw = np.empty((len(X_raw.values), len(X_raw.values[0]['data'])))
    data_y_raw = np.empty((len(Y_raw.values), len(Y_raw.values[0]['data'])))
    for i in range(0, len(X.values)):
        data_x_raw[i, :] = X_raw.values[i]['data']
        data_y_raw[i, :] = Y_raw.values[i]['data']

    xdata_raw = np.transpose(data_x_raw)
    ydata_raw = np.transpose(data_y_raw)

    if MODE == 1:
        # AR with gtruth
        if SUB_SAMPLING:
            n_skip_elems = 30  # test 0.2 - max 27
            pred_start_idx = (n_k + n_b)*n_skip_elems
            pred_list = []
            for test_idx in range((n_k + n_b)*n_skip_elems, xdata.shape[0]):
                invals1 = []
                for k in range(1, n_a): invals1.append(ydata[test_idx - k*n_skip_elems, rand_subj_idx])
                for k in range(n_k, n_k + n_b): invals1.append(xdata[test_idx - k*n_skip_elems, rand_subj_idx])
                for k in range(n_c): invals1.append(noise[test_idx - k*n_skip_elems])
                narmax_input_test = np.array(invals1).transpose()
                pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
                pred_list.append(pred_tensor.cpu().detach().numpy())
            pred_vector = np.array(pred_list)
            prediction = pred_vector * np.std(pred_vector) + np.mean(pred_vector)
        else:
            pred_list = []
            for test_idx in range(n_k+n_b, xdata.shape[0]):
                invals1 = []
                for k in range(1, n_a): invals1.append(ydata[test_idx - k, rand_subj_idx])
                for k in range(n_k, n_k + n_b): invals1.append(xdata[test_idx - k, rand_subj_idx])
                # for k in range(n_c): invals1.append(noise[test_idx - k])
                narmax_input_test = np.array(invals1).transpose()
                pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
                pred_list.append(pred_tensor.cpu().detach().numpy())
            pred_vector = np.array(pred_list)
            # prediction = pred_vector * np.std(pred_vector) + np.mean(pred_vector)
            # prediction = y_scalers[rand_subj_idx].inverse_transform(pred_vector)
    elif MODE == 2:
        # AR with pred
        pred_list = []
        for test_idx in range(n_k+n_b, xdata.shape[0]):
            prev_pred_list = []
            for buff_idx in range(test_idx - n_a, test_idx):
                buffer = []
                for k in range(1, n_a): buffer.append(ydata[test_start_idx + buff_idx - k, rand_subj_idx])
                for k in range(n_k, n_k + n_b): buffer.append(xdata[test_start_idx + buff_idx - k, rand_subj_idx])
                for k in range(n_c): buffer.append(noise[buff_idx - k])
                narmax_input_prev = np.array(buffer).transpose()
                prev_pred_tensor = model(torch.from_numpy(narmax_input_prev).float().to(DEVICE))
                prev_pred_list.append(prev_pred_tensor.cpu().detach().numpy())
            prev_pred_vector = np.array(prev_pred_list)
            invals2 = []
            for k in range(1, n_a): invals2.append(prev_pred_vector[n_a - k])
            for k in range(n_k, n_k + n_b): invals2.append(xdata[test_idx - k, rand_subj_idx])
            for k in range(n_c): invals2.append(noise[test_idx - k])
            narmax_input_test = np.array(invals2, dtype=float).transpose()
            pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
            pred_list.append(pred_tensor.cpu().detach().numpy())
        pred_vector = np.array(pred_list)
        prediction = pred_vector * np.std(pred_vector) + np.mean(pred_vector)
    else:
        print("Wrong AR_MODE parameter, chose 1 for ground truth AR, 2 for predictions AR.")

    # plots
    plt_time = np.linspace(start=time[pred_start_idx], stop=60, num=xdata.shape[0]-(n_k+n_b)*n_skip_elems)
    max_Y = np.max(ydata_raw[:, rand_subj_idx])
    min_Y = np.min(ydata_raw[:, rand_subj_idx])
    y_denorm = ydata[(n_k + n_b):, rand_subj_idx] * (max_Y - min_Y) + min_Y
    prediction = pred_vector * (max_Y - min_Y) + min_Y
    # y_denorm = ydata[:, rand_subj_idx] * np.std(ydata[:, rand_subj_idx]) + np.mean(ydata[:, rand_subj_idx])
    # y_denorm = y_scalers[rand_subj_idx].inverse_transform(ydata[(n_k+n_b)*n_skip_elems:, rand_subj_idx].reshape(1, -1))
    fig_title = 'predicted vs actual force for experiment n: ' + str(rand_subj_idx)

    plt.figure(1)
    plt.plot(plt_time, prediction)
    plt.plot(plt_time, y_denorm, alpha=0.6)  # ydata[(n_k+n_b)*n_skip_elems:, rand_subj_idx]
    plt.legend(['prediction', 'truth'])
    plt.xlabel("time")
    plt.ylabel("force")
    plt.grid()
    plt.title(fig_title)

    # plt.figure(2)
    # plt.plot(plt_time, pred_vector)
    # plt.plot(plt_time, ydata[(n_k+n_b)*n_skip_elems:, rand_subj_idx], alpha=0.6)
    # plt.legend(['prediction', 'truth'])
    # plt.xlabel("time")
    # plt.ylabel("force")
    # plt.grid()
    # plt.title(fig_title)
    #
    # fig_title_2 = 'Measured force for experiment n: ' + str(rand_subj_idx)
    # plt.figure(3)
    # plt.plot(plt_time, ydata_raw[(n_k+n_b)*n_skip_elems:, rand_subj_idx])
    # plt.xlabel("time")
    # plt.ylabel("force")
    # plt.grid()
    # plt.title(fig_title)

    plt.show()
    plt.close()
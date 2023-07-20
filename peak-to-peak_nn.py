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
PREDICT = True

NET_FEATURES = 'narmax_pk2pk'
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


def train(model, epochs, X1_train, X2_train, y1_train, y2_train, X1_val, X2_val, y1_val, y2_val, log_interval, l_rate):
    model.to(DEVICE)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    train_losses = []
    test_losses = []
    test_scores = []

    noise = np.random.normal(noise_mean, noise_std, size=X1_train.shape[0])
    noise_val = np.random.normal(noise_mean, noise_std, size=X1_val.shape[0])

    for epoch in range(epochs):

        epoch_val_loss = 0
        train_loss = []
        i = 99

        # build train dataloader
        train_idx_1 = np.random.randint(low=(n_k + n_b), high=X1_train.shape[0])
        train_idx_2 = np.random.randint(low=n_b, high=X2_train.shape[0])
        invals1 = []
        invals2 = []
        for k in range(1, n_a): invals1.append(y1_train[train_idx_1 - k , i])
        for k in range(n_k, n_k + n_b): invals1.append(X1_train[train_idx_1 - k , i] + noise[train_idx_1 - k ])
        for k in range(1, n_a): invals2.append(y2_train[train_idx_2 - k , i])
        for k in range(0, n_b): invals2.append(X2_train[train_idx_2 - k , i])
        invals = np.vstack((np.array(invals1), np.array(invals2)))
        narmax_input_train = invals.reshape(-1, 1).transpose()
        outvals = []
        outvals.append(y1_train[train_idx_1, i])
        outvals.append(y2_train[train_idx_2, i])
        narmax_output_train = np.array(outvals).reshape(-1, 1).transpose()
        train_dataset = Data.TensorDataset(torch.from_numpy(narmax_input_train).float(), torch.from_numpy(narmax_output_train).float())

        # build val dataloader
        val_idx_1 = np.random.randint(low=(n_k + n_b), high=X1_val.shape[0])
        val_idx_2 = np.random.randint(low=n_b, high=X2_val.shape[0])
        invals1 = []
        invals2 = []
        for k in range(1, n_a): invals1.append(y1_val[val_idx_1 - k , i])
        for k in range(n_k, n_k + n_b): invals1.append(X1_val[val_idx_1 - k , i] + noise_val[val_idx_1 - k ])
        for k in range(1, n_a): invals2.append(y2_val[val_idx_2 - k , i])
        for k in range(0, n_b): invals2.append(X2_val[val_idx_2 - k , i])
        invals = np.vstack((np.array(invals1), np.array(invals2)))
        narmax_input_val = invals.reshape(-1, 1).transpose()
        outvals = []
        outvals.append(y1_val[val_idx_1, i])
        outvals.append(y2_val[val_idx_2, i])
        narmax_output_val = np.array(outvals).reshape(-1, 1).transpose()
        val_dataset = Data.TensorDataset(torch.from_numpy(narmax_input_val).float(), torch.from_numpy(narmax_output_val).float())

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


dataframe = pd.read_pickle('dataset100_norm.pkl')
# dataframe_xy = pd.read_pickle('dataset100_xy_norm_-1_1_m0.pkl')

# X1 = dataframe_xy["x1"]
# X2 = dataframe_xy["x2"]
X = dataframe["x"]
Y = dataframe["y"]

# data_x1 = np.empty((len(X1.values), len(X1.values[0]['data'])))
# data_x2 = np.empty((len(X2.values), len(X2.values[0]['data'])))
data_x = np.empty((len(X.values), len(X.values[0]['data'])))
data_y = np.empty((len(Y.values), len(Y.values[0]['data'])))
for i in range(0, len(X.values)):
    # data_x1[i, :] = X1.values[i]['data']
    # data_x2[i, :] = X2.values[i]['data']
    data_x[i, :] = X.values[i]['data']
    data_y[i, :] = Y.values[i]['data']

time = np.linspace(start=0, stop=60, num=data_x.shape[1])
# x1data = np.transpose(data_x1)
# x2data = np.transpose(data_x2)
xdata = np.transpose(data_x)
ydata = np.transpose(data_y)

# random_pk_test = np.random.randint(low=0, high=ydata.shape[1])
peak_locations = []
peak_values = []
peak_periods = []
for idx in range(0, ydata.shape[1]):
    peak_idxs, peak_vals, peak_lags = peak_to_peak(ydata[:, idx])
    peak_locations.append(peak_idxs)
    peak_values.append(peak_vals)
    peak_periods.append(peak_lags)

max_len = 0
for idx in range(0, ydata.shape[1]):
    if len(peak_locations[idx]) > max_len:
        max_len = len(peak_locations[idx])

for idx in range(0, ydata.shape[1]):
    peak_periods[idx] = np.append(peak_periods[idx], 0)
    if len(peak_locations[idx]) < max_len:
        for i in range(len(peak_locations[idx]), max_len):
            peak_locations[idx] = np.append(peak_locations[idx], 0)
            peak_values[idx] = np.append(peak_values[idx], 0)
            peak_periods[idx] = np.append(peak_periods[idx], 0)


pk_locs = np.transpose(np.array(peak_locations))
pk_vals = np.transpose(np.array(peak_values))
pk_period = np.transpose(np.array(peak_periods))

# set input and output vector
input1 = xdata
input2 = pk_locs
output1 = ydata
output2 = pk_period

# create training and validation set
seed = 113
X1_train, X_tmp, y1_train, y_tmp = train_test_split(input1, output1, test_size=0.5, random_state=seed)
X1_test, X1_val, y1_test, y1_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)
X2_train, X_tmp, y2_train, y_tmp = train_test_split(input2, output2, test_size=0.5, random_state=seed)
X2_test, X2_val, y2_test, y2_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)

# define NARMAX parameters
n_b = 5
n_a = n_c = 6
n_k = 38
noise_mean = 0
noise_std = 0.4127

narmax_len = n_a + n_b - 1

D_in, H, D_out = narmax_len*2, narmax_len*2, 2
model = MLP(D_in, H, D_out)

if TRAIN:
    epochs = 50000
    batch = narmax_len * 2

    train(model, epochs, X1_train, X2_train, y1_train, y2_train, X1_val, X2_val, y1_val, y2_val, epochs, 1e-4)

    # save model
    pickle.dump(model, open(FILE_NAME, 'wb'))

if PREDICT:
    # load model
    model = pickle.load(open(FILE_NAME, 'rb'))

    # See model predictions
    test_start_idx = 0
    pred_start_idx = (n_k + n_b)
    # test_start_idx = X_train.shape[0] + X_val.shape[0]
    # pred_start_idx = X_train.shape[0] + X_val.shape[0] + n_k + n_b
    rand_subj_idx = np.random.randint(low=0, high=xdata.shape[1])
    noise = np.random.normal(noise_mean, noise_std, size=xdata.shape[0])

    dataframe_raw = pd.read_pickle('dataset100_raw.pkl')
    X_raw = dataframe_raw["x"]
    Y_raw = dataframe_raw["y"]
    data_x_raw = np.empty((len(X_raw.values), len(X_raw.values[0]['data'])))
    data_y_raw = np.empty((len(Y_raw.values), len(Y_raw.values[0]['data'])))
    for i in range(0, len(X.values)):
        data_x_raw[i, :] = X_raw.values[i]['data']
        data_y_raw[i, :] = Y_raw.values[i]['data']

    xdata_raw = np.transpose(data_x_raw)
    ydata_raw = np.transpose(data_y_raw)

    pred_x_list = []
    pred_y_list = []
    for test_idx in range(0, pk_locs.shape[0]):  # pk_locs[n_b:, rand_subj_idx]:
        invals1 = []
        invals2 = []
        for k in range(1, n_a): invals1.append(ydata[pk_locs[test_idx, rand_subj_idx] - k, rand_subj_idx])
        for k in range(n_k, n_k + n_b): invals1.append(xdata[pk_locs[test_idx, rand_subj_idx] - k, rand_subj_idx])
        for k in range(1, n_a): invals2.append(pk_vals[test_idx - k, rand_subj_idx])
        for k in range(0, n_b): invals2.append(pk_period[test_idx - k, rand_subj_idx])
        invals = np.vstack((np.array(invals1), np.array(invals2)))
        narmax_input_test = invals.reshape(-1, 1).transpose()
        pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
        pred_x_list.append(pred_tensor[0, 0].cpu().detach().numpy())
        pred_y_list.append(pred_tensor[0, 1].cpu().detach().numpy())
    pred_x_vector = np.array(pred_x_list)
    pred_y_vector = np.array(pred_y_list)

    print("actual pk2pk intervals: ", pk_period[:, rand_subj_idx])
    print("predicted pk2pk intervals: ", pred_y_vector)

    # plots
    plt_time = np.linspace(start=time[pred_start_idx], stop=60, num=xdata.shape[0]-(n_k+n_b))
    max_Y = np.max(ydata_raw[:, rand_subj_idx])
    min_Y = np.min(ydata_raw[:, rand_subj_idx])
    y_denorm = (ydata[(n_k + n_b):, rand_subj_idx] + 1) * (max_Y - min_Y)/2 + min_Y
    prediction = (pred_x_vector + 1) * (max_Y - min_Y)/2 + min_Y
    fig_title = 'predicted vs actual force for experiment n: ' + str(rand_subj_idx)

    plt.figure(1)
    plt.plot(ydata[(n_k + n_b):, rand_subj_idx])
    plt.plot(pk_locs[:, rand_subj_idx], pred_x_vector, 'x')
    plt.plot(pk_locs[:, rand_subj_idx], pk_vals[:, rand_subj_idx], 'o')
    plt.legend(['signal', 'prediction', 'truth'])
    plt.xlabel("time")
    plt.ylabel("force peaks")
    plt.grid()
    plt.title(fig_title)

    plt.show()
    plt.close()
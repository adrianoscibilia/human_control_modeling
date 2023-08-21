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
from sklearn.metrics import r2_score
import sklearn.metrics as skl


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN = False
PREDICT = False
EVALUATE = True
MODEL_ORDER = 2
NET_FEATURES = 'narmax2_pk2pk_2108'  # 'narmax10_pk2pk_listed'
FILE_NAME = NET_FEATURES + '.pkl'
fig_dir = '/home/adriano/Pictures/NN_model_Figures/'


class MLP(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 2 * H)
        self.linear3 = torch.nn.Linear(2 * H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        self.to(DEVICE)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        y_pred = self.linear4(torch.tanh(x))
        return y_pred


def train(model, epochs, X1_train, y1_train, X1_val, y1_val, pk_idxs1, log_interval, l_rate):
    model.to(DEVICE)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    train_losses = []
    test_losses = []
    test_scores = []
    best_so_far = np.inf
    counter = 0
    # noise = np.random.normal(noise_mean, noise_std, size=X1_train.shape[0])
    # noise_val = np.random.normal(noise_mean, noise_std, size=X1_val.shape[0])
    for epoch in range(epochs):

        epoch_val_loss = 0
        train_loss = []
        train_subj = np.random.randint(low=0, high=X1_train.shape[1])

        # build train dataloader
        # train_peaks_1 = []
        # for sel_pk in X1_train[:, i]:
        #     if sel_pk < X1_train.shape[0]:
        #         train_peaks_1.append(sel_pk)
        train_idx_1 = np.random.choice(pk_idxs1[train_subj][MODEL_ORDER:])
        # train_peaks_2 = []
        # for sel_pk in X2_train[:, i]:
        #     if sel_pk < X1_train.shape[0]:
        #         train_peaks_2.append(sel_pk)
        # train_idx_2 = np.random.choice(pk_idxs2[train_subj][:])
        invals1 = []
        # invals2 = []
        for k in range(1, n_a): invals1.append(y1_train[train_idx_1 - k , train_subj])
        for k in range(n_k, n_k + n_b): invals1.append(X1_train[train_idx_1 - k , train_subj])  # + noise[train_idx_1 - k ])
        # for k in range(1, n_a): invals2.append(y2_train[train_idx_2 - k , train_subj])
        # for k in range(n_k, n_k + n_b): invals2.append(X2_train[train_idx_2 - k , train_subj])  # + noise[train_idx_2 - k ])
        invals = np.array(invals1)  # np.vstack((np.array(invals1), np.array(invals2)))
        narmax_input_train = invals.reshape(-1, 1).transpose()
        outvals = []
        outvals.append(y1_train[train_idx_1, train_subj])
        # outvals.append(y2_train[train_idx_2, train_subj])
        narmax_output_train = np.array(outvals).reshape(-1, 1).transpose()
        train_dataset = Data.TensorDataset(torch.from_numpy(narmax_input_train).float(), torch.from_numpy(narmax_output_train).float())

        # build val dataloader
        # while True:
        #     val_peaks_1 = []
        #     val_peaks_2 = []
        #     for sel_pk_1 in X1_val[:, i]:
        #         if sel_pk_1 > X1_train.shape[0]:
        #             val_peaks_1.append(sel_pk_1)
        #     for sel_pk_2 in X2_val[:, i]:
        #         if sel_pk_2 > X2_train.shape[0]:
        #             val_peaks_2.append(sel_pk_2)
        #     if (len(val_peaks_1) > 0) and (len(val_peaks_2) > 0):
        #         break
        #     i = np.random.randint(low=0, high=99)

        val_subject = np.random.randint(low=0, high=X1_val.shape[1])
        val_idx_1 = np.random.choice(pk_idxs1[X1_train.shape[1] + val_subject][MODEL_ORDER:])
        # val_idx_2 = np.random.choice(pk_idxs2[X1_train.shape[1] + val_subject][:])
        invals1 = []
        invals2 = []
        for k in range(1, n_a): invals1.append(y1_val[val_idx_1 - k , val_subject])
        for k in range(n_k, n_k + n_b): invals1.append(X1_val[val_idx_1 - k , val_subject])  # + noise_val[val_idx_1 - k ])
        # for k in range(1, n_a): invals2.append(y2_val[val_idx_2 - k , val_subject])
        # for k in range(n_k, n_k + n_b): invals2.append(X2_val[val_idx_2 - k , val_subject])  # + noise[val_idx_2 - k ])
        invals = np.array(invals1)  # np.vstack((np.array(invals1), np.array(invals2)))
        narmax_input_val = invals.reshape(-1, 1).transpose()
        outvals = []
        outvals.append(y1_val[val_idx_1, val_subject])
        # outvals.append(y2_val[val_idx_2, val_subject])
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
                if epoch_val_loss < best_so_far:
                    best_so_far = epoch_val_loss
                    counter = 0
                else:
                    counter += 1
                if counter > 5000:
                    break
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


def peak_to_peak(input_array, PLOT):
    height_adjusted = 0.1 - input_array[0]
    peaks, _ = signal.find_peaks(np.abs(input_array), height=height_adjusted, distance=150)

    peaks_amp = []
    for p in peaks:
        peaks_amp.append(input_array[p])

    # time_diff = []
    # for t_peak in range(1, len(peaks)):
    #     time_diff.append((peaks[t_peak] - peaks[t_peak - 1]) * 0.008)

    amplitudes = np.array(peaks_amp)
    # times = np.array(time_diff)

    if PLOT:
        plt.figure(1)
        plt.plot(input_array)
        plt.plot(peaks, amplitudes, 'x')
        plt.grid()
        plt.show()

    return peaks, amplitudes


# dataframe = pd.read_pickle('dataset_ref_f_mod_norm.pkl')
dataframe_xy = pd.read_pickle('dataset_ref_f_norm.pkl')

X1 = dataframe_xy["x1"]
X2 = dataframe_xy["x2"]
Y1 = dataframe_xy["y1"]
Y2 = dataframe_xy["y2"]
# X = dataframe["x"]
# Y = dataframe["y"]

# data_x1 = np.empty((len(X1.values), len(X1.values[0]['data'])))
# data_x2 = np.empty((len(X2.values), len(X2.values[0]['data'])))
# data_y1 = np.empty((len(Y1.values), len(Y1.values[0]['data'])))
# data_y2 = np.empty((len(Y2.values), len(Y2.values[0]['data'])))
data_x = np.empty((2*len(X1.values), len(X1.values[0]['data'])))
data_y = np.empty((2*len(Y1.values), len(Y1.values[0]['data'])))
for i in range (0, len(X1.values)):
    data_x[i, :] = X1.values[i]['data']
    data_x[i+100, :] = X2.values[i]['data']
    data_y[i, :] = Y1.values[i]['data']
    data_y[i+100, :] = Y2.values[i]['data']

time = np.linspace(start=0, stop=60, num=data_x.shape[1])
# x1data = np.transpose(data_x1)
# x2data = np.transpose(data_x2)
# y1data = np.transpose(data_y1)
# y2data = np.transpose(data_y2)
xdata = np.transpose(data_x)
ydata = np.transpose(data_y)

# random_pk_test = np.random.randint(low=0, high=ydata.shape[1])
peak_locations = []
peak_values = []
# peak1_periods = []
peak2_locations = []
peak2_values = []
# peak2_periods = []
for idx in range(0, ydata.shape[1]):
    peak_idxs, peak_vals = peak_to_peak(ydata[:, idx], False)
    peak_locations.append(peak_idxs)
    peak_values.append(peak_vals)
    # peak1_periods.append(peak_lags)
    # peak2_idxs, peak_vals = peak_to_peak(y2data[:, idx], False)
    # peak2_locations.append(peak2_idxs)
    # peak2_values.append(peak_vals)
    # # peak2_periods.append(peak_lags)

# min_len = 1000
# for idx in range(0, y1data.shape[1]):
#     if len(peak_values[idx]) < min_len:
#         min_len = len(peak_values[idx])
#     if len(peak2_values[idx]) < min_len:
#         min_len = len(peak2_values[idx])
#
# peak_locations_adj = []
# peak2_locations_adj = []
# peak2_values_adj = []
# peak_values_adj = []
# # peak_periods_adj = []
# for idx in range(0, y1data.shape[1]):
#     peak_locations_adj_row = []
#     peak2_locations_adj_row = []
#     peak_values_adj_row = []
#     peak2_values_adj_row = []
#     # peak_periods_adj_row = []
#     # peak_periods_adj_row.append(0)
#     for i in range(0, min_len):
#         peak_locations_adj_row.append(peak_locations[idx][i])
#         peak2_locations_adj_row.append(peak2_locations[idx][i])
#         peak_values_adj_row.append(peak_values[idx][i])
#         peak2_values_adj_row.append(peak2_values[idx][i])
#         # if i < min_len-1 :
#         #     peak_periods_adj_row.append(peak_periods[idx][i])
#     peak_locations_adj.append(peak_locations_adj_row)
#     peak2_locations_adj.append(peak2_locations_adj_row)
#     peak_values_adj.append(peak_values_adj_row)
#     peak2_values_adj.append(peak2_values_adj_row)
#     # peak_periods_adj.append(peak_periods_adj_row)
#
#
# pk1_locs = np.transpose(np.array(peak_locations))
# pk2_locs = np.transpose(np.array(peak2_locations))
# pk1_vals = np.transpose(np.array(peak_values))
# pk2_vals = np.transpose(np.array(peak2_values))
# # pk_period = np.transpose(np.array(peak_periods_adj))

# set input and output vector
input1 = np.transpose(xdata)
# input2 = np.transpose(x2data)
output1 = np.transpose(ydata)
# output2 = np.transpose(y2data)

# create training and validation set
seed = 113
X1_train, X1_tmp, y1_train, y1_tmp = train_test_split(input1, output1, test_size=0.5, random_state=seed)
X1_test, X1_val, y1_test, y1_val = train_test_split(X1_tmp, y1_tmp, test_size=0.5, random_state=seed)
# X2_train, X2_tmp, y2_train, y2_tmp = train_test_split(input2, output2, test_size=0.5, shuffle=False)
# X2_test, X2_val, y2_test, y2_val = train_test_split(X2_tmp, y2_tmp, test_size=0.5, shuffle=False)

input1 = np.transpose(input1)
# input2 = np.transpose(input2)
output1 =np.transpose(output1)
# output2 =np.transpose(output2)
X1_train = np.transpose(X1_train)
X1_val = np.transpose(X1_val)
X1_test = np.transpose(X1_test)
# X2_train = np.transpose(X2_train)
# X2_val = np.transpose(X2_val)
# X2_test = np.transpose(X2_test)
y1_train = np.transpose(y1_train)
y1_val = np.transpose(y1_val)
y1_test = np.transpose(y1_test)
# y2_train = np.transpose(y2_train)
# y2_val = np.transpose(y2_val)
# y2_test = np.transpose(y2_test)

# define NARMAX parameters
n_b = MODEL_ORDER
n_a = n_c = MODEL_ORDER + 1
n_k = 38
noise_mean = 0
noise_std = 0.4127

narmax_len = n_a + n_b - 1

D_in, H, D_out = narmax_len, narmax_len, 1
model = MLP(D_in, H, D_out)

if TRAIN:
    epochs = xdata.shape[1]  # 30 * xdata.shape[1] * 10
    batch = narmax_len * 2

    train(model, epochs, X1_train, y1_train, X1_val, y1_val, peak_locations, epochs, 1e-4)

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
    rand_subj_idx = 43  # np.random.randint(low=0, high=xdata.shape[1])
    noise = np.random.normal(noise_mean, noise_std, size=xdata.shape[0])

    dataframe_raw = pd.read_pickle('dataset_ref_f_raw.pkl')
    X1_raw = dataframe_raw["x1"]
    Y1_raw = dataframe_raw["y1"]
    X2_raw = dataframe_raw["x2"]
    Y2_raw = dataframe_raw["y2"]

    # data_x1_raw = np.empty((len(X1_raw.values), len(X1_raw.values[0]['data'])))
    # data_y1_raw = np.empty((len(Y1_raw.values), len(Y1_raw.values[0]['data'])))
    # data_x2_raw = np.empty((len(X2_raw.values), len(X2_raw.values[0]['data'])))
    # data_y2_raw = np.empty((len(Y2_raw.values), len(Y2_raw.values[0]['data'])))
    # for i in range(0, len(X1.values)):
    #     data_x1_raw[i, :] = X1_raw.values[i]['data']
    #     data_y1_raw[i, :] = Y1_raw.values[i]['data']
    #     data_x2_raw[i, :] = X2_raw.values[i]['data']
    #     data_y2_raw[i, :] = Y2_raw.values[i]['data']
    # x1data_raw = np.transpose(data_x1_raw)
    # y1data_raw = np.transpose(data_y1_raw)
    # x2data_raw = np.transpose(data_x1_raw)
    # y2data_raw = np.transpose(data_y1_raw)

    data_x_raw = np.empty((2 * len(X1_raw.values), len(X1_raw.values[0]['data'])))
    data_y_raw = np.empty((2 * len(Y1_raw.values), len(Y1_raw.values[0]['data'])))
    for i in range(0, len(X1.values)):
        data_x_raw[i, :] = X1_raw.values[i]['data']
        data_x_raw[i + 100, :] = X2_raw.values[i]['data']
        data_y_raw[i, :] = Y1_raw.values[i]['data']
        data_y_raw[i + 100, :] = Y2_raw.values[i]['data']
    xdata_raw = np.transpose(data_x_raw)
    ydata_raw = np.transpose(data_y_raw)

    pred_list = []
    # pred_y_list = []

    # lenghts = []
    # lenghts.append(len(peak_locations[rand_subj_idx][:]))
    # lenghts.append(len(peak2_locations[rand_subj_idx][:]))
    # minlen = np.min(lenghts)

    for test_idx in range(0, len(peak_locations[rand_subj_idx][:])):
        invals1 = []
        # invals2 = []
        for k in range(1, n_a): invals1.append(ydata[peak_locations[rand_subj_idx][test_idx] - k, rand_subj_idx])
        for k in range(n_k, n_k + n_b): invals1.append(xdata[peak_locations[rand_subj_idx][test_idx] - k, rand_subj_idx])
        # for k in range(1, n_a): invals2.append(y2data[peak2_locations[rand_subj_idx][test_idx] - k, rand_subj_idx])
        # for k in range(n_k, n_k + n_b): invals2.append(x2data[peak2_locations[rand_subj_idx][test_idx] - k, rand_subj_idx])
        invals = np.array(invals1)  #  np.vstack((np.array(invals1), np.array(invals2)))
        narmax_input_test = invals.reshape(-1, 1).transpose()
        pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
        pred_list.append(pred_tensor.cpu().detach().numpy())
        # pred_y_list.append(pred_tensor[0, 1].cpu().detach().numpy())
    pred_vector = np.array(pred_list)
    # pred_y_vector = np.array(pred_y_list)

    # print("actual pk2pk intervals [", len(pk_period[:, rand_subj_idx]), "]: ", pk_period[:, rand_subj_idx])
    # print("predicted pk2pk intervals [ ",  len(pred_y_vector), "]: ",  pred_y_vector)

    # plots
    plt_time = np.linspace(start=time[pred_start_idx], stop=60, num=xdata.shape[0]-(n_k+n_b))
    max_Y = np.max(ydata_raw[:, rand_subj_idx])
    min_Y = np.min(ydata_raw[:, rand_subj_idx])
    # max_Y2 = np.max(y2data_raw[:, rand_subj_idx])
    # min_Y2 = np.min(y2data_raw[:, rand_subj_idx])
    y_denorm = (ydata[:, rand_subj_idx] + 1) * (max_Y - min_Y)/2 + min_Y
    pk_denorm = (peak_values[rand_subj_idx][:] + 1) * (max_Y - min_Y)/2 + min_Y
    # y2_denorm = (y2data[(n_k + n_b):, rand_subj_idx] + 1) * (max_Y2 - min_Y2)/2 + min_Y2
    prediction = (pred_vector[:, 0, 0] + 1) * (max_Y - min_Y)/2 + min_Y
    # prediction_y = (pred_y_vector + 1) * (max_Y2 - min_Y2)/2 + min_Y2
    fig_title = 'predicted vs actual force for experiment n: ' + str(rand_subj_idx)

    plt.figure(1)
    plt.plot(y_denorm)
    plt.plot(peak_locations[rand_subj_idx][:], prediction, 'x')
    plt.plot(peak_locations[rand_subj_idx][:], pk_denorm, 'o', alpha=0.5)
    plt.legend(['signal', 'prediction', 'truth'])
    plt.xlabel("time")
    plt.ylabel("force peaks")
    plt.grid()
    plt.title(fig_title)

    plt.show()
    plt.close()

if EVALUATE:
    # load model
    model = pickle.load(open(FILE_NAME, 'rb'))

    # See model predictions
    test_start_idx = 0
    pred_start_idx = (n_k + n_b)
    noise = np.random.normal(noise_mean, noise_std, size=xdata.shape[0])

    MSE = []
    R2 = []
    for subj_idx in range(0, xdata.shape[1]):
        pred_list = []
        for test_idx in range(0, len(peak_locations[subj_idx][:])):
            invals = []
            for k in range(1, n_a): invals.append(ydata[peak_locations[subj_idx][test_idx] - k, subj_idx])
            for k in range(n_k, n_k + n_b): invals.append(xdata[peak_locations[subj_idx][test_idx] - k, subj_idx])
            invals = np.array(invals)  # np.vstack((np.array(invals1), np.array(invals2)))
            narmax_input_test = invals.reshape(-1, 1).transpose()
            pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
            pred_list.append(pred_tensor.cpu().detach().numpy())
        pred_vector = np.array(pred_list)
        MSE.append(skl.mean_squared_error(y_true=peak_values[subj_idx][:], y_pred=pred_vector[:, 0, 0]))
        R2.append(r2_score(y_true=peak_values[subj_idx][:],  y_pred=pred_vector[:, 0, 0]))

    scores_df = pd.DataFrame({
        'MSE x': MSE,
        'R2 score': R2,
    })

    writer = pd.ExcelWriter("pk2pk_scores_table.xlsx", engine='xlsxwriter')
    scores_df.to_excel(writer, sheet_name='Sheet1', startrow=1, header=False, index=False)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # Get the dimensions of the dataframe.
    (max_row, max_col) = scores_df.shape

    # Create a list of column headers, to use in add_table().
    column_settings = []
    for header in scores_df.columns:
        column_settings.append({'header': header})

    # Add the table.
    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

    # Make the columns wider for clarity.
    worksheet.set_column(0, max_col - 1, 12)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
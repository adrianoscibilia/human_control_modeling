import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils.data as Data
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
import sklearn.metrics as skl
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import xlsxwriter


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
TRAIN = True
PREDICT = True
EVALUATE = True

NET_FEATURES = 'net5_narmax10_xy_tr50_summary_cpu'
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


def train(model, epochs, X1_train, X2_train, y1_train, y2_train, X1_val, X2_val, y1_val, y2_val, log_interval, l_rate):
    model.to(DEVICE)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    train_losses = []
    test_losses = []
    test_scores = []

    noise = np.random.normal(noise_mean, noise_std, size=X1_train.shape[0])
    noise_val = np.random.normal(noise_mean, noise_std, size=X1_val.shape[0])
    best_so_far = np.inf
    counter = 0

    for epoch in range(epochs):

        epoch_val_loss = 0
        train_loss = []

        # build train dataloader
        train_idx = np.random.randint(low=(n_k + n_b), high=X1_train.shape[0])
        subj_idx = np.random.randint(low=0, high=X1_train.shape[1])
        invals1 = []
        invals2 = []
        for k in range(1, n_a): invals1.append(y1_train[train_idx - k , subj_idx])
        for k in range(n_k, n_k + n_b): invals1.append(X1_train[train_idx - k , subj_idx] + noise[train_idx - k ])
        for k in range(1, n_a): invals2.append(y2_train[train_idx - k , subj_idx])
        for k in range(n_k, n_k + n_b): invals2.append(X2_train[train_idx - k , subj_idx] + noise[train_idx - k ])
        invals = np.vstack((np.array(invals1), np.array(invals2)))
        narmax_input_train = invals.reshape(-1, 1).transpose()
        outvals = []
        outvals.append(y1_train[train_idx, subj_idx])
        outvals.append(y2_train[train_idx, subj_idx])
        narmax_output_train = np.array(outvals).reshape(-1, 1).transpose()
        train_dataset = Data.TensorDataset(torch.from_numpy(narmax_input_train).float(), torch.from_numpy(narmax_output_train).float())

        # build val dataloader
        val_idx = np.random.randint(low=(n_k + n_b), high=X1_val.shape[0])
        invals1 = []
        invals2 = []
        for k in range(1, n_a): invals1.append(y1_val[val_idx - k , subj_idx])
        for k in range(n_k, n_k + n_b): invals1.append(X1_val[val_idx - k , subj_idx] + noise_val[val_idx - k ])
        for k in range(1, n_a): invals2.append(y2_val[val_idx - k , subj_idx])
        for k in range(n_k, n_k + n_b): invals2.append(X2_val[val_idx - k , subj_idx] + noise_val[val_idx - k ])
        invals = np.vstack((np.array(invals1), np.array(invals2)))
        narmax_input_val = invals.reshape(-1, 1).transpose()
        outvals = []
        outvals.append(y1_val[val_idx, subj_idx])
        outvals.append(y2_val[val_idx, subj_idx])
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


dataframe = pd.read_pickle('dataset/dataset_ref_f_norm.pkl')

X1 = dataframe["x1"]
X2 = dataframe["x2"]
Y1 = dataframe["y1"]
Y2 = dataframe["y2"]

data_x1 = np.empty((len(X1.values), len(X1.values[0]['data'])))
data_x2 = np.empty((len(X2.values), len(X2.values[0]['data'])))
data_y1 = np.empty((len(Y1.values), len(Y1.values[0]['data'])))
data_y2 = np.empty((len(Y2.values), len(Y2.values[0]['data'])))

for i in range(0, len(X1.values)):
    data_x1[i, :] = X1.values[i]['data']
    data_x2[i, :] = X2.values[i]['data']
    data_y1[i, :] = Y1.values[i]['data']
    data_y2[i, :] = Y2.values[i]['data']

time = np.linspace(start=0, stop=60, num=data_x1.shape[1])
x1data = np.transpose(data_x1)
x2data = np.transpose(data_x2)
y1data = np.transpose(data_y1)
y2data = np.transpose(data_y2)

x1data_summary = np.empty((len(X1.values[0]['data']), 10))
x2data_summary = np.empty((len(X2.values[0]['data']), 10))
y1data_summary = np.empty((len(Y1.values[0]['data']), 10))
y2data_summary = np.empty((len(Y2.values[0]['data']), 10))

i = 0
for idx in range(0, len(X1.values)):
    if idx % 10 == 0:
        x1data_summary[:, i] = x1data[:, idx]
        x2data_summary[:, i] = x2data[:, idx]
        y1data_summary[:, i] = y1data[:, idx]
        y2data_summary[:, i] = y2data[:, idx]
        i += 1


# create training and validation set
seed = 113
X1_train, X_tmp, y1_train, y_tmp = train_test_split(x1data_summary, y1data_summary, test_size=0.5, shuffle=False)
X1_test, X1_val, y1_test, y1_val = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)
X2_train, X_tmp, y2_train, y_tmp = train_test_split(x2data_summary, y2data_summary, test_size=0.5, shuffle=False)
X2_test, X2_val, y2_test, y2_val = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)

# define NARMAX parameters
n_b = 10
n_a = n_c = 11
n_k = 38
noise_mean = 0
noise_std = 0.4127

narmax_len = n_a + n_b - 1

D_in, H, D_out = narmax_len * 2, narmax_len * 2, 2
model = MLP(D_in, H, D_out)

if TRAIN:
    epochs = 50000 * X1_train.shape[1]
    batch = narmax_len * 2

    train(model, epochs, X1_train, X2_train, y1_train, y2_train, X1_val, X2_val, y1_val, y2_val, epochs, 1e-4)

    # save model
    pickle.dump(model, open(FILE_NAME, 'wb'))

if PREDICT:
    n_skip_elems = 1
    RANDSTART = 85
    RANDSTOP = RANDSTART + 5
    # load model
    model = pickle.load(open(FILE_NAME, 'rb'))

    # See model prediction
    test_start_idx = 0
    pred_start_idx = (n_k + n_b - 1)*n_skip_elems
    rand_subj_idx = np.random.randint(low=RANDSTART, high=RANDSTOP)
    # rand_subj_idx = np.random.randint(low=0, high=x1data.shape[1])
    noise = np.random.normal(noise_mean, noise_std, size=x1data.shape[0])

    dataframe_raw = pd.read_pickle('dataset/dataset_ref_f_raw.pkl')
    X1_raw = dataframe_raw["x1"]
    X2_raw = dataframe_raw["x2"]
    Y1_raw = dataframe_raw["y1"]
    Y2_raw = dataframe_raw["y2"]
    data_x1_raw = np.empty((len(X1_raw.values), len(X1_raw.values[0]['data'])))
    data_x2_raw = np.empty((len(X2_raw.values), len(X2_raw.values[0]['data'])))
    data_y1_raw = np.empty((len(Y1_raw.values), len(Y1_raw.values[0]['data'])))
    data_y2_raw = np.empty((len(Y2_raw.values), len(Y2_raw.values[0]['data'])))
    for i in range(0, len(X1.values)):
        data_x1_raw[i, :] = X1_raw.values[i]['data']
        data_y1_raw[i, :] = Y1_raw.values[i]['data']
        data_x2_raw[i, :] = X2_raw.values[i]['data']
        data_y2_raw[i, :] = Y2_raw.values[i]['data']

    x1data_raw = np.transpose(data_x1_raw)
    x2data_raw = np.transpose(data_x2_raw)
    y1data_raw = np.transpose(data_y1_raw)
    y2data_raw = np.transpose(data_y2_raw)

    # x1data_raw_summary = np.empty((len(X1_raw.values[0]['data']), 10))
    # x2data_raw_summary = np.empty((len(X2_raw.values[0]['data']), 10))
    # y1data_raw_summary = np.empty((len(Y1_raw.values[0]['data']), 10))
    # y2data_raw_summary = np.empty((len(Y2_raw.values[0]['data']), 10))
    #
    # i = 0
    # for idx in range(0, len(X1.values)):
    #     if idx % 10 == 0:
    #         x1data_raw_summary[:, i] = x1data_raw[:, idx]
    #         x2data_raw_summary[:, i] = x2data_raw[:, idx]
    #         y1data_raw_summary[:, i] = y1data_raw[:, idx]
    #         y2data_raw_summary[:, i] = y2data_raw[:, idx]
    #         i += 1

    pred_x_list = []
    pred_y_list = []
    for test_idx in range(n_skip_elems*(n_k+n_b-1), x1data_summary.shape[0]):
        invals1 = []
        invals2 = []
        for k in range(0, n_a-1): invals1.append(y1data[test_idx - k*n_skip_elems, rand_subj_idx])
        for k in range(n_k-1, n_k + n_b - 1): invals1.append(x1data[test_idx - k*n_skip_elems, rand_subj_idx])
        for k in range(0, n_a-1): invals2.append(y2data[test_idx - k*n_skip_elems, rand_subj_idx])
        for k in range(n_k-1, n_k + n_b - 1): invals2.append(x2data[test_idx - k*n_skip_elems, rand_subj_idx])
        invals = np.vstack((np.array(invals1), np.array(invals2)))
        narmax_input_test = invals.reshape(-1, 1).transpose()
        pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
        pred_x_list.append(pred_tensor[0, 0].cpu().detach().numpy())
        pred_y_list.append(pred_tensor[0, 1].cpu().detach().numpy())
    pred_x_vector = np.array(pred_x_list)
    pred_y_vector = np.array(pred_y_list)

    # plots
    plt_time = np.linspace(start=time[pred_start_idx], stop=60, num=x1data.shape[0]-(n_k+n_b-1)*n_skip_elems)
    max_Yx = np.max(y1data_raw[:, rand_subj_idx])
    max_Yy = np.max(y2data_raw[:, rand_subj_idx])
    min_Yx = np.min(y1data_raw[:, rand_subj_idx])
    min_Yy = np.min(y2data_raw[:, rand_subj_idx])
    y1_denorm = (y1data[(n_k + n_b - 1)*n_skip_elems:, rand_subj_idx] + 1) * (max_Yx - min_Yx)/2 + min_Yx
    y2_denorm = (y2data[(n_k + n_b - 1)*n_skip_elems:, rand_subj_idx] + 1) * (max_Yy - min_Yy)/2 + min_Yy
    prediction_x = (pred_x_vector + 1) * (max_Yx - min_Yx)/2 + min_Yx
    prediction_y = (pred_y_vector + 1) * (max_Yy - min_Yy)/2 + min_Yy
    fig_title = 'predicted vs actual force for experiment n: ' + str(rand_subj_idx)

    plt.figure(1)
    plt.plot(plt_time, prediction_x)
    plt.plot(plt_time, y1_denorm, alpha=0.6)
    plt.legend(['prediction', 'truth'])
    plt.xlabel("time")
    plt.ylabel("force x")
    plt.grid()
    plt.title(fig_title)

    plt.figure(2)
    plt.plot(plt_time, prediction_y)
    plt.plot(plt_time, y2_denorm, alpha=0.6)
    plt.legend(['prediction', 'truth'])
    plt.xlabel("time")
    plt.ylabel("force y")
    plt.grid()
    plt.title(fig_title)

    plt.show()
    plt.close()

if EVALUATE:
    # load model
    model = pickle.load(open(FILE_NAME, 'rb'))

    dataframe_raw = pd.read_pickle('dataset/dataset_ref_f_raw.pkl')
    X1_raw = dataframe_raw["x1"]
    X2_raw = dataframe_raw["x2"]
    Y1_raw = dataframe_raw["y1"]
    Y2_raw = dataframe_raw["y2"]
    data_x1_raw = np.empty((len(X1_raw.values), len(X1_raw.values[0]['data'])))
    data_x2_raw = np.empty((len(X2_raw.values), len(X2_raw.values[0]['data'])))
    data_y1_raw = np.empty((len(Y1_raw.values), len(Y1_raw.values[0]['data'])))
    data_y2_raw = np.empty((len(Y2_raw.values), len(Y2_raw.values[0]['data'])))
    for i in range(0, len(X1.values)):
        data_x1_raw[i, :] = X1_raw.values[i]['data']
        data_y1_raw[i, :] = Y1_raw.values[i]['data']
        data_x2_raw[i, :] = X2_raw.values[i]['data']
        data_y2_raw[i, :] = Y2_raw.values[i]['data']
    x1data_raw = np.transpose(data_x1_raw)
    x2data_raw = np.transpose(data_x2_raw)
    y1data_raw = np.transpose(data_y1_raw)
    y2data_raw = np.transpose(data_y2_raw)

    # See model predictions
    test_start_idx = 0
    pred_start_idx = (n_k + n_b)
    noise = np.random.normal(noise_mean, noise_std, size=x1data.shape[0])

    MSE_x = []
    MSE_y = []
    R2_x = []
    R2_y = []
    for subj_idx in range(0, x1data.shape[1]):
        pred_x_list = []
        pred_y_list = []
        for test_idx in range(n_k + n_b, x1data_summary.shape[0]):
            invals1 = []
            invals2 = []
            for k in range(1, n_a): invals1.append(y1data[test_idx - k, subj_idx])
            for k in range(n_k, n_k + n_b): invals1.append(x1data[test_idx - k, subj_idx])
            for k in range(1, n_a): invals2.append(y2data[test_idx - k, subj_idx])
            for k in range(n_k, n_k + n_b): invals2.append(x2data[test_idx - k, subj_idx])
            invals = np.vstack((np.array(invals1), np.array(invals2)))
            narmax_input_test = invals.reshape(-1, 1).transpose()
            pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
            pred_x_list.append(pred_tensor[0, 0].cpu().detach().numpy())
            pred_y_list.append(pred_tensor[0, 1].cpu().detach().numpy())
        pred_x_vector = np.array(pred_x_list)
        pred_y_vector = np.array(pred_y_list)
        max_Yx = np.max(y1data_raw[:, subj_idx])
        max_Yy = np.max(y2data_raw[:, subj_idx])
        min_Yx = np.min(y1data_raw[:, subj_idx])
        min_Yy = np.min(y2data_raw[:, subj_idx])
        y1_denorm = (y1data[(n_k + n_b):, subj_idx]+1) * (max_Yx - min_Yx) / 2 + min_Yx
        y2_denorm = (y2data[(n_k + n_b):, subj_idx]+1) * (max_Yy - min_Yy) / 2 + min_Yy
        prediction_x = (pred_x_vector + 1) * (max_Yx - min_Yx) / 2 + min_Yx
        prediction_y = (pred_y_vector + 1) * (max_Yy - min_Yy) / 2 + min_Yy
        MSE_x.append(skl.mean_squared_error(y_true=y1_denorm, y_pred=prediction_x, squared=False))
        MSE_y.append(skl.mean_squared_error(y_true=y2_denorm, y_pred=prediction_y, squared=False))
        R2_x.append(r2_score(y_true=y1data[(n_k + n_b):, subj_idx],  y_pred=pred_x_vector))
        R2_y.append(r2_score(y_true=y2data[(n_k + n_b):, subj_idx], y_pred=pred_y_vector))

    scores_df = pd.DataFrame({
        'MSE x': MSE_x,
        'MSE y': MSE_y,
        'R2 score x': R2_x,
        'R2 score y': R2_y
    })

    writer = pd.ExcelWriter("tables/scores_table_2308.xlsx", engine='xlsxwriter')
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

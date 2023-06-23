import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils.data as Data
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from scipy.io import loadmat


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN = False
PREDICT = True
SUB_SAMPLING = True
AR_MODE = 2

NET_FEATURES = 'mlp_narmax_s10-18-27'
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


class FullyConnectedNN(torch.nn.Module):
    def __init__(self, input_len, output_len, hidden_dim, depth):
        super(FullyConnectedNN, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.fc_layers = torch.nn.Sequential()

        for i in range(depth):
            in_features = self.input_len if i == 0 else self.hidden_dim
            self.fc_layers.add_module(f"fc{i}", torch.nn.Linear(in_features, self.hidden_dim))
            self.fc_layers.add_module(f"relu{i}", torch.nn.ReLU())

        self.classifier = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, self.output_len),
                                              torch.nn.Softmax(dim=1))
        self.to(DEVICE)
        # print number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, x):
        x = self.fc_layers(x.view(-1, self.input_len))
        x = self.classifier(x)
        return x


def train(model, epochs, train_loader, val_loader, log_interval, l_rate):
    model.to(DEVICE)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    train_losses = []
    test_losses = []
    test_scores = []
    best_so_far = np.inf
    counter = 0
    for epoch in range(epochs):
        epoch_val_loss = 0
        train_loss = []
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
            if counter > 20:
                clear_output(wait=True)

                # plot testing loss
                fig, ax = plt.subplots()
                ax.plot(test_losses, label='Testing Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()

                # # plot average f1 score
                # fig, ax = plt.subplots()
                # ax.plot([score['macro avg']['f1-score'] for score in test_scores], label='Testing F1 Score Macro Avg')
                # ax.plot([score['macro avg']['precision'] for score in test_scores],
                #         label='Testing Precision Score Macro Avg')
                # ax.plot([score['macro avg']['recall'] for score in test_scores], label='Testing Recall Score Macro Avg')
                # ax.set_xlabel('Epoch')
                # ax.set_ylabel('Score')
                # ax.legend()

                plt.show()

                # print(
                #     f"Epoch {epoch + 1}/{epochs}: Training Loss: {epoch_train_loss:.4f} Test Loss: {epoch_val_loss:.4f} \nTest Score:\n {test_score} ")
                break
            test_losses.append(epoch_val_loss.cpu().detach().numpy().tolist())
        print("Iteration: ", epoch, " Loss: ", epoch_train_loss, " Validation loss: ", epoch_val_loss)
        test_score = classification_report(y_true, y_pred, zero_division=0, output_dict=False)
        test_scores.append(classification_report(y_true, y_pred, zero_division=0, output_dict=True))

        model.train()

        if (epoch + 1) % log_interval == 0:
            clear_output(wait=True)

            # # plot testing loss
            # fig, ax = plt.subplots()
            # ax.plot(test_losses, label='Testing Loss')
            # ax.set_xlabel('Epoch')
            # ax.set_ylabel('Loss')
            # ax.legend()
            #
            # # plot average f1 score
            # fig, ax = plt.subplots()
            # ax.plot([score['macro avg']['f1-score'] for score in test_scores], label='Testing F1 Score Macro Avg')
            # ax.plot([score['macro avg']['precision'] for score in test_scores],
            #         label='Testing Precision Score Macro Avg')
            # ax.plot([score['macro avg']['recall'] for score in test_scores], label='Testing Recall Score Macro Avg')
            # ax.set_xlabel('Epoch')
            # ax.set_ylabel('Score')
            # ax.legend()
            #
            # plt.show()

            # print(
            #     f"Epoch {epoch + 1}/{epochs}: Training Loss: {epoch_train_loss:.4f} Test Loss: {epoch_val_loss:.4f} \nTest Score:\n {test_score} ")


def pre_train(n_skip_elems, X_train, y_train, X_val, y_val):
    n_skip_elems = n_skip_elems
    train_idx = np.random.randint(low=(n_k + n_b), high=X_train.shape[0])
    invals = []
    noise = np.random.normal(noise_mean, noise_std, size=X_train.shape[0])
    for k in range(1, n_a): invals.append(y_train[train_idx - k * n_skip_elems, i])
    for k in range(n_k, n_k + n_b): invals.append(X_train[train_idx - k * n_skip_elems, i])
    for k in range(n_c): invals.append(noise[train_idx - k * n_skip_elems])

    narmax_input_train = np.array(invals).reshape(-1, 1).transpose()
    narmax_output_train = y_train[train_idx, i].reshape(-1, 1)
    train_dataset = Data.TensorDataset(torch.from_numpy(narmax_input_train).float(),
                                       torch.from_numpy(narmax_output_train).float())

    val_idx = np.random.randint(low=(n_k + n_b), high=X_val.shape[0])
    invals = []
    noise = np.random.normal(noise_mean, noise_std, size=X_val.shape[0])
    for k in range(1, n_a): invals.append(y_val[val_idx - k * n_skip_elems, i])
    for k in range(n_k, n_k + n_b): invals.append(X_val[val_idx - k * n_skip_elems, i])
    for k in range(n_c): invals.append(noise[val_idx - k * n_skip_elems])

    narmax_input_val = np.array(invals).reshape(-1, 1).transpose()
    narmax_output_val = y_val[val_idx, i].reshape(-1, 1)
    val_dataset = Data.TensorDataset(torch.from_numpy(narmax_input_val).float(),
                                     torch.from_numpy(narmax_output_val).float())

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=False)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)
    return train_loader, val_loader

dataframe = pd.read_pickle('./dataset_error_force_del_norm_1705.pkl')

X = dataframe["x"]
Y = dataframe["y"]

data_x = np.empty((len(X.values), len(X.values[0]["data"])))
data_y = np.empty((len(Y.values), len(Y.values[0]["data"])))
for i in range(0, len(X.values)):
    data_x[i, :] = X.values[i]["data"]
    data_y[i, :] = Y.values[i]["data"]

time = np.linspace(start=0, stop=60, num=data_x.shape[1])
xdata = np.transpose(data_x)
ydata = np.transpose(data_y)

# create training and validation set
seed = 113
X_train, X_tmp, y_train, y_tmp = train_test_split(xdata, ydata, test_size=0.3, shuffle=False)
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)

# define NARMAX parameters
n_b = 5
n_a = n_c = 6
n_k = 38
noise_mean = 0
noise_std = 0.4127

narmax_len = n_a + n_b + n_c - 1

D_in, H, D_out = narmax_len, narmax_len, 1
model = MLP(D_in, H, D_out)

if TRAIN:
    # initialize model and start training
    epochs = 10
    batch = narmax_len

    for i in range(0, X_train.shape[1]):
        for rep in range(epochs):
            train_loader, val_loader = pre_train(1, X_train, y_train, X_val, y_val)
            train(model=model, epochs=epochs, train_loader=train_loader, val_loader=val_loader, log_interval=epochs,
                  l_rate=1e-4)

            if SUB_SAMPLING:
                train_loader, val_loader = pre_train(10, X_train, y_train, X_val, y_val)
                train(model=model, epochs=epochs, train_loader=train_loader, val_loader=val_loader, log_interval=epochs,
                      l_rate=1e-4)
                train_loader, val_loader = pre_train(18, X_train, y_train, X_val, y_val)
                train(model=model, epochs=epochs, train_loader=train_loader, val_loader=val_loader, log_interval=epochs,
                      l_rate=1e-4)
                train_loader, val_loader = pre_train(27, X_train, y_train, X_val, y_val)
                train(model=model, epochs=epochs, train_loader=train_loader, val_loader=val_loader, log_interval=epochs,
                      l_rate=1e-4)

    # save model
    pickle.dump(model, open(FILE_NAME, 'wb'))

if PREDICT:
    # load model
    model = pickle.load(open(FILE_NAME, 'rb'))

    # # Test metrics OLD
    # r2_scores = np.empty(data_x.shape[0])
    # for i in range(data_x.shape[0]):
    #     out_tensor = model(torch.from_numpy(data_x[i, -D_in:]).float().to(DEVICE))
    #     out_vector = out_tensor.cpu().detach().numpy()
    #     r2_scores[i] = r2_score(y_true=data_y[i, -D_out:], y_pred=out_vector)

    # See model predictions
    test_start_idx = X_train.shape[0] + X_val.shape[0]
    pred_start_idx = X_train.shape[0] + X_val.shape[0] + n_k + n_b
    rand_subj_idx = np.random.randint(low=0, high=X_test.shape[1])
    noise = np.random.normal(noise_mean, noise_std, size=X_test.shape[0])

    if AR_MODE == 1:
        # AR with gtruth
        if SUB_SAMPLING:
            n_skip_elems = 27  # max 27
            pred_list = []
            for test_idx in range(n_k + n_b, X_test.shape[0]):
                invals1 = []
                for k in range(1, n_a): invals1.append(y_test[test_idx - k*n_skip_elems, rand_subj_idx])
                for k in range(n_k, n_k + n_b): invals1.append(X_test[test_idx - k*n_skip_elems, rand_subj_idx])
                for k in range(n_c): invals1.append(noise[test_idx - k*n_skip_elems])
                narmax_input_test = np.array(invals1).transpose()
                pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
                pred_list.append(pred_tensor.cpu().detach().numpy())
            pred_vector = np.array(pred_list)
        else:
            pred_list = []
            for test_idx in range(n_k+n_b, X_test.shape[0]):
                invals1 = []
                for k in range(1, n_a): invals1.append(y_test[test_idx - k, rand_subj_idx])
                for k in range(n_k, n_k + n_b): invals1.append(X_test[test_idx - k, rand_subj_idx])
                for k in range(n_c): invals1.append(noise[test_idx - k])
                narmax_input_test = np.array(invals1).transpose()
                pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
                pred_list.append(pred_tensor.cpu().detach().numpy())
            pred_vector = np.array(pred_list)
    if AR_MODE == 2:
        # AR with pred
        pred_list = []
        for test_idx in range(n_k+n_b, X_test.shape[0]):
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
            for k in range(n_k, n_k + n_b): invals2.append(X_test[test_idx - k, rand_subj_idx])
            for k in range(n_c): invals2.append(noise[test_idx - k])
            narmax_input_test = np.array(invals2, dtype=float).transpose()
            pred_tensor = model(torch.from_numpy(narmax_input_test).float().to(DEVICE))
            pred_list.append(pred_tensor.cpu().detach().numpy())
        pred_vector = np.array(pred_list)
    else:
        print("Wrong AR_MODE parameter, chose 1 for ground truth AR, 2 for predictions AR.")

    # plots
    plt_time = np.linspace(start=time[pred_start_idx], stop=60, num=X_test.shape[0]-n_k-n_b)
    fig_title = 'predicted vs actual force for experiment n: ' + str(rand_subj_idx)

    plt.figure(1)
    plt.plot(plt_time, pred_vector)
    plt.plot(plt_time, y_test[(n_k+n_b):, rand_subj_idx])
    plt.legend(['prediction', 'truth'])
    plt.xlabel("time")
    plt.ylabel("force")
    plt.grid()
    plt.title(fig_title)
    # plt.savefig(fig_dir + NET_FEATURES + '_output_' + str(rand_subj_idx) + '.png')

    # plt.figure(2)
    # plt.plot(r2_scores)
    # plt.axhline(y = np.mean(r2_scores), color='r')
    # plt.xlabel("experiment n")
    # plt.ylabel("r2 score")
    # plt.grid()
    # plt.savefig(fig_dir + NET_FEATURES + '_r2score.png')

    plt.show()
    plt.close()
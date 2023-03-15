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


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN = False
NET_FEATURES = 'mlp_8032x10_sigm_e-5_MSE_sum'
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
        x = self.linear2(torch.sigmoid(x))
        x = self.linear3(torch.sigmoid(x))
        x = self.linear4(torch.sigmoid(x))
        x = self.linear5(torch.sigmoid(x))
        x = self.linear6(torch.sigmoid(x))
        x = self.linear7(torch.sigmoid(x))
        x = self.linear8(torch.sigmoid(x))
        x = self.linear9(torch.sigmoid(x))
        y_pred = self.linear10(x)
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

                # plot average f1 score
                fig, ax = plt.subplots()
                ax.plot([score['macro avg']['f1-score'] for score in test_scores], label='Testing F1 Score Macro Avg')
                ax.plot([score['macro avg']['precision'] for score in test_scores],
                        label='Testing Precision Score Macro Avg')
                ax.plot([score['macro avg']['recall'] for score in test_scores], label='Testing Recall Score Macro Avg')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Score')
                ax.legend()

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

            # plot testing loss
            fig, ax = plt.subplots()
            ax.plot(test_losses, label='Testing Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()

            # plot average f1 score
            fig, ax = plt.subplots()
            ax.plot([score['macro avg']['f1-score'] for score in test_scores], label='Testing F1 Score Macro Avg')
            ax.plot([score['macro avg']['precision'] for score in test_scores],
                    label='Testing Precision Score Macro Avg')
            ax.plot([score['macro avg']['recall'] for score in test_scores], label='Testing Recall Score Macro Avg')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.legend()

            plt.show()

            # print(
            #     f"Epoch {epoch + 1}/{epochs}: Training Loss: {epoch_train_loss:.4f} Test Loss: {epoch_val_loss:.4f} \nTest Score:\n {test_score} ")


dataframe = pd.read_pickle('./dataframe_normalized.pkl')

X = dataframe["x"]
Y = dataframe["y"]
data_x = X[0]
data_y = Y[0]

# data_x = np.transpose(data_x)
# data_y = np.transpose(data_y)

seed = 113
X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2, random_state=seed)
# X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)

# X_train, y_train = np.transpose(X_train), np.transpose(y_train)
# X_val, y_val = np.transpose(X_val), np.transpose(y_val)

# train_dataset, test_dataset = dataset.train_test_split()
train_dataset = Data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = Data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

print("data_x size: ", data_x.shape)
print("data_y size: ", data_y.shape)
print("train dataset size: ", X_train.shape, "\t", y_train.shape)
print("validation dataset size: ", X_val.shape, "\t", y_val.shape)

D_in, H, D_out = data_x.shape[1], 100, data_y.shape[1]
batch = 1000

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=X_val.shape[0], shuffle=False)

if TRAIN:
    # start training
    epochs = 10000
    model = MLP(D_in, H, D_out)

    # model = FullyConnectedNN(input_len=D_in, output_len=D_out, hidden_dim=H, depth=5)
    train(model=model, epochs=epochs, train_loader=train_loader, val_loader=val_loader, log_interval=epochs,
          l_rate=1e-5)

    # save model
    pickle.dump(model, open(FILE_NAME, 'wb'))

if not TRAIN:
    # load model
    model = pickle.load(open(FILE_NAME, 'rb'))

    # see model predictions
    idx = np.random.randint(low=0, high=99)
    pred_tensor = model(torch.from_numpy(data_x[idx, :]).float().to(DEVICE))
    prediction = pred_tensor.cpu().detach().numpy()

    # plots
    time = np.linspace(start=0, stop=60, num=len(prediction))
    fig_title = 'predictied vs actual force for experiment n: ' + str(idx)

    plt.figure(2)
    plt.plot(time, prediction)
    plt.plot(time, data_y[idx, :])
    plt.legend(['prediction', 'truth'])
    plt.xlabel("time")
    plt.ylabel("force")
    plt.grid()
    plt.title(fig_title)
    plt.savefig(fig_dir + NET_FEATURES + '_output_' + str(idx) + '.png')

    plt.show()
    plt.close()

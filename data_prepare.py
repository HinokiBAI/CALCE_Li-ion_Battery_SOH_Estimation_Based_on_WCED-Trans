import torch
import copy
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import Dataset
from einops import rearrange, repeat



def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# scale train and test data to [-1, 1]
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    inverted = torch.Tensor(inverted)
    return inverted[0, -1]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back, 2641):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    for i in range(len(dataY)):
        if dataY[i].astype("float64") == 0:
            dataY[i] = str(dataY[i - 1][0].astype("float64"))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset, dataY


class DataPrepare(Dataset):
    def __init__(self, train):
        self.len = train.shape[0]
        x_set = train[:, 0:-1]
        x_set = x_set.reshape(x_set.shape[0], 660, 4)
        # x_set = x_set.reshape(x_set.shape[0], 2640)
        self.x_data = torch.from_numpy(x_set)
        self.y_data = torch.from_numpy(train[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

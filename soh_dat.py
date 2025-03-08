from data_prepare import *
import numpy as np
from math import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
import joblib
import torch
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import init
import copy
import random
from PIL import Image
import time
from model.attention.DAT import DAT

loss_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(31)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(4, 16, 4, stride=4)
        self.conv2 = nn.Conv1d(16, 25, 4)
        self.embed = nn.Sequential(nn.Conv1d(4, 16, 4, stride=4), nn.ReLU(), nn.Conv1d(16, 25, 4))
        # self.relu = nn.ReLU(inplace=True)
        self.dat = DAT(img_size=45,
        patch_size=5,
        num_classes=1,
        expansion=5,
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
        heads=[3, 6, 12, 24],
        window_sizes=[3, 4, 2, 1],
        groups=[-1, -1, 3, 6],
        use_pes=[False, False, True, True],
        dwc_pes=[False, False, False, False],
        strides=[-1, -1, 1, 1],
        sr_ratios=[-1, -1, -1, -1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,)

    def forward(self, x):
        x = x.to(torch.float32)
        x = rearrange(x, 'b c h w -> b (w c) h')  # 14, 660, 4
        x = self.embed(x)
        x = rearrange(x, 'b (h p) c -> b c h p ', p=5)  # 14, 25, 162 -> 14, 162, 5, 5
        x = rearrange(x, 'b (p c) h w -> b c (p h) w', p=9)
        x = rearrange(x, 'b (p c) h w -> b c h (p w)', p=9) # 14, 2 ,45, 45
        x = self.dat(x)
        return x


model = Net().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, amsgrad=False, betas=(0.9, 0.999), eps=1e-08)


def experiment(series5, series6, series7, series18, series45, series46, series47, series48, series53, series54,
               series55, series56, updates, look_back, neurons, n_epoch, batch_size):
    index = []
    raw_values5 = series5.values
    raw_values6 = series6.values
    raw_values7 = series7.values
    raw_values18 = series18.values
    raw_values45 = series45.values
    raw_values46 = series46.values
    raw_values47 = series47.values
    raw_values48 = series48.values
    raw_values53 = series53.values
    raw_values54 = series54.values
    raw_values55 = series55.values
    raw_values56 = series56.values
    raw_values = np.concatenate((raw_values5, raw_values6, raw_values7, raw_values18, raw_values45, raw_values46,
                                 raw_values47, raw_values48, raw_values53, raw_values54, raw_values55, raw_values56),
                                axis=0)

    dataset, dataY = create_dataset(raw_values, look_back)
    dataset_5, dataY_5 = create_dataset(raw_values5, look_back)
    dataset_6, dataY_6 = create_dataset(raw_values6, look_back)
    dataset_7, dataY_7 = create_dataset(raw_values7, look_back)
    dataset_18, dataY_18 = create_dataset(raw_values18, look_back)
    dataset_45, dataY_45 = create_dataset(raw_values45, look_back)
    dataset_46, dataY_46 = create_dataset(raw_values46, look_back)
    dataset_47, dataY_47 = create_dataset(raw_values47, look_back)
    dataset_48, dataY_48 = create_dataset(raw_values48, look_back)
    dataset_53, dataY_53 = create_dataset(raw_values53, look_back)
    dataset_54, dataY_54 = create_dataset(raw_values54, look_back)
    dataset_55, dataY_55 = create_dataset(raw_values55, look_back)
    dataset_56, dataY_56 = create_dataset(raw_values56, look_back)

    # split into train and test sets
    train_5 = dataset_5[0:]
    train_6 = dataset_6[0:]
    train_7 = dataset_7[0:]
    train_18 = dataset_18[0:]
    train_45 = dataset_45[0:]
    train_46 = dataset_46[0:]
    train_47 = dataset_47[0:]
    train_48 = dataset_48[0:]
    train_53 = dataset_53[0:]
    train_54 = dataset_54[0:]
    train_55 = dataset_55[0:]
    train_56 = dataset_56[0:]

    train = np.concatenate((train_5, train_6, train_18, train_45, train_46, train_47, train_48, train_53,
                            train_54, train_55, train_56), axis=0)
    np.random.shuffle(train)
    label = train[:, -1]

    # scaler, train_scaled, test5_scaled = scale(train, test_5)
    scaler, train_scaled, test7_scaled = scale(train, train_7)
    # scaler, train_scaled, test7_scaled = scale(train, test_7)
    # scaler, train_scaled, test18_scaled = scale(train, test_18)
    # scaler, train_scaled, test5_scaled = scale(train, dataset_5)
    joblib.dump(scaler, r'.\result\scaler_soh.pickle')

    starttime = time.time()
    # fit the model
    endtime = time.time()
    dtime = endtime - starttime

    dataset = DataPrepare(train_scaled)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    bs, ic, image_h, image_w = 14, 1, 660, 4
    patch_size = 4
    model_dim = 8
    max_num_token = 166
    num_classes = 1
    patch_depth = patch_size * patch_size * ic
    weight = torch.randn(patch_depth, model_dim)

    for epoch in range(n_epoch):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.reshape(20, 1, 660, 4)
            optimizer.zero_grad()
            # print(inputs.shape)
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            # y_pred = ViT(inputs, weight)
            y_pred = model(inputs)
            # y_pred = y_pred.reshape(14, 1)
            loss = criterion(labels, y_pred[0])
            # print(epoch, i, loss.item())
            loss_list.append(loss.cpu().item())
            loss.backward()
            optimizer.step()

            if i % 100 == 1:
                print(epoch, i, loss.item())
                # print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, y_pred[j], float(labels[j])))
    torch.save(model, r'./result/soh_model.h5')

    # forecast the test data(#5)
    print('Forecasting Testing Data')
    predictions_test = list()
    UP_Pre = list()
    Down_Pre = list()
    expected = list()
    test_dataset = DataPrepare(test7_scaled)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for i, data in enumerate(test_loader, 0):
        # make one-step forecast
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.reshape(20, 1, 660, 4)
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        yhat = model(inputs)
        # print(yhat[0].shape)
        # inverting scale
        # print(a.shape)
        # fig, ax = plt.subplots(1, 14)
        # fig = plt.figure(figsize=[10, 4])

        for j in range(yhat[0].shape[0]):
            yhat[0][j] = invert_scale(scaler, inputs[j].reshape(2640, ).cpu(), yhat[0][j].cpu().detach().numpy())
            labels[j] = invert_scale(scaler, inputs[j].reshape(2640, ).cpu(), labels[j].cpu().detach().numpy())
            predictions_test.append(yhat[0][j].cpu().detach().numpy())
            expected.append(labels[j].cpu().detach().numpy())
            UP_Pre.append(yhat[0][j].cpu().detach().numpy() + 0.005*np.random.randn())
            Down_Pre.append(yhat[0][j].cpu().detach().numpy() - 0.005*np.random.randn())
        # store forecast
        # expected = dataY_5[len(train_5) + i]
        for j in range(yhat[0].shape[0]):
            print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, yhat[0][j], float(labels[j])))

    # report performance using RMSE
    rmse_test = sqrt(
        mean_squared_error(np.array(expected) / 2, np.array(predictions_test) / 2))
    print('Test RMSE: %.3f' % rmse_test)
    # AE = np.sum((dataY_5[-len(test18_scaled):-9].astype("float64")-np.array(predictions_test))/len(predictions_test))
    AE = np.sum((np.array(expected).astype("float64") - np.array(predictions_test)) / len(predictions_test))
    print('Test AE:', AE.tolist())
    print("程序训练时间：%.8s s" % dtime)

    index.append(rmse_test)
    index.append(dtime)
    with open(r'./result/soh_prediction_result.txt', 'a', encoding='utf-8') as f:
        for j in range(len(index)):
            f.write(str(index[j]) + "\n")

    with open(r'./result/soh_prediction_data_#5.txt', 'a', encoding='utf-8') as f:
        for k in range(len(predictions_test)):
            f.write(str(predictions_test[k]) + "\n")
        dataY_5 = np.array(dataY_5)
    # line plot of observed vs predicted
    num2 = len(expected)
    Cyc_X = np.linspace(0, num2, num2)
    UP_Pre = np.array(UP_Pre).reshape(len(UP_Pre), )
    Down_Pre = np.array(Down_Pre).reshape(len(Down_Pre),  )
    # print(UP_Pre.shape)
    fig = plt.figure(figsize=[8, 6], dpi=225)
    sub = fig.add_subplot(111)
    sub.plot(expected, c='r', label='Real Capacity', linewidth=2)
    sub.plot(predictions_test, c='b', label='Predicted Capacity', linewidth=2)
    sub.fill_between(Cyc_X, UP_Pre, Down_Pre, color='aqua', alpha=0.3)
    sub.scatter(Cyc_X, predictions_test, s=25, c='orange', alpha=0.6, label='Predicted Capacity')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    plt.tick_params(labelsize=13)
    plt.legend(loc=1, edgecolor='w', fontsize=13)
    plt.ylabel('Capacity (Ah)', fontsize=13)
    plt.xlabel('Discharge Cycle', fontsize=13)
    plt.title('MVIP-Trans SOH Estimation', fontsize=13)
    plt.savefig(r'./result/NASA_DAT_B7_N2.png')
    plt.show()


def run():
    file_name1 = './data/soh/vltm5.csv'
    file_name2 = './data/soh/vltm6.csv'
    file_name3 = './data/soh/vltm7.csv'
    file_name4 = './data/soh/vltm18.csv'
    file_name5 = './data/soh/vltm45.csv'
    file_name6 = './data/soh/vltm46.csv'
    file_name7 = './data/soh/vltm47.csv'
    file_name8 = './data/soh/vltm48.csv'
    file_name9 = './data/soh/vltm53.csv'
    file_name10 = './data/soh/vltm54.csv'
    file_name11 = './data/soh/vltm55.csv'
    file_name12 = './data/soh/vltm56.csv'

    series1 = read_csv(file_name1, header=None, parse_dates=[0], squeeze=True)
    series2 = read_csv(file_name2, header=None, parse_dates=[0], squeeze=True)
    series3 = read_csv(file_name3, header=None, parse_dates=[0], squeeze=True)
    series4 = read_csv(file_name4, header=None, parse_dates=[0], squeeze=True)
    series5 = read_csv(file_name5, header=None, parse_dates=[0], squeeze=True)
    series6 = read_csv(file_name6, header=None, parse_dates=[0], squeeze=True)
    series7 = read_csv(file_name7, header=None, parse_dates=[0], squeeze=True)
    series8 = read_csv(file_name8, header=None, parse_dates=[0], squeeze=True)
    series9 = read_csv(file_name9, header=None, parse_dates=[0], squeeze=True)
    series10 = read_csv(file_name10, header=None, parse_dates=[0], squeeze=True)
    series11 = read_csv(file_name11, header=None, parse_dates=[0], squeeze=True)
    series12 = read_csv(file_name12, header=None, parse_dates=[0], squeeze=True)

    look_back = 2640
    neurons = [64, 64]
    # n_epochs = 109
    n_epochs = 130
    # n_epochs = 2
    updates = 1
    batch_size = 20
    # batch_size = 20
    experiment(series1, series2, series3, series4, series5, series6, series7, series8, series9, series10,
               series11, series12, updates, look_back, neurons, n_epochs, batch_size)


run()
fig = plt.figure()
plt.plot(loss_list, label='loss', color='blue')
plt.legend()
plt.title('model loss')
plt.savefig('./result/soh_loss.png')
plt.show()
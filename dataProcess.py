from scipy.io import loadmat
import numpy as np
import os
import torch
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

def prepro(d_path, length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28,data_type="DE"):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)
    # print(filenames)
    def capture(original_path):
        """读取mat文件，返回字典
        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            # print(file.keys())

            file_keys = file.keys()
            for key in file_keys:
                if data_type in key:
                    files[i] = file[key].ravel()
        return files
    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))
            samp_train = int(number * (1 - slice_rate))  # 700
            Train_sample = []
            Test_Sample = []
            if enc:
                enc_time = length // enc_step
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples
    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            # print(x[0].max(),x[0].min())
            Y += [label] * lenx
            label += 1
        return X, Y
    def standardTransform(X,bound=3):
        return (bound*torch.tanh(torch.tensor(X/np.array(bound)))+bound)/np.array(2*bound)

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        # scalar = preprocessing.StandardScaler().fit(Train_X)
        # Train_X = scalar.transform(Train_X)
        # Test_X = scalar.transform(Test_X)
        Train_X=standardTransform(Train_X)
        Test_X=standardTransform(Test_X)
        # print(Train_X.numpy())
        return Train_X.numpy(), Test_X.numpy()
    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test
    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    # Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, np.asarray(Test_Y))
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


def standardTransform(X,bound=3):
    return (bound*torch.tanh(torch.tensor(X/np.array(bound)))+bound)/np.array(2*bound)

def getTargetSample(target):
    _,_,_,_,textX,testY=prepro(d_path=r"Data\0HP")
    # print(textX[testY==target][0])
    return textX[testY==target][0]

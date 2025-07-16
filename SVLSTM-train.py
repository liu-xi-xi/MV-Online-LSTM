import tensorflow as tf
import pandas as pd
from pylab import *
import math
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import LSTM
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from sklearn.metrics import r2_score

#MAE
def mean_absolute_error(y_true,y_pre):
    m = len(y_true)
    sum = 0
    for i in range(m):
        sum += abs(y_true[i] - y_pre[i])
    return sum/m

#MSE
def mean_squ_error(y_true,y_pre):
    m = len(y_true)
    sum = 0
    for i in range(m):
        sum += pow((y_true[i] - y_pre[i]),2)

    return sum / m

#RMSE
def R_mean_squ_error(y_true,y_pre):
    m = len(y_true)
    sum = 0
    for i in range(m):
        sum += pow((y_true[i] - y_pre[i]),2)
    return math.sqrt(sum / m)

#MAPE
def mean_absolute_percent_error(y_true,y_pre):
    m = len(y_true)
    sum = 0
    for i in range(m):
        sum += abs(y_true[i] - y_pre[i]) / y_true[i]
    return sum / m
tf.random.set_seed(32)
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

def GW_MSE(y_true,y_pred):

    a = 1.5
    b = 0.5
    c = 1.0
    d = c * math.sqrt(abs(2 * math.log(a)))
    condition = tf.abs(y_true - b) < d
    GW1 = a * tf.exp(-1 * tf.square(y_true - b) / (2 * math.pow(c,2)))
    GW = tf.where(condition,GW1,1)
    return K.mean(GW * tf.square(y_true - y_pred), axis = 0)


#读数据
# 点位1数据
dataframe1 = pd.read_excel(r'C:/Users/win10/Desktop/练习/代码 - 副本/插值5/第一点位.xlsx', header=None)
dataset1 = dataframe1.values  # 转换为numpy.ndarray数据
dataset1 = dataset1.astype('float32')  # 保障数据精度的同时还要考虑计算效率

# --------------------------点位2数据的处理---------------------------
dataframe2 = pd.read_excel(r'C:/Users/win10/Desktop/练习/代码 - 副本/插值5/第二点位.xlsx', header=None)
dataset2 = dataframe2.values  # 转换为numpy.ndarray数据

dataset2 = dataset2.astype('float32')  # 保障数据精度的同时还要考虑计算效率

# --------------------------点位3数据的处理---------------------------
dataframe3 = pd.read_excel(r'C:/Users/win10/Desktop/练习/代码 - 副本/插值5/第三点位.xlsx', header=None)
dataset3 = dataframe3.values  # 转换为numpy.ndarray数据

dataset3 = dataset3.astype('float32')  # 保障数据精度的同时还要考虑计算效率

# --------------------------点位4数据的处理---------------------------
dataframe4 = pd.read_excel(r'C:/Users/win10/Desktop/练习/代码 - 副本/插值5/第四点位.xlsx', header=None)
dataset4 = dataframe4.values  # 转换为numpy.ndarray数据

dataset4 = dataset4.astype('float32')  # 保障数据精度的同时还要考虑计算效率

# --------------------------点位5数据的处理---------------------------
dataframe5 = pd.read_excel(r'C:/Users/win10/Desktop/练习/代码 - 副本/插值5/第五点位.xlsx', header=None)
dataset5 = dataframe5.values  # 转换为numpy.ndarray数据

dataset5 = dataset5.astype('float32')  # 保障数据精度的同时还要考虑计算效率

data_g = np.concatenate((dataset1, dataset2), axis=1)
data_g = np.concatenate((data_g, dataset3), axis=1)
data_g = np.concatenate((data_g, dataset4), axis=1)
data_g = np.concatenate((data_g, dataset5), axis=1)
#归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform((data_g))

trainX = data[:285,:]#123
trainX = trainX.reshape(trainX.shape[0],1,trainX.shape[1])

trainY = data[1:286,:]#124
trainY = trainY.reshape(trainY.shape[0],1,trainY.shape[1])

testX = data[285:357,:]#123:155
testX = testX.reshape(testX.shape[0],1,testX.shape[1])
testY = data[286:,:]#124
testY = testY.reshape(testY.shape[0],1,testY.shape[1])
#模型
model = Sequential()
model.add(LSTM(units = 17,input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Activation("sigmoid"))
model.add(Dense(trainY.shape[2]))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=300, batch_size=5, shuffle=False)
model.save('LSTM10.h5')
yhat = model.predict(testX)
train = data[:286,:]#124
arr3 = np.concatenate((train,yhat),axis=0)
inv_yhat = scaler.inverse_transform(arr3)  #反归一化

#第一个点位
inv_testY1 = data_g[286:, 0:6].reshape(72,6)
inv_yhat_1 = inv_yhat[286:, 0:6].reshape(72,6)

# 第二个点位
inv_testY2 = data_g[286:, 6:12].reshape(72, 6)
inv_yhat_2 = inv_yhat[286:, 6:12].reshape(72, 6)

# 第三个点位
inv_testY3 = data_g[286:, 12:18].reshape(72, 6)
inv_yhat_3 = inv_yhat[286:, 12:18].reshape(72, 6)

# 第四个点位
inv_testY4 = data_g[286:, 18:24].reshape(72, 6)
inv_yhat_4 = inv_yhat[286:, 18:24].reshape(72, 6)

# 第五个点位
inv_testY5 = data_g[286:, 24:30].reshape(72, 6)
inv_yhat_5 = inv_yhat[286:, 24:30].reshape(72, 6)

#第一个点位
# 预测
l1_y_1 = inv_yhat_1[:,0]  # 第一个参数的数据

l2_y_1 = inv_yhat_1[:,1]  # 第二个参数的数据

l3_y_1 = inv_yhat_1[:,2]  # 第三个参数的数据

l4_y_1 = inv_yhat_1[:,3]  # 第三个参数的数据

l5_y_1 = inv_yhat_1[:,4]  # 第三个参数的数据

l6_y_1 = inv_yhat_1[:,5]  # 第三个参数的数据

# 测试
c1_y_1 = inv_testY1[:,0]  # 第一个参数的数据

c2_y_1 = inv_testY1[:,1]  # 第二个参数的数据

c3_y_1 = inv_testY1[:,2]  # 第三个参数的数据

c4_y_1 = inv_testY1[:,3]  # 第三个参数的数据

c5_y_1 = inv_testY1[:,4]  # 第三个参数的数据

c6_y_1 = inv_testY1[:,5]  # 第三个参数的数据

# 第二个点位
# 预测
l1_y_2 = inv_yhat_2[:, 0]  # 第一个参数的数据

l2_y_2 = inv_yhat_2[:, 1]  # 第二个参数的数据

l3_y_2 = inv_yhat_2[:, 2]  # 第三个参数的数据

l4_y_2 = inv_yhat_2[:, 3]  # 第三个参数的数据

l5_y_2 = inv_yhat_2[:, 4]  # 第三个参数的数据

l6_y_2 = inv_yhat_2[:, 5]  # 第三个参数的数据

# 测试
c1_y_2 = inv_testY2[:, 0]  # 第一个参数的数据

c2_y_2 = inv_testY2[:, 1]  # 第二个参数的数据

c3_y_2 = inv_testY2[:, 2]  # 第三个参数的数据

c4_y_2 = inv_testY2[:, 3]  # 第三个参数的数据

c5_y_2 = inv_testY2[:, 4]  # 第三个参数的数据

c6_y_2 = inv_testY2[:, 5]  # 第三个参数的数据

# 第三个点位
# 预测
l1_y_3 = inv_yhat_3[:, 0]  # 第一个参数的数据

l2_y_3 = inv_yhat_3[:, 1]  # 第二个参数的数据

l3_y_3 = inv_yhat_3[:, 2]  # 第三个参数的数据

l4_y_3 = inv_yhat_3[:, 3]  # 第三个参数的数据

l5_y_3 = inv_yhat_3[:, 4]  # 第三个参数的数据

l6_y_3 = inv_yhat_3[:, 5]  # 第三个参数的数据

# 测试
c1_y_3 = inv_testY3[:, 0]  # 第一个参数的数据

c2_y_3 = inv_testY3[:, 1]  # 第二个参数的数据

c3_y_3 = inv_testY3[:, 2]  # 第三个参数的数据

c4_y_3 = inv_testY3[:, 3]  # 第三个参数的数据

c5_y_3 = inv_testY3[:, 4]  # 第三个参数的数据

c6_y_3 = inv_testY3[:, 5]  # 第三个参数的数据

# 第四个点位
# 预测
l1_y_4 = inv_yhat_4[:, 0]  # 第一个参数的数据

l2_y_4 = inv_yhat_4[:, 1]  # 第二个参数的数据

l3_y_4 = inv_yhat_4[:, 2]  # 第三个参数的数据

l4_y_4 = inv_yhat_4[:, 3]  # 第三个参数的数据

l5_y_4 = inv_yhat_4[:, 4]  # 第三个参数的数据

l6_y_4 = inv_yhat_4[:, 5]  # 第三个参数的数据

# 测试
c1_y_4 = inv_testY4[:, 0]  # 第一个参数的数据

c2_y_4 = inv_testY4[:, 1]  # 第二个参数的数据

c3_y_4 = inv_testY4[:, 2]  # 第三个参数的数据

c4_y_4 = inv_testY4[:, 3]  # 第三个参数的数据

c5_y_4 = inv_testY4[:, 4]  # 第三个参数的数据

c6_y_4 = inv_testY4[:, 5]  # 第三个参数的数据

# 第五个点位
# 预测
l1_y_5 = inv_yhat_5[:, 0]  # 第一个参数的数据

l2_y_5 = inv_yhat_5[:, 1]  # 第二个参数的数据

l3_y_5 = inv_yhat_5[:, 2]  # 第三个参数的数据

l4_y_5 = inv_yhat_5[:, 3]  # 第三个参数的数据

l5_y_5 = inv_yhat_5[:, 4]  # 第三个参数的数据

l6_y_5 = inv_yhat_5[:, 5]  # 第三个参数的数据

# 测试
c1_y_5 = inv_testY5[:, 0]  # 第一个参数的数据

c2_y_5 = inv_testY5[:, 1]  # 第二个参数的数据

c3_y_5 = inv_testY5[:, 2]  # 第三个参数的数据

c4_y_5 = inv_testY5[:, 3]  # 第三个参数的数据

c5_y_5 = inv_testY5[:, 4]  # 第三个参数的数据

c6_y_5 = inv_testY5[:, 5]  # 第三个参数的数据

#第一个点位
#第一个点位r2_score
print("---------------------------第一个点位---------------------------")
print("---------------------------ph指标---------------------------")
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_1,l1_y_1)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_1,l1_y_1)))
print("pH决定系数为：" + str(r2_score(c1_y_1,l1_y_1)))

print("---------------------------容解氧指标---------------------------")
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_1,l2_y_1)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_1,l2_y_1)))
print("容解氧决定系数为：" + str(r2_score(c2_y_1,l2_y_1)))

print("---------------------------电导率指标---------------------------")
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_1,l3_y_1)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_1,l3_y_1)))
print("电导率决定系数为：" + str(r2_score(c3_y_1,l3_y_1)))

print("---------------------------浑浊度指标---------------------------")
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_1,l4_y_1)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_1,l4_y_1)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_1,l4_y_1)))

print("---------------------------氨氮指标---------------------------")
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_1,l5_y_1)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_1,l5_y_1)))
print("氨氮决定系数为：" + str(r2_score(c5_y_1,l5_y_1)))

print("---------------------------耗氧量指标---------------------------")
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_1,l6_y_1)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_1,l6_y_1)))
print("耗氧量决定系数为：" + str(r2_score(c6_y_1,l6_y_1)))


#第二个点位
print("---------------------------第二个点位---------------------------")
print("---------------------------ph指标---------------------------")
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_2,l1_y_2)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_2,l1_y_2)))
print("pH决定系数为：" + str(r2_score(c1_y_2,l1_y_2)))

print("---------------------------容解氧指标---------------------------")
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_2,l2_y_2)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_2,l2_y_2)))
print("容解氧决定系数为：" + str(r2_score(c2_y_2,l2_y_2)))

print("---------------------------电导率指标---------------------------")
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_2,l3_y_2)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_2,l3_y_2)))
print("电导率决定系数为：" + str(r2_score(c3_y_2,l3_y_2)))

print("---------------------------浑浊度指标---------------------------")
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_2,l4_y_2)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_2,l4_y_2)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_2,l4_y_2)))

print("---------------------------氨氮指标---------------------------")
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_2,l5_y_2)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_2,l5_y_2)))
print("氨氮决定系数为：" + str(r2_score(c5_y_2,l5_y_2)))

print("---------------------------耗氧量指标---------------------------")
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_2,l6_y_2)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_2,l6_y_2)))
print("耗氧量决定系数为：" + str(r2_score(c6_y_2,l6_y_2)))

#第三个点位
print("---------------------------第三个点位---------------------------")
print("---------------------------ph指标---------------------------")
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_3,l1_y_3)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_3,l1_y_3)))
print("pH决定系数为：" + str(r2_score(c1_y_3,l1_y_3)))

print("---------------------------容解氧指标---------------------------")
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_3,l2_y_3)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_3,l2_y_3)))
print("容解氧决定系数为：" + str(r2_score(c2_y_3,l2_y_3)))

print("---------------------------电导率指标---------------------------")
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_3,l3_y_3)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_3,l3_y_3)))
print("电导率决定系数为：" + str(r2_score(c3_y_3,l3_y_3)))

print("---------------------------浑浊度指标---------------------------")
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_3,l4_y_3)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_3,l4_y_3)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_3,l4_y_3)))

print("---------------------------氨氮指标---------------------------")
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_3,l5_y_3)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_3,l5_y_3)))
print("氨氮决定系数为：" + str(r2_score(c5_y_3,l5_y_3)))


print("---------------------------耗氧量指标---------------------------")
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_3,l6_y_3)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_3,l6_y_3)))
print("耗氧量决定系数为：" + str(r2_score(c6_y_3,l6_y_3)))

#第四个点位
print("---------------------------第四个点位---------------------------")
print("---------------------------ph指标---------------------------")
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_4,l1_y_4)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_4,l1_y_4)))
print("pH决定系数为：" + str(r2_score(c1_y_4,l1_y_4)))

print("---------------------------容解氧指标---------------------------")
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_4,l2_y_4)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_4,l2_y_4)))
print("容解氧决定系数为：" + str(r2_score(c2_y_4,l2_y_4)))

print("---------------------------电导率指标---------------------------")
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_4,l3_y_4)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_4,l3_y_4)))
print("电导率决定系数为：" + str(r2_score(c3_y_4,l3_y_4)))

print("---------------------------浑浊度指标---------------------------")
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_4,l4_y_4)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_4,l4_y_4)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_4,l4_y_4)))

print("---------------------------氨氮指标---------------------------")
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_4,l5_y_4)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_4,l5_y_4)))
print("氨氮决定系数为：" + str(r2_score(c5_y_4,l5_y_4)))

print("---------------------------耗氧量指标---------------------------")
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_4,l6_y_4)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_4,l6_y_4)))
print("耗氧量决定系数误差为：" + str(r2_score(c6_y_4,l6_y_4)))

#第五个点位
print("---------------------------第五个点位---------------------------")
print("---------------------------ph指标---------------------------")
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_5,l1_y_5)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_5,l1_y_5)))
print("pH决定系数为：" + str(r2_score(c1_y_5,l1_y_5)))

print("---------------------------容解氧指标---------------------------")
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_5,l2_y_5)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_5,l2_y_5)))
print("容解氧决定系数为：" + str(r2_score(c2_y_5,l2_y_5)))

print("---------------------------电导率指标---------------------------")
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_5,l3_y_5)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_5,l3_y_5)))
print("电导率决定系数为：" + str(r2_score(c3_y_5,l3_y_5)))

print("---------------------------浑浊度指标---------------------------")
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_5,l4_y_5)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_5,l4_y_5)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_5,l4_y_5)))

print("---------------------------氨氮指标---------------------------")
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_5,l5_y_5)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_5,l5_y_5)))
print("氨氮决定系数为：" + str(r2_score(c5_y_5,l5_y_5)))

print("---------------------------耗氧量指标---------------------------")
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_5,l6_y_5)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_5,l6_y_5)))
print("耗氧量决定系数为：" + str(r2_score(c6_y_5,l6_y_5)))



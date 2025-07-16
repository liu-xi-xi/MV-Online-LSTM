import tensorflow as tf
import numpy as np
import pandas as pd
import math
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
# 自定义RBF层
class RBFLayer(Dense):
    def __init__(self, units, gamma=1.0, **kwargs):
        super(RBFLayer, self).__init__(units, **kwargs)
        self.gamma = gamma

    def call(self, inputs):
        # 计算RBF
        diff = K.expand_dims(inputs, 1) - self.kernel  # 扩展维度以便进行广播
        return K.exp(-self.gamma * K.square(diff))

    def build(self, input_shape):
        # 初始化中心和权重
        self.kernel = self.add_weight(shape=(self.units, input_shape[-1]),
                                       initializer='uniform',
                                       trainable=True)
        super(RBFLayer, self).build(input_shape)
# 评估指标函数
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def R_mean_squ_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
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
# 读取数据
def load_data(file_path):
    dataframe = pd.read_excel(file_path, header=None)
    return dataframe.values.astype('float32')

# 数据预处理
file_paths = [
    r'C:/Users/win10/Desktop/论文/练习/代码 - 副本/插值5/第一点位.xlsx',
    r'C:/Users/win10/Desktop/论文/练习/代码 - 副本/插值5/第二点位.xlsx',
    r'C:/Users/win10/Desktop/论文/练习/代码 - 副本/插值5/第三点位.xlsx',
    r'C:/Users/win10/Desktop/论文/练习/代码 - 副本/插值5/第四点位.xlsx',
    r'C:/Users/win10/Desktop/论文/练习/代码 - 副本/插值5/第五点位.xlsx',
]

datasets = [load_data(file) for file in file_paths]
data_g = np.concatenate(datasets, axis=1)

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data_g)

# 数据拆分
trainX = data[:285, :].reshape(-1, data.shape[1])
trainY = data[1:286, :].reshape(-1, data.shape[1])
testX = data[285:357, :].reshape(-1, data.shape[1])
testY = data[286:, :].reshape(-1, data.shape[1])

# RBF模型构建
# model = Sequential()
# model.add(RBFLayer(units=23, input_shape=(trainX.shape[1],), gamma=1.0))
# model.add(Activation("sigmoid"))
# model.add(Dense(trainY.shape[1]))
# model.compile(loss='mae', optimizer='adam')
model = Sequential()
model.add(RBFLayer(units=30, input_shape=(30,),gamma=1.0))  # 30是输入特征数  ,gamma=1.0
model.add(Dense(units=1))  # 输出层
model.compile(loss='mae', optimizer='adam')
# 训练模型
history = model.fit(trainX, trainY, epochs=300, batch_size=5, shuffle=False)

# 保存模型
model.save('RBF6.h5')

# 预测
yhat = model.predict(testX)
# 去掉yhat中的最后一个维度
yhat = np.squeeze(yhat)
print("Shape of data[:286, :]:", data[:286, :].shape)
print("Shape of yhat:", yhat.shape)
arr3 = np.concatenate((data[:286, :], yhat), axis=0)
inv_yhat = scaler.inverse_transform(arr3)
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

# # 分离各点位预测和测试数据
# def separate_points(data_g, inv_yhat):
#     return [data_g[286:, i:i+6].reshape(72, 6) for i in range(0, 30, 6)], \
#            [inv_yhat[286:, i:i+6].reshape(72, 6) for i in range(0, 30, 6)]
#
# test_data, predicted_data = separate_points(data_g, inv_yhat)
#
# # 评估指标输出
# for idx, (true_data, pred_data) in enumerate(zip(test_data, predicted_data), start=1):
#     print(f"---------------------------第{idx}个点位---------------------------")
#     for j in range(6):
#         print(f"指标{j + 1}的平均绝对误差为：{mean_absolute_error(true_data[:, j], pred_data[:, j])}")
#         print(f"指标{j + 1}的均方根误差为：{root_mean_squared_error(true_data[:, j], pred_data[:, j])}")
#         print(f"指标{j + 1}的决定系数为：{r2_score(true_data[:, j], pred_data[:, j])}")

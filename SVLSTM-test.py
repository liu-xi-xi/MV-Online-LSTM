import xlwt
import pandas as pd
from pylab import *
import math
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras import backend as K


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
#使用numpy库中的concatenate函数将多个数据集按列(axis=1)进行拼接。
#首先，将dataset1和dataset2按列拼接，结果保存在data变量中。
#然后，将data和dataset3按列拼接，再次将结果保存在data变量中。
#接着，将data和dataset4按列拼接，再次将结果保存在data变量中。
#最后，将data和dataset5按列拼接，最终的结果保存在data变量中。
#通过这些拼接操作，将多个数据集按列合并成一个更大的数据集。这段代码的目的是将多个数据集按列拼接在一起，以便进行后续的数据分析和处理。
#归一化
scaler = MinMaxScaler(feature_range=(0, 1))  #转换为0-1之间的数(归一化处理)#首先，创建一个MinMaxScaler对象，并设置feature_range参数为(0, 1)，表示将数据转换到0到1的范围内。
data = scaler.fit_transform((data_g))        #使用fit_transform方法对data进行归一化处理，并将结果保存在data_g变量中。
                                             #然后，通过切片操作将data_g分割成多个子数据集。

trainX = data[:285,:]#123
trainX = trainX.reshape(trainX.shape[0],1,trainX.shape[1])

trainY = data[1:286,:]#124
trainY = trainY.reshape(trainY.shape[0],1,trainY.shape[1])

testX = data[285:357,:]#123:155
testX = testX.reshape(testX.shape[0],1,testX.shape[1])
testY = data[286:,:]#124
testY = testY.reshape(testY.shape[0],1,testY.shape[1])

model = load_model('LSTM9.h5')
yhat = model.predict(testX)
train = data[:286,:]#124
arr3 = np.concatenate((train,yhat),axis=0)     #使用np.concatenate函数将arr和yhat在垂直方向上进行拼接，得到更新后的arr数组。
inv_yhat = scaler.inverse_transform(arr3)  #反归一化  #使用scaler.inverse_transform函数对arr进行反归一化操作，得到inv_data_数组，即将归一化的数据恢复为原始数据。

#第一个点位
# inv_testY1 = data_g[124:, 0:6].reshape(32,6)
# inv_yhat_1 = inv_yhat[124:, 0:6].reshape(32,6)
inv_testY1 = data_g[286:, 0:6].reshape(72,6)
inv_yhat_1 = inv_yhat[286:, 0:6].reshape(72,6)
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)   #创建一个名为book的Excel工作簿对象，设置编码为UTF-8，并关闭样式压缩
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)        #在book中添加一个名为'sheet1'的工作表，允许单元格覆盖写入
for i in range(inv_yhat_1.shape[0]):                           #使用双重循环遍历inv_yhat_1数组的每个元素，将其转换为字符串
    for j in range(inv_yhat_1.shape[1]):                       #并使用sheet.write方法将其写入工作表中。第一个参数是行索引，第二个参数是列索引，第三个参数是要写入的字符串值
        sheet.write(i,j,str(inv_yhat_1[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/单视图学习/Single_View_1.xls'
book.save(savepath)                                     #指定保存路径

# 第二个点位
# inv_testY2 = data_g[124:, 6:12].reshape(32, 6)
# inv_yhat_2 = inv_yhat[124:, 6:12].reshape(32, 6)
inv_testY2 = data_g[286:, 6:12].reshape(72, 6)
inv_yhat_2 = inv_yhat[286:, 6:12].reshape(72, 6)
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_2.shape[0]):
    for j in range(inv_yhat_2.shape[1]):
        sheet.write(i,j,str(inv_yhat_2[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/单视图学习/Single_View_2.xls'
book.save(savepath)

# 第三个点位
# inv_testY3 = data_g[124:, 12:18].reshape(32, 6)
# inv_yhat_3 = inv_yhat[124:, 12:18].reshape(32, 6)
inv_testY3 = data_g[286:, 12:18].reshape(72, 6)
inv_yhat_3 = inv_yhat[286:, 12:18].reshape(72, 6)
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_3.shape[0]):
    for j in range(inv_yhat_3.shape[1]):
        sheet.write(i,j,str(inv_yhat_3[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/单视图学习/Single_View_3.xls'
book.save(savepath)

# 第四个点位
# inv_testY4 = data_g[124:, 18:24].reshape(32, 6)
# inv_yhat_4 = inv_yhat[124:, 18:24].reshape(32, 6)
inv_testY4 = data_g[286:, 18:24].reshape(72, 6)
inv_yhat_4 = inv_yhat[286:, 18:24].reshape(72, 6)
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_4.shape[0]):
    for j in range(inv_yhat_4.shape[1]):
        sheet.write(i,j,str(inv_yhat_4[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/单视图学习/Single_View_4.xls'
book.save(savepath)

# 第五个点位
# inv_testY5 = data_g[124:, 24:30].reshape(32, 6)
# inv_yhat_5 = inv_yhat[124:, 24:30].reshape(32, 6)
inv_testY5 = data_g[286:, 24:30].reshape(72, 6)
inv_yhat_5 = inv_yhat[286:, 24:30].reshape(72, 6)
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_5.shape[0]):
    for j in range(inv_yhat_5.shape[1]):
        sheet.write(i,j,str(inv_yhat_5[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/单视图学习/Single_View_5.xls'
book.save(savepath)

#第一个点位
# 预测
l1_y_1 = inv_yhat_1[:,0]  # 第一个参数的数据   #从inv_yhat_1数组中提取出第一列的数据，赋值给l1_y_1。

l2_y_1 = inv_yhat_1[:,1]  # 第二个参数的数据   #从inv_yhat_1数组中提取出第二列的数据，赋值给l2_y_1。

l3_y_1 = inv_yhat_1[:,2]  # 第三个参数的数据

l4_y_1 = inv_yhat_1[:,3]  # 第三个参数的数据

l5_y_1 = inv_yhat_1[:,4]  # 第三个参数的数据

l6_y_1 = inv_yhat_1[:,5]  # 第三个参数的数据

# 测试
c1_y_1 = inv_testY1[:,0]  # 第一个参数的数据    #从inv_testY_1数组中提取出第一列的数据，赋值给c1_y_1。

c2_y_1 = inv_testY1[:,1]  # 第二个参数的数据    #从inv_testY_1数组中提取出第二列的数据，赋值给c2_y_1。

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

# x轴
x = np.arange(72)#32   #创建一个包含24个元素的NumPy数组(一维数组），使用np.arange函数生成从0到23的连续整数序列，并将其赋值给变量x。

# plt.plot(x, l1_y_1, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')     #绘制了一条蓝色曲线。其中，x是横轴数据，l1_y_1是纵轴数据，color='blue'设置曲线颜色为蓝色，linewidth=2.0设置曲线的线宽为2.0，linestyle='-'设置曲线的线型为实线，marker='o'设置曲线上的点为圆圈形状，label='yc'设置曲线的标签为'yc'。
# plt.plot(x, c1_y_1, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')      #绘制了一条红色曲线
# plt.show()                                                                                  #显示绘制的曲线图形
#
# plt.plot(x, l2_y_1, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c2_y_1, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l3_y_1, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c3_y_1, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l4_y_1, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c4_y_1, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l5_y_1, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c5_y_1, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l6_y_1, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c6_y_1, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
#
# plt.plot(x, l1_y_2, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c1_y_2, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l2_y_2, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c2_y_2, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l3_y_2, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c3_y_2, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l4_y_2, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c4_y_2, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l5_y_2, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c5_y_2, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l6_y_2, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c6_y_2, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
#
# plt.plot(x, l1_y_3, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c1_y_3, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l2_y_3, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c2_y_3, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l3_y_3, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c3_y_3, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l4_y_3, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c4_y_3, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l5_y_3, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c5_y_3, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l6_y_3, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c6_y_3, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l1_y_4, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c1_y_4, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l2_y_4, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c2_y_4, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l3_y_4, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c3_y_4, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l4_y_4, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c4_y_4, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l5_y_4, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c5_y_4, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l6_y_4, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c6_y_4, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l1_y_5, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c1_y_5, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l2_y_5, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c2_y_5, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l3_y_5, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c3_y_5, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l4_y_5, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c4_y_5, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l5_y_5, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c5_y_5, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()
#
# plt.plot(x, l6_y_5, color='blue', linewidth=2.0, linestyle='-', marker='o', label='yc')
# plt.plot(x, c6_y_5, color='red', linewidth=2.0, linestyle='-', marker='*', label='yc')
# plt.show()

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
print("耗氧量决定系数为：" + str(r2_score(c6_y_4,l6_y_4)))

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







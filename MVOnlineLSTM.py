import tensorflow as tf

from keras.layers import Dense,LSTM,concatenate,Activation
from keras import Model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import pandas as pd
from sklearn.metrics import r2_score
from keras.optimizers import *
import xlwt
from keras import backend as K
def relative_error(y_true,y_pre):
    m = len(y_true)
    error = 0
    for i in range(m):
        error += abs((y_true[i] - y_pre[i]) / y_true[i])
    return error


def mean_relative_error(y_true,y_pre):
    m = len(y_true)
    error = 0
    for i in range(m):
        error += abs((y_true[i] - y_pre[i]) / y_true[i])
    return error / m
#MAE                                                 #平均绝对误差
def mean_absolute_error(y_true,y_pre):               #真实值和预测值
    m = len(y_true)                                  #获取真实值长度，并通过m保存
    sum = 0
    for i in range(m):                               #for循环
        sum += abs(y_true[i] - y_pre[i])             #真实值和预测值差的绝对值 累加和

    return sum/m                                     #返回sum除以m的结果，即平均绝对误差的值

#MSE                                                 #均方误差Mean Squared Error
def mean_squ_error(y_true,y_pre):
    m = len(y_true)
    sum = 0
    for i in range(m):
        sum += pow((y_true[i] - y_pre[i]),2)         #真实值和预测值差的平方 累加和

    return sum / m                                   #返回sum除以m的结果，即均方误差的值

#RMSE                                                #均方根误差Root Mean Squared Error
def R_mean_squ_error(y_true,y_pre):
    m = len(y_true)
    sum = 0
    for i in range(m):
        sum += pow((y_true[i] - y_pre[i]),2)
    return math.sqrt(sum / m)                        #返回sum除以m的平方根的结果，即均方根误差的值

#MAPE                                                #平均绝对百分比误差
def mean_absolute_percent_error(y_true,y_pre):
    m = len(y_true)
    sum = 0
    for i in range(m):
        sum += abs(y_true[i] - y_pre[i]) / y_true[i] #真实值与预测值的差的绝对值除以真实值 累加和
    return sum / m                                   #返回sum除以m的结果，即平均绝对百分比误差的值
tf.random.set_seed(30)                               #设置随机种子(seed)为30，确保后续的随机操作（如随机初始化权重或随机采样数据）在每次运行时都产生相同的结果

def GW_MSE(y_true,y_pred):                                                #带有权重的均方误差（Weighted Mean Squared Error）
                                                                          #a、b、c是预先设定的数值
    a = 1.5
    b = 0.5
    c = 1.0
    d = c * math.sqrt(abs(2 * math.log(a)))                               #2乘以 以a为底的对数的绝对值 的平方根，然后乘以c
    condition = tf.abs(y_true - b) < d                                    #比较真实值与b之间的差的绝对值是否小于d，得到一个条件(condition)
    GW1 = a * tf.exp(-1 * tf.square(y_true - b) / (2 * math.pow(c,2)))    #指定公式
    GW = tf.where(condition,GW1,1)                                        #计算出一个权重值GW
    return K.mean(GW * tf.square(y_true - y_pred), axis = 0)              #返回带有权重的均方误差的平均值



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
dataframe4 = pd.read_excel(r'C:/Users/win10/Desktop/练习/代码 - 副本/插值5/第四点位.xlsx', header=None)  #E:\水质实验插值\第四点位_水质数据.xlsx
dataset4 = dataframe4.values  # 转换为numpy.ndarray数据

dataset4 = dataset4.astype('float32')  # 保障数据精度的同时还要考虑计算效率

# --------------------------点位5数据的处理---------------------------
dataframe5 = pd.read_excel(r'C:/Users/win10/Desktop/练习/代码 - 副本/插值5/第五点位.xlsx', header=None)
dataset5 = dataframe5.values  # 转换为numpy.ndarray数据

dataset5 = dataset5.astype('float32')  # 保障数据精度的同时还要考虑计算效率

data = np.concatenate((dataset1, dataset2), axis=1)
data = np.concatenate((data, dataset3), axis=1)
data = np.concatenate((data, dataset4), axis=1)
data = np.concatenate((data, dataset5), axis=1)


#先进性归一化处理
scaler = MinMaxScaler(feature_range=(0, 1))   #首先，创建一个MinMaxScaler对象，并设置feature_range参数为(0, 1)，表示将数据转换到0到1的范围内。
#转换为0-1之间的数(归一化处理)
data_g = scaler.fit_transform((data))         #使用fit_transform方法对data进行归一化处理，并将结果保存在data_g变量中。
                                              #然后，通过切片操作将data_g分割成多个子数据集。
dataset_g1 = data_g[:,0:6]                    #dataset_g1包含data_g的第一列到第六列
dataset_g2 = data_g[:,6:12]                   #dataset_g2包含第七列到第十二列
dataset_g3 = data_g[:,12:18]                  #dataset_g3包含第十三列到第十八列
dataset_g4 = data_g[:,18:24]                  #dataset_g4包含第十九列到第二十四列
dataset_g5 = data_g[:,24:30]                  #dataset_g5包含第二十五列到第三十列，这些子数据集是归一化后的数据，范围都在0到1之间

trainX1 = dataset_g1[:len(dataset_g1) - 1,:].reshape(len(dataset_g1) - 1,1,1,6)
trainY1 = dataset_g1[1:len(dataset_g1),:].reshape(len(dataset_g1) - 1,6)
testX1 = dataset_g1[int(len(dataset_g1) * 0.8) - 1:,:]
testX1 = testX1.reshape(len(testX1),1,1,6)

trainX2 = dataset_g2[:len(dataset_g2) - 1,:].reshape(len(dataset_g2) - 1,1,1,6)
trainY2 = dataset_g2[1:len(dataset_g2),:].reshape(len(dataset_g2) - 1,6)
testX2 = dataset_g2[int(len(dataset_g2) * 0.8) - 1:,:]
testX2 = testX2.reshape(len(testX1),1,1,6)

trainX3 = dataset_g3[:len(dataset_g3) - 1,:].reshape(len(dataset_g3) - 1,1,1,6)
trainY3 = dataset_g3[1:len(dataset_g3),:].reshape(len(dataset_g3) - 1,6)
testX3 = dataset_g3[int(len(dataset_g3) * 0.8) - 1:,:]
testX3 = testX3.reshape(len(testX3),1,1,6)

trainX4 = dataset_g4[:len(dataset_g4) - 1,:].reshape(len(dataset_g4) - 1,1,1,6)
trainY4 = dataset_g4[1:len(dataset_g4),:].reshape(len(dataset_g4) - 1,6)
testX4 = dataset_g4[int(len(dataset_g4) * 0.8) - 1:,:]
testX4 = testX4.reshape(len(testX4),1,1,6)

trainX5 = dataset_g5[:len(dataset_g5) - 1,:].reshape(len(dataset_g5) - 1,1,1,6)
trainY5 = dataset_g5[1:len(dataset_g5),:].reshape(len(dataset_g5) - 1,6)
testX5 = dataset_g5[int(len(dataset_g5) * 0.8) - 1:,:]
testX5 = testX5.reshape(len(testX5),1,1,6)

trainY =  np.concatenate((trainY1, trainY2), axis=1)
trainY =  np.concatenate((trainY, trainY3), axis=1)
trainY =  np.concatenate((trainY, trainY4), axis=1)
trainY =  np.concatenate((trainY, trainY5), axis=1)     #首先，使用numpy库中的concatenate函数将多个训练集按列(axis=1)进行拼接。
trainY = trainY.reshape(len(trainY),1,1,30)             #接下来，使用reshape函数将trainY转换为一个形状为(len(testX1), 1, 1, 30)的四维数组。
                                                        #这里的(len(trainY))表示训练集样本的数量，1表示每个样本的高度，1表示每个样本的宽度，30表示每个样本的特征数量。
#模型搭建
input_1=tf.keras.layers.Input(shape=(1,6))              #使用Keras库中的Input函数定义了五个输入层(input_1, input_2, input_3, input_4, input_5)。
input_2=tf.keras.layers.Input(shape=(1,6))              #每个输入层的形状(shape)被设置为(1, 6)，表示每个输入层接受一个形状为(1, 6)的输入。
input_3=tf.keras.layers.Input(shape=(1,6))              #这里的1表示每个输入样本的高度，6表示每个输入样本的特征数量。
input_4=tf.keras.layers.Input(shape=(1,6))              #这段代码的目的是定义了五个输入层，用于接收不同的输入数据。
input_5=tf.keras.layers.Input(shape=(1,6))              #在神经网络模型中，可以将这些输入层与其他层连接起来，以构建一个多输入的模型，以便处理多个输入数据。

                                               #这段代码是使用Keras库中的LSTM层和激活函数来构建神经网络模型的一部分。
hidden_1 = LSTM(units = 25)(input_1)           #首先，通过LSTM层将输入层input_1连接到一个具有18个神经元(units)的隐藏层(hidden_1)。
act1 = Activation("sigmoid")(hidden_1)         #LSTM是一种循环神经网络层，它可以处理序列数据，并具有记忆能力。这里的units参数指定了隐藏层中的神经元数量。
hidden_2 = LSTM(units = 24)(input_2)           #接下来，通过激活函数Activation("sigmoid")对隐藏层的输出进行激活。这里使用的是sigmoid激活函数，它将输出值限制在0到1之间。
act2 = Activation("sigmoid")(hidden_2)         #激活函数的作用是引入非线性特性，使神经网络能够学习更复杂的模式和关系。
hidden_3 = LSTM(units = 24)(input_3)
act3 = Activation("sigmoid")(hidden_3)
hidden_4 = LSTM(units = 26)(input_4)
act4 = Activation("sigmoid")(hidden_4)
hidden_5 = LSTM(units = 25)(input_5)
act5 = Activation("sigmoid")(hidden_5)
concat = concatenate([act1,act2,act3,act4,act5])   #将act1、act2、act3、act4和act5这五个张量进行连接（concatenate），得到一个新的张量concat。
out = Dense(30)(concat)                            #将concat输入到一个具有30个神经元的全连接层（Dense）中，得到输出张量out。
model=Model(inputs=[input_1,input_2,input_3,input_4,input_5],outputs=[out])  #创建了一个模型（model），该模型的输入是input_1、input_2、input_3、input_4和input_5这五个张量，输出是out张量。
sgd = SGD(learning_rate= 0.6, momentum=0.4, nesterov=True)                   #使用随机梯度下降（SGD）优化器，设置学习率为0.6，动量为0.4，并启用Nesterov加速。
model.compile(loss='mse',optimizer=sgd)                                      #使用均方误差（MSE）作为损失函数，将优化器和损失函数编译到模型中。
basesize = int(len(dataset_g1) * 0.8) - 2                                    #定义了一个变量basesize，其值为dataset_g1长度的80%再减去2。这个变量可能用于数据集的划分或其他用途。
yhat = np.zeros((1,30))                                                                                 #创建一个形状为(1,30)的全零数组yhat
for i in range(len(trainX1) - 1):                                                                       #通过循环遍历trainX1的长度减1的范围，对模型进行批量训练。
    model.train_on_batch([trainX1[i],trainX2[i],trainX3[i],trainX4[i],trainX5[i],], trainY[i])          #每次训练时，将trainX1[i]、trainX2[i]、trainX3[i]、trainX4[i]和trainX5[i]作为输入，trainY[i]作为目标输出。

    if i >= basesize:
        yhat1 = model.predict_on_batch([testX1[i - basesize],testX2[i - basesize],testX3[i - basesize],testX4[i - basesize],testX5[i - basesize]])
        yhat1 = np.reshape(yhat1,(1,30))              #如果i大于等于basesize，使用模型对testX1[i - basesize]、testX2[i - basesize]、testX3[i - basesize]、testX4[i - basesize]和testX5[i - basesize]进行批量预测，得到预测结果yhat1。
        yhat = np.concatenate((yhat,yhat1),axis=0)    #将yhat1的形状重塑为(1,30)，确保与yhat的形状一致。使用np.concatenate函数将yhat1与yhat在垂直方向上进行拼接，得到更新后的yhat。


yhat = yhat[1:, :]
arr = data_g[:286, :]
arr = np.concatenate((arr, yhat), axis=0)
inv_data_ = scaler.inverse_transform(arr)
# 第一个点位
inv_data_1 = inv_data_[:, 0:6]
inv_yhat_1 = inv_data_1[286:, :]
inv_testY_1 = dataset1[286:, :]
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)           #创建一个名为book的Excel工作簿对象，设置编码为UTF-8，并关闭样式压缩
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)                #在book中添加一个名为'sheet1'的工作表，允许单元格覆盖写入
for i in range(inv_yhat_1.shape[0]):                                   #使用双重循环遍历inv_yhat_1数组的每个元素，将其转换为字符串
    for j in range(inv_yhat_1.shape[1]):                               #并使用sheet.write方法将其写入工作表中。第一个参数是行索引，第二个参数是列索引，第三个参数是要写入的字符串值
        sheet.write(i,j,str(inv_yhat_1[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图在线学习/MV_Online_1.xls'   #指定保存路径
book.save(savepath)                                                                                         #使用book.save方法将工作簿保存到指定路径

book1 = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book1.add_sheet('sheet1', cell_overwrite_ok=True)
for i in range(inv_testY_1.shape[0]):
    for j in range(inv_testY_1.shape[1]):
        sheet.write(i, j, str(inv_testY_1[i][j]))

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/真实值/第一点位.xls'
book1.save(savepath)

inv_data_2 = inv_data_[:,6:12]
inv_yhat_2 = inv_data_2[286:,:]
inv_testY_2 = dataset2[286:,:]
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_2.shape[0]):
    for j in range(inv_yhat_2.shape[1]):
        sheet.write(i,j,str(inv_yhat_2[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图在线学习/MV_Online_2.xls'
book.save(savepath)

book1 = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book1.add_sheet('sheet1', cell_overwrite_ok=True)
for i in range(inv_testY_2.shape[0]):
    for j in range(inv_testY_2.shape[1]):
        sheet.write(i, j, str(inv_testY_2[i][j]))

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/真实值/第二点位.xls'
book1.save(savepath)
#第三个点位

inv_data_3 = inv_data_[:,12:18]
inv_yhat_3 = inv_data_3[286:,:]
inv_testY_3 = dataset3[286:,:]
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_3.shape[0]):
    for j in range(inv_yhat_3.shape[1]):
        sheet.write(i,j,str(inv_yhat_3[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组
savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图在线学习/MV_Online_3.xls'
book.save(savepath)

book1 = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book1.add_sheet('sheet1', cell_overwrite_ok=True)
for i in range(inv_testY_3.shape[0]):
    for j in range(inv_testY_3.shape[1]):
        sheet.write(i, j, str(inv_testY_3[i][j]))

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/真实值/第三点位.xls'
book1.save(savepath)


inv_data_4 = inv_data_[:,18:24]
inv_yhat_4 = inv_data_4[286:,:]
inv_testY_4 = dataset4[286:,:]
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_4.shape[0]):
    for j in range(inv_yhat_4.shape[1]):
        sheet.write(i,j,str(inv_yhat_4[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组
savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图在线学习/MV_Online_4.xls'
book.save(savepath)

book1 = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book1.add_sheet('sheet1', cell_overwrite_ok=True)
for i in range(inv_testY_4.shape[0]):
    for j in range(inv_testY_4.shape[1]):
        sheet.write(i, j, str(inv_testY_4[i][j]))

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/真实值/第四点位.xls'
book1.save(savepath)


inv_data_5 = inv_data_[:,24:30]
inv_yhat_5 = inv_data_5[286:,:]
inv_testY_5 = dataset5[286:,:]
#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_5.shape[0]):
    for j in range(inv_yhat_5.shape[1]):
        sheet.write(i,j,str(inv_yhat_5[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组
savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图在线学习/MV_Online_5.xls'
book.save(savepath)

book1 = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book1.add_sheet('sheet1', cell_overwrite_ok=True)
for i in range(inv_testY_5.shape[0]):
    for j in range(inv_testY_5.shape[1]):
        sheet.write(i, j, str(inv_testY_5[i][j]))

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/真实值/第五点位.xls'
book1.save(savepath)

#第一个点位
#预测
l1_y_1 = inv_yhat_1[:,0]  #第一个参数的数据        #从inv_yhat_1数组中提取出第一列的数据，赋值给l1_y_1。

l2_y_1 = inv_yhat_1[:,1]   #第二个参数的数据       #从inv_yhat_1数组中提取出第二列的数据，赋值给l2_y_1。

l3_y_1 = inv_yhat_1[:,2]  #第三个参数的数据                         .

l4_y_1 = inv_yhat_1[:,3]  #第三个参数的数据                         .

l5_y_1 = inv_yhat_1[:,4]  #第三个参数的数据                         .

l6_y_1 = inv_yhat_1[:,5]  #第三个参数的数据        #从inv_yhat_1数组中提取出第六列的数据，赋值给l6_y_1。

#测试
c1_y_1 = inv_testY_1[:,0] #第一个参数的数据        #从inv_testY_1数组中提取出第一列的数据，赋值给c1_y_1。

c2_y_1 = inv_testY_1[:,1]   #第二个参数的数据      #从inv_testY_1数组中提取出第二列的数据，赋值给c2_y_1。

c3_y_1 = inv_testY_1[:,2]  #第三个参数的数据                         .

c4_y_1 = inv_testY_1[:,3]  #第三个参数的数据                         .

c5_y_1 = inv_testY_1[:,4]  #第三个参数的数据                         .

c6_y_1 = inv_testY_1[:,5]  #第三个参数的数据       #从inv_testY_1数组中提取出第六列的数据，赋值给c6_y_1。

#第二个点位
#预测
l1_y_2 = inv_yhat_2[:,0]  #第一个参数的数据

l2_y_2 = inv_yhat_2[:,1]   #第二个参数的数据

l3_y_2 = inv_yhat_2[:,2]  #第三个参数的数据

l4_y_2 = inv_yhat_2[:,3]  #第三个参数的数据

l5_y_2 = inv_yhat_2[:,4]  #第三个参数的数据

l6_y_2 = inv_yhat_2[:,5]  #第三个参数的数据

#测试
c1_y_2 = inv_testY_2[:,0] #第一个参数的数据

c2_y_2 = inv_testY_2[:,1]   #第二个参数的数据

c3_y_2 = inv_testY_2[:,2]  #第三个参数的数据

c4_y_2 = inv_testY_2[:,3]  #第四个参数的数据

c5_y_2 = inv_testY_2[:,4]  #第五个参数的数据

c6_y_2 = inv_testY_2[:,5]  #第六个参数的数据

#第三个点位
#预测
l1_y_3 = inv_yhat_3[:,0]  #第一个参数的数据

l2_y_3 = inv_yhat_3[:,1]   #第二个参数的数据

l3_y_3 = inv_yhat_3[:,2]  #第三个参数的数据

l4_y_3 = inv_yhat_3[:,3]  #第四个参数的数据

l5_y_3 = inv_yhat_3[:,4]  #第五个参数的数据

l6_y_3 = inv_yhat_3[:,5]  #第六个参数的数据

#测试
c1_y_3 = inv_testY_3[:,0] #第一个参数的数据

c2_y_3 = inv_testY_3[:,1]   #第二个参数的数据

c3_y_3 = inv_testY_3[:,2]  #第三个参数的数据

c4_y_3 = inv_testY_3[:,3]  #第四个参数的数据

c5_y_3 = inv_testY_3[:,4]  #第五个参数的数据

c6_y_3 = inv_testY_3[:,5]  #第六个参数的数据

#第四个点位
#预测
l1_y_4 = inv_yhat_4[:,0]  #第一个参数的数据

l2_y_4 = inv_yhat_4[:,1]   #第二个参数的数据

l3_y_4 = inv_yhat_4[:,2]  #第三个参数的数据

l4_y_4 = inv_yhat_4[:,3]  #第四个参数的数据

l5_y_4 = inv_yhat_4[:,4]  #第五个参数的数据

l6_y_4 = inv_yhat_4[:,5]  #第六个参数的数据

#测试
c1_y_4 = inv_testY_4[:,0] #第一个参数的数据

c2_y_4 = inv_testY_4[:,1]   #第二个参数的数据

c3_y_4 = inv_testY_4[:,2]  #第三个参数的数据

c4_y_4 = inv_testY_4[:,3]  #第四个参数的数据

c5_y_4 = inv_testY_4[:,4]  #第五个参数的数据

c6_y_4 = inv_testY_4[:,5]  #第六个参数的数据

#第五个点位
#预测
l1_y_5 = inv_yhat_5[:,0]  #第一个参数的数据

l2_y_5 = inv_yhat_5[:,1]   #第二个参数的数据

l3_y_5 = inv_yhat_5[:,2]  #第三个参数的数据

l4_y_5 = inv_yhat_5[:,3]  #第四个参数的数据

l5_y_5 = inv_yhat_5[:,4]  #第五个参数的数据

l6_y_5 = inv_yhat_5[:,5]  #第六个参数的数据

#测试
c1_y_5 = inv_testY_5[:,0] #第一个参数的数据

c2_y_5 = inv_testY_5[:,1]   #第二个参数的数据

c3_y_5 = inv_testY_5[:,2]  #第三个参数的数据

c4_y_5 = inv_testY_5[:,3]  #第四个参数的数据

c5_y_5 = inv_testY_5[:,4]  #第五个参数的数据

c6_y_5 = inv_testY_5[:,5]  #第六个参数的数据
#x轴
x = np.arange(72)          #创建一个包含24个元素的NumPy数组(一维数组），使用np.arange函数生成从0到23的连续整数序列，并将其赋值给变量x。
#第一个点位
from pylab import *                                                                  #使用Python的matplotlib库进行绘图操作。通过from pylab import *语句导入了pylab模块，它包含了许多数学函数和绘图功能。
plt.plot(x,l1_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')   #绘制了一条蓝色曲线。其中，x是横轴数据，l1_y_1是纵轴数据，color='blue'设置曲线颜色为蓝色，linewidth=2.0设置曲线的线宽为2.0，linestyle='-'设置曲线的线型为实线，marker='o'设置曲线上的点为圆圈形状，label='yc'设置曲线的标签为'yc'。
plt.plot(x,c1_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')    #绘制了一条红色曲线
plt.show()                                                                           #显示绘制的曲线图形

plt.plot(x,l2_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c2_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l3_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c3_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l4_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c4_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l5_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c5_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l6_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c6_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

#第二个点位
plt.plot(x,l1_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c1_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l2_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c2_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l3_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c3_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l4_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c4_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l5_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c5_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l6_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c6_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

#第三个点位
plt.plot(x,l1_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c1_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l2_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c2_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l3_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c3_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l4_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c4_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l5_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c5_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l6_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c6_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

#第四个点位
plt.plot(x,l1_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c1_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l2_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c2_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l3_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c3_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l4_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c4_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l5_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c5_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l6_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c6_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

#第五个点位
plt.plot(x,l1_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c1_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l2_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c2_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l3_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c3_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l4_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c4_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l5_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c5_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()

plt.plot(x,l6_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
plt.plot(x,c6_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
plt.show()


print("---------------------------第一个点位---------------------------")
print("---------------------------ph指标---------------------------")
# print("pH相对误差为：" + str(relative_error(c1_y_1,l1_y_1)))
# print("pH平均相对误差为：" + str(mean_relative_error(c1_y_1,l1_y_1)))
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_1,l1_y_1)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_1,l1_y_1)))
print("pH决定系数为：" + str(r2_score(c1_y_1,l1_y_1)))                       #决定系数（R-squared score）

print("---------------------------容解氧指标---------------------------")
# print("容解氧相对误差为：" + str(relative_error(c2_y_1,l2_y_1)))
# print("容解氧平均相对误差为：" + str(mean_relative_error(c2_y_1,l2_y_1)))
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_1,l2_y_1)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_1,l2_y_1)))
print("容解氧决定系数为：" + str(r2_score(c2_y_1,l2_y_1)))

print("---------------------------电导率指标---------------------------")
# print("电导率相对误差为：" + str(relative_error(c3_y_1,l3_y_1)))
# print("电导率平均相对误差为：" + str(mean_relative_error(c3_y_1,l3_y_1)))
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_1,l3_y_1)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_1,l3_y_1)))
print("电导率决定系数为：" + str(r2_score(c3_y_1,l3_y_1)))

print("---------------------------浑浊度指标---------------------------")
# print("浑浊度相对误差为：" + str(relative_error(c4_y_1,l4_y_1)))
# print("浑浊度平均相对误差为：" + str(mean_relative_error(c4_y_1,l4_y_1)))
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_1,l4_y_1)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_1,l4_y_1)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_1,l4_y_1)))

print("---------------------------氨氮指标---------------------------")
# print("氨氮相对误差为：" + str(relative_error(c5_y_1,l5_y_1)))
# print("氨氮平均相对误差为：" + str(mean_relative_error(c5_y_1,l5_y_1)))
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_1,l5_y_1)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_1,l5_y_1)))
print("氨氮决定系数为：" + str(r2_score(c5_y_1,l5_y_1)))

print("---------------------------耗氧量指标---------------------------")
# print("耗氧量相对误差为：" + str(relative_error(c6_y_1,l6_y_1)))
# print("耗氧量平均相对误差为：" + str(mean_relative_error(c6_y_1,l6_y_1)))
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_1,l6_y_1)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_1,l6_y_1)))
print("耗氧量决定系数为：" + str(r2_score(c6_y_1,l6_y_1)))

#第二个点位
print("---------------------------第二个点位---------------------------")
print("---------------------------ph指标---------------------------")
# print("pH相对误差为：" + str(relative_error(c1_y_2,l1_y_2)))
# print("pH平均相对误差为：" + str(mean_relative_error(c1_y_2,l1_y_2)))
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_2,l1_y_2)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_2,l1_y_2)))
print("pH决定系数为：" + str(r2_score(c1_y_2,l1_y_2)))

print("---------------------------容解氧指标---------------------------")
# print("容解氧相对误差为：" + str(relative_error(c2_y_2,l2_y_2)))
# print("容解氧平均相对误差为：" + str(mean_relative_error(c2_y_2,l2_y_2)))
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_2,l2_y_2)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_2,l2_y_2)))
print("容解氧决定系数为：" + str(r2_score(c2_y_2,l2_y_2)))

print("---------------------------电导率指标---------------------------")
# print("电导率相对误差为：" + str(relative_error(c3_y_2,l3_y_2)))
# print("电导率平均相对误差为：" + str(mean_relative_error(c3_y_2,l3_y_2)))
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_2,l3_y_2)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_2,l3_y_2)))
print("电导率决定系数为：" + str(r2_score(c3_y_2,l3_y_2)))

print("---------------------------浑浊度指标---------------------------")
# print("浑浊度相对误差为：" + str(relative_error(c4_y_2,l4_y_2)))
# print("浑浊度平均相对误差为：" + str(mean_relative_error(c4_y_2,l4_y_2)))
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_2,l4_y_2)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_2,l4_y_2)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_2,l4_y_2)))

print("---------------------------氨氮指标---------------------------")
# print("氨氮相对误差为：" + str(relative_error(c5_y_2,l5_y_2)))
# print("氨氮平均相对误差为：" + str(mean_relative_error(c5_y_2,l5_y_2)))
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_2,l5_y_2)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_2,l5_y_2)))
print("氨氮决定系数为：" + str(r2_score(c5_y_2,l5_y_2)))

print("---------------------------耗氧量指标---------------------------")
# print("耗氧量相对误差为：" + str(relative_error(c6_y_2,l6_y_2)))
# print("耗氧量平均相对误差为：" + str(mean_relative_error(c6_y_2,l6_y_2)))
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_2,l6_y_2)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_2,l6_y_2)))
print("耗氧量决定系数为：" + str(r2_score(c6_y_2,l6_y_2)))

#第三个点位
print("---------------------------第三个点位---------------------------")
print("---------------------------ph指标---------------------------")
# print("pH相对误差为：" + str(relative_error(c1_y_3,l1_y_3)))
# print("pH平均相对误差为：" + str(mean_relative_error(c1_y_3,l1_y_3)))
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_3,l1_y_3)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_3,l1_y_3)))
print("pH决定系数为：" + str(r2_score(c1_y_3,l1_y_3)))

print("---------------------------容解氧指标---------------------------")
# print("容解氧相对误差为：" + str(relative_error(c2_y_3,l2_y_3)))
# print("容解氧平均相对误差为：" + str(mean_relative_error(c2_y_3,l2_y_3)))
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_3,l2_y_3)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_3,l2_y_3)))
print("容解氧决定系数为：" + str(r2_score(c2_y_3,l2_y_3)))

print("---------------------------电导率指标---------------------------")
# print("电导率相对误差为：" + str(relative_error(c3_y_3,l3_y_3)))
# print("电导率平均相对误差为：" + str(mean_relative_error(c3_y_3,l3_y_3)))
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_3,l3_y_3)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_3,l3_y_3)))
print("电导率决定系数为：" + str(r2_score(c3_y_3,l3_y_3)))

print("---------------------------浑浊度指标---------------------------")
# print("浑浊度相对误差为：" + str(relative_error(c4_y_3,l4_y_3)))
# print("浑浊度平均相对误差为：" + str(mean_relative_error(c4_y_3,l4_y_3)))
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_3,l4_y_3)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_3,l4_y_3)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_3,l4_y_3)))

print("---------------------------氨氮指标---------------------------")
# print("氨氮相对误差为：" + str(relative_error(c5_y_3,l5_y_3)))
# print("氨氮平均相对误差为：" + str(mean_relative_error(c5_y_3,l5_y_3)))
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_3,l5_y_3)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_3,l5_y_3)))
print("氨氮决定系数为：" + str(r2_score(c5_y_3,l5_y_3)))

print("---------------------------耗氧量指标---------------------------")
# print("耗氧量相对误差为：" + str(relative_error(c6_y_3,l6_y_3)))
# print("耗氧量平均相对误差为：" + str(mean_relative_error(c6_y_3,l6_y_3)))
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_3,l6_y_3)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_3,l6_y_3)))
print("耗氧量决定系数为：" + str(r2_score(c6_y_3,l6_y_3)))

#第四个点位
print("---------------------------第四个点位---------------------------")
print("---------------------------ph指标---------------------------")
# print("pH相对误差为：" + str(relative_error(c1_y_4,l1_y_4)))
# print("pH平均相对误差为：" + str(mean_relative_error(c1_y_4,l1_y_4)))
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_4,l1_y_4)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_4,l1_y_4)))
print("pH决定系数为：" + str(r2_score(c1_y_4,l1_y_4)))

print("---------------------------容解氧指标---------------------------")
# print("容解氧相对误差为：" + str(relative_error(c2_y_4,l2_y_4)))
# print("容解氧平均相对误差为：" + str(mean_relative_error(c2_y_4,l2_y_4)))
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_4,l2_y_4)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_4,l2_y_4)))
print("容解氧决定系数为：" + str(r2_score(c2_y_4,l2_y_4)))

print("---------------------------电导率指标---------------------------")
# print("电导率相对误差为：" + str(relative_error(c3_y_4,l3_y_4)))
# print("电导率平均相对误差为：" + str(mean_relative_error(c3_y_4,l3_y_4)))
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_4,l3_y_4)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_4,l3_y_4)))
print("电导率决定系数为：" + str(r2_score(c3_y_4,l3_y_4)))

print("---------------------------浑浊度指标---------------------------")
# print("浑浊度相对误差为：" + str(relative_error(c4_y_4,l4_y_4)))
# print("浑浊度平均相对误差为：" + str(mean_relative_error(c4_y_4,l4_y_4)))
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_4,l4_y_4)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_4,l4_y_4)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_4,l4_y_4)))

print("---------------------------氨氮指标---------------------------")
# print("氨氮相对误差为：" + str(relative_error(c5_y_4,l5_y_4)))
# print("氨氮平均相对误差为：" + str(mean_relative_error(c5_y_4,l5_y_4)))
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_4,l5_y_4)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_4,l5_y_4)))
print("氨氮决定系数为：" + str(r2_score(c5_y_4,l5_y_4)))

print("---------------------------耗氧量指标---------------------------")
# print("耗氧量相对误差为：" + str(relative_error(c6_y_4,l6_y_4)))
# print("耗氧量平均相对误差为：" + str(mean_relative_error(c6_y_4,l6_y_4)))
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_4,l6_y_4)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_4,l6_y_4)))
print("耗氧量决定系数为：" + str(r2_score(c6_y_4,l6_y_4)))

#第五个点位
print("---------------------------第五个点位---------------------------")
print("---------------------------ph指标---------------------------")
# print("pH相对误差为：" + str(relative_error(c1_y_5,l1_y_5)))
# print("pH平均相对误差为：" + str(mean_relative_error(c1_y_5,l1_y_5)))
print("pH平均绝对误差为：" + str(mean_absolute_error(c1_y_5,l1_y_5)))
print("pH均方根误差为：" + str(R_mean_squ_error(c1_y_5,l1_y_5)))
print("pH决定系数为：" + str(r2_score(c1_y_5,l1_y_5)))

print("---------------------------容解氧指标---------------------------")
# print("容解氧相对误差为：" + str(relative_error(c2_y_5,l2_y_5)))
# print("容解氧平均相对误差为：" + str(mean_relative_error(c2_y_5,l2_y_5)))
print("容解氧平均绝对误差为：" + str(mean_absolute_error(c2_y_5,l2_y_5)))
print("容解氧均方根误差为：" + str(R_mean_squ_error(c2_y_5,l2_y_5)))
print("容解氧决定系数为：" + str(r2_score(c2_y_5,l2_y_5)))

print("---------------------------电导率指标---------------------------")
# print("电导率相对误差为：" + str(relative_error(c3_y_5,l3_y_5)))
# print("电导率平均相对误差为：" + str(mean_relative_error(c3_y_5,l3_y_5)))
print("电导率平均绝对误差为：" + str(mean_absolute_error(c3_y_5,l3_y_5)))
print("电导率均方根误差为：" + str(R_mean_squ_error(c3_y_5,l3_y_5)))
print("电导率决定系数为：" + str(r2_score(c3_y_5,l3_y_5)))

print("---------------------------浑浊度指标---------------------------")
# print("浑浊度相对误差为：" + str(relative_error(c4_y_5,l4_y_5)))
# print("浑浊度平均相对误差为：" + str(mean_relative_error(c4_y_5,l4_y_5)))
print("浑浊度平均绝对误差为：" + str(mean_absolute_error(c4_y_5,l4_y_5)))
print("浑浊度均方根误差为：" + str(R_mean_squ_error(c4_y_5,l4_y_5)))
print("浑浊度决定系数为：" + str(r2_score(c4_y_5,l4_y_5)))

print("---------------------------氨氮指标---------------------------")
# print("氨氮相对误差为：" + str(relative_error(c5_y_5,l5_y_5)))
# print("氨氮平均相对误差为：" + str(mean_relative_error(c5_y_5,l5_y_5)))
print("氨氮平均绝对误差为：" + str(mean_absolute_error(c5_y_5,l5_y_5)))
print("氨氮均方根误差为：" + str(R_mean_squ_error(c5_y_5,l5_y_5)))
print("氨氮决定系数为：" + str(r2_score(c5_y_5,l5_y_5)))

print("---------------------------耗氧量指标---------------------------")
# print("耗氧量相对误差为：" + str(relative_error(c6_y_5,l6_y_5)))
# print("耗氧量平均相对误差为：" + str(mean_relative_error(c6_y_5,l6_y_5)))
print("耗氧量平均绝对误差为：" + str(mean_absolute_error(c6_y_5,l6_y_5)))
print("耗氧量均方根误差为：" + str(R_mean_squ_error(c6_y_5,l6_y_5)))
print("耗氧量决定系数为：" + str(r2_score(c6_y_5,l6_y_5)))
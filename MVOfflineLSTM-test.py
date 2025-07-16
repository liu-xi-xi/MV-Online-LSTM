from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import r2_score
import pandas as pd
import xlwt
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
scaler = MinMaxScaler(feature_range=(0, 1))     #转换为0-1之间的数(归一化处理)
data_g = scaler.fit_transform((data))

dataset_g1 = data_g[:,0:6]
dataset_g2 = data_g[:,6:12]
dataset_g3 = data_g[:,12:18]
dataset_g4 = data_g[:,18:24]
dataset_g5 = data_g[:,24:30]

#训练集
trainX1 = dataset_g1[:285,:]
trainX1 = trainX1.reshape(len(trainX1),1,6)

trainX2 = dataset_g2[:285,:]
trainX2 = trainX2.reshape(len(trainX2),1,6)

trainX3 = dataset_g3[:285,:]
trainX3 = trainX3.reshape(len(trainX3),1,6)

trainX4 = dataset_g4[:285,:]
trainX4 = trainX4.reshape(len(trainX4),1,6)

trainX5 = dataset_g5[:285,:]
trainX5 = trainX5.reshape(len(trainX5),1,6)
trainY1 = dataset_g1[1:286,:]
trainY2 = dataset_g2[1:286,:]
trainY3 = dataset_g3[1:286,:]
trainY4 = dataset_g4[1:286,:]
trainY5 = dataset_g5[1:286,:]

trainY = np.concatenate((trainY1, trainY2), axis=1)
trainY = np.concatenate((trainY, trainY3), axis=1)
trainY = np.concatenate((trainY, trainY4), axis=1)
trainY = np.concatenate((trainY, trainY5), axis=1)
trainY = trainY.reshape(len(trainY),1,30)

#测试集
testX1 = dataset_g1[285:357].reshape(72,1,6)
testX2 = dataset_g2[285:357].reshape(72,1,6)
testX3 = dataset_g3[285:357].reshape(72,1,6)
testX4 = dataset_g4[285:357].reshape(72,1,6)
testX5 = dataset_g5[285:357].reshape(72,1,6)

testY1 = dataset_g1[286:].reshape(72,1,6)
testY2 = dataset_g2[286:].reshape(72,1,6)
testY3 = dataset_g3[286:].reshape(72,1,6)
testY4 = dataset_g4[286:].reshape(72,1,6)
testY5 = dataset_g5[286:].reshape(72,1,6)

testY = np.concatenate((testY1, testY2), axis=1)
testY = np.concatenate((testY, testY3), axis=1)
testY = np.concatenate((testY, testY4), axis=1)
testY = np.concatenate((testY, testY5), axis=1)
testY = testY.reshape(len(testY),1,30)
model = load_model('MV-LSTM.h5')
yhat = model.predict([testX1,testX2,testX3,testX4,testX5])

yhat = yhat.reshape(72,30)
arr = data_g[:286,:]
arr = np.concatenate((arr, yhat), axis=0)

inv_data_ = scaler.inverse_transform(arr)  #反归一化

inv_data_1 = inv_data_[:,0:6]
inv_yhat_1 = inv_data_1[286:,:]
inv_testY_1 = dataset1[286:,:]


#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_1.shape[0]):
    for j in range(inv_yhat_1.shape[1]):
        sheet.write(i,j,str(inv_yhat_1[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图离线学习/MV_Offline_1.xls'
book.save(savepath)

#第二个点位

inv_data_2 = inv_data_[:,6:12]
inv_yhat_2 = inv_data_2[286:,:]
inv_testY_2 = dataset2[286:,:]

#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_2.shape[0]):
    for j in range(inv_yhat_2.shape[1]):
        sheet.write(i,j,str(inv_yhat_2[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图离线学习/MV_Offline_2.xls'
book.save(savepath)

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

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图离线学习/MV_Offline_3.xls'
book.save(savepath)

#第四个点位
inv_data_4 = inv_data_[:,18:24]
inv_yhat_4 = inv_data_4[286:,:]
inv_testY_4 = dataset4[286:,:]


#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_4.shape[0]):
    for j in range(inv_yhat_4.shape[1]):
        sheet.write(i,j,str(inv_yhat_4[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图离线学习/MV_Offline_4.xls'
book.save(savepath)

#第五个点位
inv_data_5 = inv_data_[:,24:30]
inv_yhat_5 = inv_data_5[286:,:]
inv_testY_5 = dataset5[286:,:]


#数据提取
book = xlwt.Workbook(encoding = 'utf-8',style_compression=0)
sheet = book.add_sheet('sheet1',cell_overwrite_ok=True)
for i in range(inv_yhat_5.shape[0]):
    for j in range(inv_yhat_5.shape[1]):
        sheet.write(i,j,str(inv_yhat_5[i][j]))  #第一个参数是行，第二个参数是列，第三个参数是col元组

savepath = 'C:/Users/win10/Desktop/练习/SV-MV/多视图离线学习/MV_Offline_5.xls'
book.save(savepath)

#第一个点位
#预测
l1_y_1 = inv_yhat_1[:,0]  #第一个参数的数据

l2_y_1 = inv_yhat_1[:,1]   #第二个参数的数据

l3_y_1 = inv_yhat_1[:,2]  #第三个参数的数据

l4_y_1 = inv_yhat_1[:,3]  #第三个参数的数据

l5_y_1 = inv_yhat_1[:,4]  #第三个参数的数据

l6_y_1 = inv_yhat_1[:,5]  #第三个参数的数据

#测试
c1_y_1 = inv_testY_1[:,0] #第一个参数的数据

c2_y_1 = inv_testY_1[:,1]   #第二个参数的数据

c3_y_1 = inv_testY_1[:,2]  #第三个参数的数据

c4_y_1 = inv_testY_1[:,3]  #第三个参数的数据

c5_y_1 = inv_testY_1[:,4]  #第三个参数的数据

c6_y_1 = inv_testY_1[:,5]  #第三个参数的数据

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

c4_y_2 = inv_testY_2[:,3]  #第三个参数的数据

c5_y_2 = inv_testY_2[:,4]  #第三个参数的数据

c6_y_2 = inv_testY_2[:,5]  #第三个参数的数据

#第三个点位
#预测
l1_y_3 = inv_yhat_3[:,0]  #第一个参数的数据

l2_y_3 = inv_yhat_3[:,1]   #第二个参数的数据

l3_y_3 = inv_yhat_3[:,2]  #第三个参数的数据

l4_y_3 = inv_yhat_3[:,3]  #第三个参数的数据

l5_y_3 = inv_yhat_3[:,4]  #第三个参数的数据

l6_y_3 = inv_yhat_3[:,5]  #第三个参数的数据

#测试
c1_y_3 = inv_testY_3[:,0] #第一个参数的数据

c2_y_3 = inv_testY_3[:,1]   #第二个参数的数据

c3_y_3 = inv_testY_3[:,2]  #第三个参数的数据

c4_y_3 = inv_testY_3[:,3]  #第三个参数的数据

c5_y_3 = inv_testY_3[:,4]  #第三个参数的数据

c6_y_3 = inv_testY_3[:,5]  #第三个参数的数据

#第四个点位
#预测
l1_y_4 = inv_yhat_4[:,0]  #第一个参数的数据

l2_y_4 = inv_yhat_4[:,1]   #第二个参数的数据

l3_y_4 = inv_yhat_4[:,2]  #第三个参数的数据

l4_y_4 = inv_yhat_4[:,3]  #第三个参数的数据

l5_y_4 = inv_yhat_4[:,4]  #第三个参数的数据

l6_y_4 = inv_yhat_4[:,5]  #第三个参数的数据

#测试
c1_y_4 = inv_testY_4[:,0] #第一个参数的数据

c2_y_4 = inv_testY_4[:,1]   #第二个参数的数据

c3_y_4 = inv_testY_4[:,2]  #第三个参数的数据

c4_y_4 = inv_testY_4[:,3]  #第三个参数的数据

c5_y_4 = inv_testY_4[:,4]  #第三个参数的数据

c6_y_4 = inv_testY_4[:,5]  #第三个参数的数据

#第五个点位
#预测
l1_y_5 = inv_yhat_5[:,0]  #第一个参数的数据

l2_y_5 = inv_yhat_5[:,1]   #第二个参数的数据

l3_y_5 = inv_yhat_5[:,2]  #第三个参数的数据

l4_y_5 = inv_yhat_5[:,3]  #第三个参数的数据

l5_y_5 = inv_yhat_5[:,4]  #第三个参数的数据

l6_y_5 = inv_yhat_5[:,5]  #第三个参数的数据

#测试
c1_y_5 = inv_testY_5[:,0] #第一个参数的数据

c2_y_5 = inv_testY_5[:,1]   #第二个参数的数据

c3_y_5 = inv_testY_5[:,2]  #第三个参数的数据

c4_y_5 = inv_testY_5[:,3]  #第三个参数的数据

c5_y_5 = inv_testY_5[:,4]  #第三个参数的数据

c6_y_5 = inv_testY_5[:,5]  #第三个参数的数据
#x轴
x = np.arange(72)
from pylab import *
# #第一个点位
# plt.plot(x,l1_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c1_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l2_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c2_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l3_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c3_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l4_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c4_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l5_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c5_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l6_y_1,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c6_y_1,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# #第二个点位
# plt.plot(x,l1_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c1_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l2_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c2_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l3_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c3_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l4_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c4_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l5_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c5_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l6_y_2,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c6_y_2,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# #第三个点位
# plt.plot(x,l1_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c1_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l2_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c2_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l3_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c3_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l4_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c4_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l5_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c5_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l6_y_3,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c6_y_3,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# #第四个点位
# plt.plot(x,l1_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c1_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l2_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c2_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l3_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c3_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l4_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c4_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l5_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c5_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l6_y_4,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c6_y_4,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# #第五个点位
# plt.plot(x,l1_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c1_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l2_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c2_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l3_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c3_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l4_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c4_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l5_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c5_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()
#
# plt.plot(x,l6_y_5,color='blue',linewidth=2.0, linestyle='-',marker='o',label='yc')
# plt.plot(x,c6_y_5,color='red',linewidth=2.0, linestyle='-',marker='*',label='yc')
# plt.show()

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
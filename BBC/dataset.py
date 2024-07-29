import scipy.io
import pandas as pd 
import numpy as np
def Handwritten_numerals(shuffle=False):
    # 加载MAT文件
    mat = scipy.io.loadmat('../data/Handwritten_numerals.mat')
    # 提取cell和double数据
    data = mat['data']
    labels = mat['labels']
    random_list=range(0,2000)
    if(shuffle):
        # 生成1-2000的随机列表
        import random
        random_list = random.sample(range(0, 2000), 2000)
    #print(random_list)
    data0_list=[]
    data1_list=[]
    data2_list=[]
    data3_list=[]
    data4_list=[]
    data5_list=[]
    label_list=[]
    for i in random_list:
        data0_list.append(data[0][0][i].astype('float32'))
        data1_list.append(data[1][0][i].astype('float32'))
        data2_list.append(data[2][0][i].astype('float32'))
        data3_list.append(data[3][0][i].astype('float32'))
        data4_list.append(data[4][0][i].astype('float32'))
        data5_list.append(data[5][0][i].astype('float32'))
        label_list.append(labels[i][0])
    #print(label_list)
    return data0_list,data1_list,data2_list,data3_list,data4_list,data5_list,label_list
    
def Cal7(shuffle=False):
    # 加载MAT文件
    mat = scipy.io.loadmat('../data/Caltech101-7.mat')
    # 提取cell和double数据
    data = mat['data']
    labels = mat['labels']
    #print(data[0][1])
    length=len(labels)
    random_list=range(length)
    if(shuffle):
        # 生成1-2000的随机列表
        import random
        random_list = random.sample(range(length), length)
    #print(random_list)
    data0_list=[]
    data1_list=[]
    data2_list=[]
    data3_list=[]
    data4_list=[]
    data5_list=[]
    label_list=[]
    for i in random_list:
        data0_list.append(data[0][0][i].astype('float32'))
        data1_list.append(data[0][1][i].astype('float32'))
        data2_list.append(data[0][2][i].astype('float32'))
        data3_list.append(data[0][3][i].astype('float32'))
        data4_list.append(data[0][4][i].astype('float32'))
        data5_list.append(data[0][5][i].astype('float32'))
        label_list.append(labels[i][0]-1)
    #print(label_list)
    return data0_list,data1_list,data2_list,data3_list,data4_list,data5_list,label_list
    
def BBC(shuffle=False):
    # 加载MAT文件
    mat = scipy.io.loadmat('../data/BBC4view_685.mat')
    # 提取cell和double数据
    data = mat['data']
    labels = mat['labels']
    #print(data[0][1])
    length=len(labels)
    random_list=range(length)
    if(shuffle):
        # 生成1-2000的随机列表
        import random
        random_list = random.sample(range(length), length)
    #print(random_list)
    data0_list=[]
    data1_list=[]
    data2_list=[]
    data3_list=[]
    #data4_list=[]
    #data5_list=[]
    label_list=[]
    for i in random_list:
        data0_list.append(data[0][0][i].astype('float32'))
        data1_list.append(data[0][1][i].astype('float32'))
        data2_list.append(data[0][2][i].astype('float32'))
        data3_list.append(data[0][3][i].astype('float32'))
        #data4_list.append(data[0][4][i].astype('float32'))
        #data5_list.append(data[0][5][i].astype('float32'))
        label_list.append(labels[i][0]-1)
    #print(label_list)
    return data0_list,data1_list,data2_list,data3_list,label_list
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:59:09 2021

@author: gw
"""

from tkinter import *
from tkinter.tix import Tk,Control,ComboBox #升级的组合控件包
from tkinter.messagebox import showinfo,showwarning,showerror #各种消息提示框
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #画布
from matplotlib.figure import Figure


#界面初始设置
root = Tk() #初始化Tk

root.title("神经网络_GDAL_tkinter")
root.geometry("800x600")
root.resizable(width=True,height=True)
root.tk.eval('package require Tix')
var = StringVar()  #文本变量储存器
#------------------------------------------------------------------------------
import numpy as np
from osgeo import gdal
import cv2
from PIL import Image,ImageTk # 导入图像处理函数库

from scipy import io as spio
from scipy import optimize
from matplotlib import pyplot as plt
from scipy import optimize
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import time

global input_layer_size,hidden_layer_size,out_put_layer
input_layer_size=6
hidden_layer_size=25
out_put_layer=8

#1.定义sigmoid函数和代价函数
def sigmoid(z):
    h = np.zeros((len(z),1))
    
    h = 1.0/(1.0+np.exp(-z))
    return h
    
    
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,Lambda,X,y):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,
                                                                         input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):length].reshape(num_labels,
                                                                              hidden_layer_size+1)
    m = X.shape[0]
    class_y = np.zeros((m,num_labels))
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1) 
    
    #计算正则化项
    Theta1_x = Theta1[:,1:Theta1.shape[1]]
    Theta2_x = Theta2[:,1:Theta2.shape[1]]
    term = np.dot(np.transpose(np.vstack((Theta1_x.reshape(-1,1),Theta2_x.reshape(-1,1)))),
                  np.vstack((Theta1_x.reshape(-1,1),Theta2_x.reshape(-1,1))))
    
    #正向传播
    a1 = np.hstack((np.ones((m,1)),X))
    z2 = np.dot(a1,np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = np.dot(a2,np.transpose(Theta2))
    h = sigmoid(z3)
    
    J = -(np.dot(np.transpose(class_y.reshape(-1,1)),np.log(h.reshape(-1,1)))+
          np.dot(np.transpose(1-class_y.reshape(-1,1)),np.log(1-h.reshape(-1,1)))+
          Lambda*term/2)/m
    return np.ravel(J)

#2.定义梯度Sigmoid函数和梯度下降函数
def gradSigmoid(z):
    g = np.zeros((z.shape))
    
    g = sigmoid(z)*(1-sigmoid(z))
    return g
    
def nnGradient(nn_params,input_layer_size,hidden_layer_size,num_labels,Lambda,X,y):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,
                                                                         input_layer_size+1).copy()
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):length].reshape(num_labels,
                                                                              hidden_layer_size+1
                                                                              ).copy()
    m = X.shape[0]
    class_y = np.zeros((m,num_labels))
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==(i+1)).reshape(1,-1) 
    
    Theta1_x = Theta1[:,1:Theta1.shape[1]]
    Theta2_x = Theta2[:,1:Theta2.shape[1]]
    
    #正向传播
    n = X.shape[1]
    a1 = np.hstack((np.ones((m,1)),X))
    z2 = np.dot(a1,np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = np.dot(a2,np.transpose(Theta2))
    h = sigmoid(z3)
    
    
    Theta1_grad = np.zeros((Theta1.shape))
    Theta2_grad = np.zeros((Theta2.shape))
    #反向传播
    detal3 = np.zeros((m,num_labels))
    detal2 = np.zeros((m,hidden_layer_size))
    
    for i in range(m):
        #detal3[i,:] = (h[i,:]-class_y[i,:])*gradSigmoid(z3[i,:])  # 均方误差的误差率
        detal3[i,:] = h[i,:] - class_y[i,:]                #交叉熵误差率
        Theta2_grad = Theta2_grad+np.dot(np.transpose(detal3[i,:].reshape(1,-1)),
                             a2[i,:].reshape(1,-1))
        
        detal2[i,:] = np.dot(detal3[i,:].reshape(1,-1),Theta2_x)*gradSigmoid(z2[i,:])
        Theta1_grad = Theta1_grad+np.dot(np.transpose(detal2[i,:].reshape(1,-1)),
                             a1[i,:].reshape(1,-1))
    
    Theta1[:,0]=0
    Theta2[:,0]=0
    
    grad = (np.vstack((Theta1_grad.reshape(-1,1),Theta2_grad.reshape(-1,1)))+Lambda*
            np.vstack((Theta1.reshape(-1,1),Theta2.reshape(-1,1))))/m
    
    return np.ravel(grad)

#3.定义debug初始化权重函数、随机初始化权重函数和验证梯度计算是否正确
def debugInitializ_Weights(fan_in,fan_out):
    W = np.zeros((fan_out,fan_in+1))
    x = np.arange(1,fan_out*(fan_in+1)+1)
    W = np.sin(x).reshape(W.shape)/10
    return W

# 随机初始化权重theta
def randInitializeWeights(L_in,L_out):
    W = np.zeros((L_out,1+L_in))    # 对应theta的权重
    epsilon_init = (6.0/(L_out+L_in))**0.5
    W = np.random.rand(L_out,1+L_in)*2*epsilon_init-epsilon_init 
    # np.random.rand(L_out,1+L_in)产生L_out*(1+L_in)大小的随机矩阵
    return W

def checkGradient(Lambda=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    initial_theta1 = debugInitializ_Weights(input_layer_size, hidden_layer_size)
    initial_theta2 = debugInitializ_Weights(hidden_layer_size, num_labels)
    X = debugInitializ_Weights(input_layer_size-1, m)
    y = np.transpose(np.mod(np.arange(1,m+1),num_labels)).reshape(-1,1)
    
    nn_params = np.vstack((initial_theta1.reshape(-1,1),initial_theta2.reshape(-1,1)))
    '''BP求出梯度'''
    grad = nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, Lambda, X, y)
    '''使用数值法计算梯度'''
    num_grad = np.zeros((nn_params.shape[0]))
    step = np.zeros((nn_params.shape[0]))
    e=1e-4
    for i in range(nn_params.shape[0]):
        step[i] = e
        loss1 = nnCostFunction(nn_params-step.reshape(-1,1), input_layer_size, hidden_layer_size, 
                               num_labels, Lambda, X, y)
        loss2 = nnCostFunction(nn_params+step.reshape(-1,1), input_layer_size, hidden_layer_size, 
                               num_labels, Lambda, X, y)
        num_grad[i] = (loss2-loss1)/(2*e)
        step[i] = 0
    #显示两列的比较
    res = np.hstack((num_grad.reshape(-1,1),grad.reshape(-1,1)))
    # print("梯度计算的结果，第一列为数值法计算得到的，第二列为BP得到的：")
    # print(res)
    return res


    
#5.定义预测函数     
def predict(Theta1,Theta2,X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    #p = np.zeros((m,1))
    '''正向传播，预测结果'''
    X = np.hstack((np.ones((m,1)),X))
    h1 = sigmoid(np.dot(X,np.transpose(Theta1)))
    h1 = np.hstack((np.ones((m,1)),h1))
    h2 = sigmoid(np.dot(h1,np.transpose(Theta2)))
    
    '''
    返回h中每一行最大值所在的列号
    - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
    - 最后where找到的最大概率所在的列号（列号即是对应的数字）
    '''
    #np.savetxt("h2.csv",h2,delimiter=',')
    p = np.array(np.where(h2[0,:] == np.max(h2, axis=1)[0]))  
    for i in np.arange(1, m):
        t = np.array(np.where(h2[i,:] == np.max(h2, axis=1)[i]))
        p = np.vstack((p,t))
    return p    

#打开影像和训练数据
def openData():
    global image_array
    image = gdal.Open(r"./data/hiwater_xiayou_2014.tif")
    nCols = image.RasterXSize
    nRows = image.RasterYSize
    image_array = image.ReadAsArray(0,0,nCols,nRows)
    r = image_array[3,:,:]       
    g = image_array[2,:,:]    
    b = image_array[1,:,:]
    image_RGB = cv2.merge([r,g,b])
    Imax = np.max(image_array)
    Imin = np.min(image_array)
    image_RGB = ((image_RGB - Imin) * (1/(Imax-Imin)) * 255).astype('uint8')
    # plt.imshow(image_RGB)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()    
    
    #缩放影像
    scale_percent = 20       # percent of original size
    width = int(image_RGB.shape[1] * scale_percent / 100)
    height = int(image_RGB.shape[0] * scale_percent / 100)
    dim = (width, height)      
    resized = cv2.resize(image_RGB, dim, interpolation = cv2.INTER_AREA)   


    var.set('已显示')
    global img_png,label_Img
    resized = Image.fromarray(resized)
    img_png = ImageTk.PhotoImage(resized)
    label_Img = Label(root, image=img_png)
    label_Img.place(x=450,y=100)
    
    global X_train,y_train
    data = np.loadtxt("./data/hiwater_xiayou_class_practice2.txt",delimiter=",",dtype=np.float64)
    X_train=data[:,0:-1]
    y_train=data[:,-1]
    
def neuralNetwork(input_layer_size,hidden_layer_size,out_put_layer,X_train,y_train):
    X = X_train
    y = y_train
    
    m,n = X.shape
    
    Lambda = 1
    
    initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size); 
    initial_Theta2 = randInitializeWeights(hidden_layer_size,out_put_layer)
    
    initial_nn_params = np.vstack((initial_Theta1.reshape(-1,1),initial_Theta2.reshape(-1,1)))  
    #展开theta    
    #np.savetxt("testTheta.csv",initial_nn_params,delimiter=",")
    start = time.time()
    print(X.shape)
    result = optimize.fmin_cg(nnCostFunction, initial_nn_params, fprime=nnGradient, 
                              args=(input_layer_size,hidden_layer_size,out_put_layer,Lambda,X,y),
                              maxiter=100)
    print (u'执行时间：',time.time()-start)
    print (result)
    
    global Theta1,Theta2
    length = result.shape[0]
    Theta1 = result[0:hidden_layer_size*(input_layer_size+1)].reshape(
        hidden_layer_size,input_layer_size+1)
    Theta2 = result[hidden_layer_size*(input_layer_size+1):length].reshape(
        out_put_layer,hidden_layer_size+1)    

    '''预测'''
    p = predict(Theta1,Theta2,X)
    pred_r = (u"预测准确度为：%f%%"%np.mean(np.float64(p == y.reshape(-1,1))*100))
    print(pred_r)
    var.set(pred_r)
    
    # res = np.hstack((p,y.reshape(-1,1)))
    # np.savetxt("predict.csv", res, delimiter=',')

def predict_image(X,Theta1,Theta2):
    var.set('已预测')
    xShape1=X.shape[1]
    xShape2=X.shape[2]
    result_image = np.zeros((X.shape[1],X.shape[2]))
    
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    X = np.transpose(X)
    
    p = predict(Theta1,Theta2,X)
            
    result_image = p.reshape(xShape1,xShape2)
    Imax = np.nanmax(result_image)
    Imin = np.nanmin(result_image)
    result_image = ((result_image - Imin) * (1/(Imax-Imin)) * 255).astype('uint8')

    #图像及画布
    fig = plt.figure(figsize=(4.5,4),dpi=100)#图像比例
    f_plot =fig.add_subplot(111)#划分区域
    canvas_spice = FigureCanvasTkAgg(fig,root)
    canvas_spice.get_tk_widget().place(x=300,y=100)#放置位置 
    
    #缩放影像
    scale_percent = 20       # percent of original size
    width = int(result_image.shape[1] * scale_percent / 100)
    height = int(result_image.shape[0] * scale_percent / 100)
    dim = (width, height)      
    resized = cv2.resize(result_image, dim, interpolation = cv2.INTER_AREA) 
    plt.imshow(resized)
    plt.xticks([])
    plt.yticks([])
    #plt.show()  
    canvas_spice.draw()
    

#------------------------------------------------------------------------------
#组件
#1.标签
label = Label(root, text="实现遥感分类-BP神经网络",
              bg="pink",bd=10,font=("Airal",12),width=28,height=1)
label.pack(side=TOP)

#2.按钮
button1 = Button(root, text = "QUIT", command = root.destroy, 
                 activeforeground = "black",
                 activebackground = "blue", bg = "red", fg = "white")
button1.pack(side=BOTTOM)

button2 = Button(root, text="显示原始影像", command = openData,
                 activeforeground = "black",
                 activebackground = "blue", bg = "Turquoise", fg = "white")
button2.place(x=100,y=100)

button3 = Button(root, text="神经网络训练", command = lambda:neuralNetwork(input_layer_size,hidden_layer_size,out_put_layer,X_train,y_train),
                 activeforeground = "black",
                 activebackground = "blue", bg = "Turquoise", fg = "white")
button3.place(x=100,y=150)

button4 = Button(root, text="预测并显示结果", command = lambda:predict_image(image_array,Theta1,Theta2),
                 activeforeground = "black",
                 activebackground = "blue", bg = "Turquoise", fg = "white")
button4.place(x=100,y=200)


#3.菜单
def click():
    print("点击了一次")
menubar = Menu(root)
fileMenu = Menu(menubar,tearoff = 0)
fileMenu.add_command(label="新建...",command=click)
fileMenu.add_command(label="打开...",command=click)
fileMenu.add_command(label="保存...",command=click)
fileMenu.add_command(label="退出...",command=root.destroy)
menubar.add_cascade(label="文件",menu = fileMenu)
root.config(menu=menubar)

#4.创建文本窗口，显示当前操作状态
Label_show = Label(root,
                   textvariable = var,
                   bg="blue",font=("Airal",12),width=28,height=2)
Label_show.place(x=100,y=300)

#5.ComboBox
cb = ComboBox(root,label="可选地表参数（供参考）：",editable = True)
for parameter in ("NDVI","FVC","NPP","LAI"):
    cb.insert(END,parameter)
cb.pack()

#------------------------------------------------------------------------------
root.mainloop()
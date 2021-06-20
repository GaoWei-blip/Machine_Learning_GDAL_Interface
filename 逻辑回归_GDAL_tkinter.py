# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:16:04 2021

@author: gw
"""

from tkinter import *
from tkinter.tix import Tk,Control,ComboBox #升级的组合控件包
from tkinter.messagebox import showinfo,showwarning,showerror #各种消息提示框
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #画布
from matplotlib.figure import Figure


#界面初始设置
root = Tk() #初始化Tk

root.title("逻辑回归_GDAL_tkinter")
root.geometry("800x600")
root.resizable(width=True,height=True)
root.tk.eval('package require Tix')
var = StringVar()  #文本变量储存器
#------------------------------------------------------------------------------
#逻辑回归+GDAL相关函数
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14) 

#遥感影像相关库
from osgeo import gdal
import cv2
from PIL import Image,ImageTk # 导入图像处理函数库
from skimage import transform #缩放图像


#1.定义s型函数和代价函数(包含正则化)
#initial_lambda是正则化参数
def sigmoid(z):
    h = np.zeros((len(z),1))
    h = 1.0/(1.0+np.exp(-z))
    return h

def computerCost(initial_theta,X,y,initial_lambda):
    m = len(y)
    J = 0
    
    h = sigmoid(np.dot(X,initial_theta))
    theta1 = initial_theta.copy()
    theta1[0]=0
    
    temp = np.dot(np.transpose(theta1),theta1)
    J = (-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))
         +temp*initial_lambda/2)/m
          
    return J
    
#2.定义梯度下降函数
def computerGradient(initial_theta,X,y,initial_lambda):
    m=len(y)
    gradient=np.zeros((initial_theta.shape[0]))
    
    h=sigmoid(np.dot(X,initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0
    
    gradient = np.dot(np.transpose(X),h-y)/m+initial_lambda/m*theta1    
    return gradient  
    
#3.特征多项式函数
def mapFeature(X1,X2):
    degree=2
    out = np.ones((X1.shape[0],1))
    
    for i in np.arange(1,degree+1):
        for j in range(i+1):
            temp = X1**(i-j)*(X2**j)
            out = np.hstack((out,temp.reshape(-1,1)))
    return out

#4.画二维图
def plot_data(X,y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    
    plt.figure(figsize=(15,12))
    plt.plot(X[pos,0],X[pos,1],"ro")
    plt.plot(X[neg,0],X[neg,1],"bo")
    plt.title(u"两个类别散点图",fontproperties=font)
    plt.show()
    
#5.画出决策边界
def plotDecisionBoundary(theta,X,y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    
    plt.figure(figsize=(15,12))
    plt.plot(X[pos,0],X[pos,1],"ro")
    plt.plot(X[neg,0],X[neg,1],"bo")
    plt.title(u"决策边界",fontproperties=font)

    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(mapFeature(u[i].reshape(1,-1),v[j].reshape(1,-1)),theta)
    
    z = np.transpose(z)
    plt.contour(u,v,z,[0,0.01],linewidths=2.0)
    plt.show()
    
#6.预测
def predict(X,theta):
    m = X.shape[0]
    p = np.zeros((m,1))
    p = sigmoid(np.dot(X,theta))
    
    for i in range(m):
        if p[i]>0.5:
            p[i]=1
        else:
            p[i]=0
    return p

#7.逻辑回归主函数
def logisticRegression():
    data = np.loadtxt("E:/学习文档/python/机器学习/MachineLearning_Python-master/\
LogisticRegression/data2.txt",delimiter=",",dtype=np.float64)
    X=data[:,0:-1]
    y=data[:,-1]
    
    plot_data(X,y)
    
    X=mapFeature(X[:,0],X[:,1])
    initial_theta = np.zeros((X.shape[1],1))
    initial_lambda = 0.1
    
    J=computerCost(initial_theta,X,y,initial_lambda)
    print(J)
    
    # num_iters=500
    # alpha = 0.1
    # J_history,result = gradientDescent(initial_theta,X,y,initial_lambda,alpha,num_iters)
    # plot_J(J_history,num_iters)
    # print(J_history.min())
    # #print(J_history[499])
    
    result = optimize.fmin_bfgs(computerCost,initial_theta,fprime=computerGradient,
                                args=(X,y,initial_lambda))
        
    print(result)
    p = predict(X,result)
    print(u'在训练集上的准确度为%f%%'%np.mean(np.float64(p==y)*100))
    
    X=data[:,0:-1]
    y=data[:,-1]
    plotDecisionBoundary(result, X, y)
    
#测试
def testLogisticRegression():
    logisticRegression()

if __name__ == "__main__":
    testLogisticRegression()

#打开数据
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
    data = np.loadtxt("./data/hiwater_xiayou_class_practice.txt",delimiter=",",dtype=np.float64)
    X_train=data[:,0:-1]
    y_train=data[:,-1]

#------------------------------------------------------------------------------

#组件
#1.标签
label = Label(root, text="实现遥感地表参数反演-逻辑回归",
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

button3 = Button(root, text="逻辑回归", command = root.destroy,
                 activeforeground = "black",
                 activebackground = "blue", bg = "Turquoise", fg = "white")
button3.place(x=100,y=150)

button4 = Button(root, text="决策边界", command = root.destroy,
                 activeforeground = "black",
                 activebackground = "blue", bg = "Turquoise", fg = "white")
button4.place(x=100,y=200)


button4 = Button(root, text="预测并显示结果", command = lambda:logisticRegression(),
                 activeforeground = "black",
                 activebackground = "blue", bg = "Turquoise", fg = "white")
button4.place(x=100,y=250)

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
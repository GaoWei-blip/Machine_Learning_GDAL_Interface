# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:56:13 2021

@author: 11350
"""

from tkinter import *
from tkinter.tix import Tk,Control,ComboBox #升级的组合控件包
from tkinter.messagebox import showinfo,showwarning,showerror #各种类型的提示框
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


root = Tk() #初始化Tk()

#root是布局的根节点，以后的布局都在它之上
root.title("hello tkinter") #设置窗口标题
root.geometry("800x600") #设置窗口大小
root.resizable(width = True, height = True) #设置是否可以变换窗口大小，默认True
root.tk.eval('package require Tix')      #引入升级包，这样才能使用升级的组合控件
var = StringVar()    # 这时文字变量储存器

#标签
label = Label(root,text = "实现遥感地表参数的线性回归",
              bg = "pink",bd = 10,font = ("Arial",12),width = 28,height = 1)
label.pack(side=TOP)


#------------------------------------------------------------------------------
#线性回归+GDAL 相关函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = r"c:\windows\fonts\simsun.ttc",size=14)
from osgeo import gdal
import cv2
from PIL import Image,ImageTk # 导入图像处理函数库
from skimage import transform #缩放图像

def computerCost(X,y,theta):
    m=len(y)
    J=0
    
    J= np.dot(np.transpose(np.dot(X,theta)-y),np.dot(X,theta)-y)/(2*m)
    return J

def GradientDescent(X,y,theta,alpha,num_iters):
    global J_history
    m=len(y)
    n=len(theta)
    
    temp = np.matrix(np.zeros((n,num_iters)))
    J_history = np.zeros((num_iters,1))
    
    for i in range(num_iters):
        temp[:,i] = theta - (alpha/m)*np.dot(np.transpose(X),np.dot(X,theta)-y)
        theta = temp[:,i]
        J_history[i] = computerCost(X,y,theta)
    
    return J_history,theta

def dataNormalize(X):
    Xmin = np.min(X,axis=1).reshape(-1,1)
    Xmax = np.max(X,axis=1).reshape(-1,1)
    X_norm = (X-Xmin)/(Xmax-Xmin)
    return X_norm

def plot_X1_X2(X):   
    X_norm=dataNormalize(X)
    
    var.set('已归一化')
    label_Img.config(image='') 
    
    #图像及画布
    fig = plt.figure(figsize=(4.5,4),dpi=100)#图像比例
    f_plot =fig.add_subplot(111)#划分区域
    canvas_spice = FigureCanvasTkAgg(fig,root)
    canvas_spice.get_tk_widget().place(x=300,y=100)#放置位置
    
    X1=X_norm[:,1]
    X2=X_norm[:,2]
    #plt.scatter(X1,X2)
    plt.title(u"归一化后两类特征结果图",fontproperties=font)
    plt.xlabel(u"X1",fontproperties=font,labelpad=2.5)
    plt.ylabel(u"X2",fontproperties=font,labelpad=0.5)
    plt.scatter(X1,X2)
    #plt.show()    
    plt.grid(True)#网格  
    canvas_spice.draw()
           
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
    data = pd.read_csv('.data/hiwater_xiayou_practice.txt')
    data = np.array(data)
    X_train = data[:,2:8]
    y_train = data[:,9].reshape(-1,1)
    
def LinearRegression(X,y,alpha,num_iters):
    var.set('已线性回归')
    global theta
    X = dataNormalize(X)
    X = np.hstack((np.ones((X.shape[0],1)),X))
    
    theta = np.zeros((X.shape[1],1))
    
    J_history,theta = GradientDescent(X, y, theta, alpha,num_iters)
    
def plotJ(J_history,num_iters):
    var.set('已绘图')
    
    #图像及画布
    fig = plt.figure(figsize=(4.5,4),dpi=100)#图像比例
    f_plot =fig.add_subplot(111)#划分区域
    canvas_spice = FigureCanvasTkAgg(fig,root)
    canvas_spice.get_tk_widget().place(x=300,y=100)#放置位置
    
    x = np.arange(1,num_iters+1)
    plt.plot(x,J_history)
    plt.title(u'代价值随迭代数的变化',fontproperties = font)
    plt.xlabel(u'迭代数',fontproperties = font)
    plt.ylabel(u'代价值',fontproperties = font)
    #plt.show()
    plt.grid(True)#网格  
    canvas_spice.draw()

def predict(X,theta):
    global result_image
    var.set('已预测')
    result_image = np.zeros((X.shape[1],X.shape[2]))
    
    X1 = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    X1 = np.transpose(X1)
    X1 = dataNormalize(X1)
    X1 = np.hstack((np.ones((X1.shape[0],1)),X1))
    result_temp = np.dot(X1,theta)
    result_image = result_temp.reshape(X.shape[1],X.shape[2])
    
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
#BUTTON
button1=Button(root,text='QUIT',command=root.destroy,activeforeground="black",
               activebackground='blue',bg='red',fg='white')
button1.pack(side=BOTTOM)

button2=Button(root,text='打开并显示原始影像',command=openData,
               activeforeground="black",activebackground='blue',
               bg='Turquoise',fg='white')
button2.place(x=100,y=100)

button3=Button(root,text='训练集归一化并绘图',command=lambda:plot_X1_X2(X_train),
               activeforeground="black",activebackground='blue',
               bg='Turquoise',fg='white')
button3.place(x=100,y=150)

button4=Button(root,text='线性回归',
               command=lambda:LinearRegression(X_train,y_train,0.1,500),
               activeforeground="black",activebackground='blue',
               bg='Turquoise',fg='white')
button4.place(x=100,y=200)

button5=Button(root,text='代价值随迭代数变化',command=lambda:plotJ(J_history,500),
               activeforeground="black",activebackground='blue',
               bg='Turquoise',fg='white')
button5.place(x=100,y=250)

button6=Button(root,text='预测并显示结果',command=lambda:predict(image_array,theta),
               activeforeground="black",activebackground='blue',
               bg='Turquoise',fg='white')
button6.place(x=100,y=300)

#ComboBox
cb=ComboBox(root,label='可选地表参数(供参考):',editable=True)
for animal in ('NDVI','FVC','NPP','LAI'):
    cb.insert(END,animal)
cb.pack()

#Menu
def click():
    print("点击了一次")
menubar=Menu(root)
filemenu=Menu(menubar,tearoff=0)
filemenu.add_command(label='新建...',command=click)
filemenu.add_command(label='打开...',command=click)
filemenu.add_command(label='保存',command=click)
filemenu.add_command(label='退出',command=root.destroy)
menubar.add_cascade(label='文件',menu=filemenu)
root.config(menu = menubar)

# 创建文本窗口，显示当前操作状态
Label_Show = Label(root,
    textvariable=var,   # 使用 textvariable 替换 text, 因为这个可以变化
    bg='blue', font=('Arial', 12), width=15, height=2)
Label_Show.place(x=100,y=350)

#运行主程序，出界面
root.mainloop()
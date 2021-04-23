# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:56:13 2021

@author: 11350
"""

from tkinter import *
from tkinter.tix import Tk,Control,ComboBox #升级的组合控件包
from tkinter.messagebox import showinfo,showwarning,showerror #各种类型的提示框


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

def GradientDescent(X,y,theta,alpha):
    m=len(y)
    theta = np.zeros((X.shape[1],1))
    for i in range(X.shape[0]):
        theta = temp[i,:]
        temp[i,:] = theta - (alpha/m)*np.dot(np.transpose(X),np.dot(X,theta)-y)
        J_history[i] = computerCost(X,y,theta)
    
    return J_history,theta

def dataNormalize(X):
    Xmin = np.min(X,axis=1).reshape(-1,1)
    Xmax = np.max(X,axis=1).reshape(-1,1)
    X_norm = (X-Xmin)/(Xmax-Xmin)
    
    X1=X_norm[:,1]
    X2=X_norm[:,2]
    plt.scatter(X1,X2)
    plt.title(u"归一化后两类特征结果图",fontproperties=font)
    plt.xlabel(u"X1",fontproperties=font)
    plt.ylabel(u"X2",fontproperties=font)
    plt.show()
    
    return X_norm

def plot_X1_X2(X_norm):
    
    X1=X_norm[:,1]
    X2=X_norm[:,2]
    plt.scatter(X1,X2)
    plt.title(u"归一化后两类特征结果图",fontproperties=font)
    plt.xlabel(u"X1",fontproperties=font)
    plt.ylabel(u"X2",fontproperties=font)
    plt.show()
    
def openData():
    image = gdal.Open(r"hiwater_xiayou_2014.tif")
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
    global img_png
    resized = Image.fromarray(resized)
    img_png = ImageTk.PhotoImage(resized)
    label_Img = Label(root, image=img_png)
    label_Img.place(x=450,y=100)
#------------------------------------------------------------------------------
#BUTTON
button1=Button(root,text='QUIT',command=root.destroy,activeforeground="black",
               activebackground='blue',bg='red',fg='white')
button1.pack(side=BOTTOM)

button2=Button(root,text='打开并显示原始影像',command=openData,
               activeforeground="black",activebackground='blue',
               bg='Turquoise',fg='white')
button2.place(x=100,y=100)

button3=Button(root,text='数据归一化并绘图',command=root.destroy,
               activeforeground="black",activebackground='blue',
               bg='Turquoise',fg='white')
button3.place(x=100,y=150)

button4=Button(root,text='线性回归',command=root.destroy,
               activeforeground="black",activebackground='blue',
               bg='Turquoise',fg='white')
button4.place(x=100,y=200)

button5=Button(root,text='代价值随迭代数变化',command=root.destroy,
               activeforeground="black",activebackground='blue',
               bg='Turquoise',fg='white')
button5.place(x=100,y=250)

button6=Button(root,text='预测并显示结果',command=root.destroy,
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
filemenu.add_command(label='关闭填写',command=root.destroy)
menubar.add_cascade(label='文件',menu=filemenu)
root.config(menu = menubar)

# 创建文本窗口，显示当前操作状态
Label_Show = Label(root,
    textvariable=var,   # 使用 textvariable 替换 text, 因为这个可以变化
    bg='blue', font=('Arial', 12), width=15, height=2)
Label_Show.place(x=100,y=350)

#运行主程序，出界面
root.mainloop()
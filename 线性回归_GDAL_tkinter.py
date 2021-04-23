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
root.geometry("800x500") #设置窗口大小
root.resizable(width = True, height = True) #设置是否可以变换窗口大小，默认True
root.tk.eval('package require Tix')      #引入升级包，这样才能使用升级的组合控件
var = StringVar()    # 这时文字变量储存器

#标签
label = Label(root,text = "实现遥感地表参数的线性回归",
              bg = "pink",bd = 10,font = ("Arial",12),width = 28,height = 2)
label.pack(side=TOP)

#------------------------------------------------------------------------------
#线性回归+GDAL 相关函数
import n
def computerCost(X,y,theta):
    m=len(y)
    J=0
    
    J=np.dot(X,theta)
    
#------------------------------------------------------------------------------
#BUTTON
button1=Button(root,text='QUIT',command=root.destroy,activeforeground="black",activebackground='blue',
              bg='red',fg='white')
button1.pack(side=BOTTOM)

button2=Button(root,text='打开并显示原始影像',command=root.destroy,
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
root.config(menu=menubar)
filemenu=Menu(menubar,tearoff=0)
menubar.add_cascade(label='文件',menu=filemenu)
filemenu.add_command(label='新建...',command=click())
filemenu.add_command(label='打开...',command=click())
filemenu.add_command(label='保存',command=click())
filemenu.add_command(label='关闭填写',command=root.destroy)

# 创建文本窗口，显示当前操作状态
Label_Show = Label(root,
    textvariable=var,   # 使用 textvariable 替换 text, 因为这个可以变化
    bg='blue', font=('Arial', 12), width=15, height=2)
Label_Show.place(x=100,y=350)

#运行主程序，出界面
root.mainloop()
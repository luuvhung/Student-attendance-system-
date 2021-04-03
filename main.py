import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
from tkinter import messagebox 
import testsvc
import themkhuonmat
import training
window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("DACN")
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
window.geometry('850x640')
window.configure(background='white')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(window, text="HỆ THỐNG ĐIỂM DANH" ,bg="black"  ,fg="white"  ,width=87  ,height=3,font=('times', 30, 'bold')) 

message.place(x=-600, y=20)

lbl = tk.Label(window, text="MSSV",width=20  ,height=2  ,fg="black"  ,font=('times', 15, ' bold ') ) 
lbl.place(x=0, y=275)

txt = tk.Entry(window,width=20   ,fg="black",font=('times', 15, ' bold '))
txt.place(x=300, y=280)

lbl2 = tk.Label(window, text="NHẬP TÊN",width=20  ,fg="black"  ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=0, y=350)

txt2 = tk.Entry(window,width=20 ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=300, y=365)


lbl5 = tk.Label(window, text="THÊM SINH VIÊN",width=20  ,fg="white"  ,bg="black"  ,height=2 ,font=('times', 15, ' bold ')) 
lbl5.place(x=0, y=200)

message3 = tk.Label(window, text="" ,fg="red"   ,bg="white",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message3.place(x=500, y=280)

message4 = tk.Label(window, text="" ,fg="red"   ,bg="white",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message4.place(x=500, y=355)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False 

def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res) 

def addSV1():
    Id=(txt.get())
    name=(txt2.get())
    if(Id==""):
        res1 = "Vui lòng nhập MSSV"
    elif(is_number(Id)==False):
        res1 = "MSSV là số"
    else:
        res1 = ""
    message3.configure(text=res1)
    if (name==""):
        res2 = "Vui lòng nhập tên"
    elif(name.isalpha()==False):
        res2 = "Tên là chữ"
    else:
        res2 = ""
    message4.configure(text= res2)
    themkhuonmat.addSV(Id,name)
    messagebox.showinfo("Thông báo","Đã lưu xong dữ liệu")

def training1():
    training.train()
    messagebox.showinfo("Thông báo","Đã Train xong")

def diemdanh():
    testsvc.diemdanh()

trackImg = tk.Button(window, text="ĐIỂM DANH", command=diemdanh ,fg="white"  ,bg="black"  ,width=20  ,height=3, activebackground = "white" ,font=('times', 15, ' bold '))
trackImg.place(x=600, y=500)
takeImg = tk.Button(window, text="LẤY ẢNH", command=addSV1 ,fg="white"  ,bg="black"  ,width=20  ,height=3, activebackground = "white" ,font=('times', 15, ' bold '))
takeImg.place(x=0, y=500)
trainImg = tk.Button(window, text="TRAIN", command=training1  ,fg="white"  ,bg="black"  ,width=20  ,height=3, activebackground = "white" ,font=('times', 15, ' bold '))
trainImg.place(x=300, y=500)
window.mainloop()
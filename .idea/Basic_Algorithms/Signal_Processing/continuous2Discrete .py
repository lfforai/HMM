import matplotlib.pyplot as plt
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc")
import numpy as np
from base_class import paint_linear as pl #自定义绘图类
from base_class import *

#一、test paint_class
o=pl("cos and sin")
# o([[0,2*np.pi,50],[-np.pi,np.pi,50]],[[np.sin,"sin(x)"],[np.cos,"cos(x)"]])

#二、第二章第5节由离散信号恢复到连续信号（本质是插值拟合）,提供3中插值方式进行比较
# if self.g_o=="sinc":
# elif self.g_o=="triangle":
# elif self.g_o=="square":
o_d=discrete2continuity(xrange=[-20,20],num=50,x_fun=np.sin,g_fun="sinc")
x_range=np.linspace(-20,20,200)
y_fit=o_d(x_range)         #由离散插值出的拟合函数
y_rel=list(np.sin(x_range))#真实函数
o=pl()
o([x_range,x_range],[[y_fit,"fit"],[y_rel,"sin"]],x_type="x_o_y_o")

#三、
# deta=10/999
# print(1/deta)
# o=pl("cos and sin")
# def fun(x):
#     return np.sin(2*np.pi*(1000+0.5)*deta*x)/np.sin(np.pi*deta*x)
# o([[-5,5,1000]],[[fun,"cos(x)"]])


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
#说明:混迭和吉斯现象
#sinc在频域上不产生混迭的间隔不能小于1，左右各0.5的截断图像
#1、时域上采样
dx=10  #在时地域上的
ds=20 #在频率上的
len=100    #时域上的采样个数N，必须是一个偶数
len2=2000 #频域上的采样个数M，必须是一个偶数
x=[]
fre=len/(2.0*dx)
for i in range(len+1):
    x.append(-1.0*dx+i/fre)
x=np.array(x)
y1=np.sinc(x)
print("在时域上的采样间隔dx：",1.0/fre)
print("频域上的间隔：",fre)

#2、频域上采样
s=[]
for i in range(len2+1):
    s.append(-ds+i*2*ds/len2)
a=np.zeros(len+1)
y2=[]
for i in range(s.__len__()):
    b=-2.0*np.pi*x*s[i]
    y=np.exp(a+1j*b)
    y2.append(np.sum(y1*y).real/fre)

#3、绘图
plt.plot(s, y2)
#有截断的sinc
# plt.plot(x,y1)
plt.show()
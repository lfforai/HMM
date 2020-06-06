import numpy as np
import matplotlib.pyplot as plt

#理想滤波器章节，视窗函数曲线图
def paint_time(func,len,range_x,text="",index=1,pltopen=True):
    if len%2==0:
       pass
    else:
       print("len must be 偶数")
       return -1
    x=[]
    dx=(2.0*range_x)/len #采样间隔
    print("采样间隔:",dx)
    for i in range(len+1):
        x.append(-1.0*range_x+i*dx)
    y=func(x)
    if pltopen==True:
        plt.figure(index) # 创建图表1
        plt.title(text)
        plt.plot(x, y)
    plt.savefig("./pitcture/"+text+".jpg")
    plt.close(index)
    return np.array(y),np.array(x),dx

def paint_scatter(x,y,index=1):
    plt.figure(index) # 创建图表1
    plt.scatter(x,y)

#一、矩形视窗，参数T=0.5
#rectangle
#时域
T=5
len=1000
range_x=10
range_fre=10
def rectangle_time_fun(x,T_in=T):
    result=[]
    for e  in  x:
        if e>T_in or e<-T_in:
           result.append(0)
        else:
           result.append(1)
    return result
index=1
paint_time(rectangle_time_fun,len,range_x,"rectangle_time",index)
#频谱
def rectangle_fre_fun(x,T_in=T):#当T=0.5时候等于np.sinc
    result=[]
    for e  in  x:
      if e==0:
        result.append(2*T_in)
      else:
        result.append(np.sin(2.0*np.pi*e*T_in)/(np.pi*e))
    return result
index=1+index
paint_time(rectangle_fre_fun,len,range_fre,"rectangle_fre",index)
# dx=(2.0*range_x)/len #采样间隔
# fN=[-1.0/((len+1)*dx),1.0/((len+1)*dx)]
# y=[0,0]
# paint_scatter(fN,y,index)

#二、矩形视窗，参数T=0.5
#triangle
#时域
def triangle_time_fun(x,T_in=T):
    result=[]
    for e  in  x:
        if e>T_in or e<-T_in:
            result.append(0)
        else:
            result.append(1-np.abs(e)/T_in)
    return result
index=1+index
paint_time(triangle_time_fun,len,range_x,"tectangle_time",index)
#频谱
def triangle_fre_fun(x,T_in=T):#当T=0.5时候等于np.sinc
    result=[]
    for e  in  x:
        if e==0:
            result.append(T_in)
        else:
            result.append(T_in*np.power(np.sin(np.pi*e*T_in)/(np.pi*e*T_in),2))
    return result
index=1+index
paint_time(triangle_fre_fun,len,range_fre,"tectangle_fre",index)

#三、钟型视窗，参数T=0.5
# Bell
#时域
beta=150.0 #4-7范围
def Bell_time_fun(x,T_in=T,beta=beta):
    result=[]
    for e  in  x:
        if e>T_in or e<-T_in:
            result.append(0)
        else:
            result.append(np.exp(-beta*np.power(e/T_in,2)))
    return result
index=1+index
paint_time(Bell_time_fun,len,range_x,"bell_time",index)
#频谱
def Bell_fre_fun(x,T_in=T,beta=beta):#当T=0.5时候等于np.sinc
    result=[]
    for e  in  x:
        result.append(np.sqrt(np.pi)*np.power(T_in,2)/beta*np.exp(-np.power(np.pi*np.power(T_in,2)*e,2)/beta))
    return result
index=1+index
paint_time(Bell_fre_fun,len,range_fre,"bell_fre",index)


#四、哈宁视窗，参数T=0.5
#Hanning
#时域
beta=150.0 #4-7范围
def Hanning_time_fun(x,T_in=T):
    result=[]
    for e  in  x:
        if e>T_in or e<-T_in:
            result.append(0)
        else:
            result.append((1+np.cos(e*np.pi/T_in))*0.5)
    return result
index=1+index
paint_time(Hanning_time_fun,len,range_x,"Hanning_time",index)
#频谱
def Hanning_fre_fun(x,T_in=T):#当T=0.5时候等于np.sinc
    result=[]
    for e  in  x:
        if e==0:
           result.append(T/(1-np.power(2*T_in*e,2.0)))
        else:
           #方法1
           # result.append(0.5*rectangle_fre_fun(list([e]),T_in)[0]+0.25*rectangle_fre_fun(list([e-1/(2*T_in)]),T_in)[0]+0.25*rectangle_fre_fun(list([e+1/(2*T_in)]),T_in)[0])
           #方法2,简化
           result.append(np.sin(2*np.pi*e*T_in)/(2*np.pi*e)/(1-np.power(2*T_in*e,2)))
    return result
index=1+index
paint_time(Hanning_fre_fun,len,range_fre,"Hanning_fre",index)
dx=(2.0*range_x)/len #采样间隔
fN=[-1.0/((len+1)*dx),1.0/((len+1)*dx)]
y=[0,0]
# paint_scatter(fN,y,index)

#五、汗明视窗，参数T=0.5
#Hamming
#时域
def Hamming_time_fun(x,T_in=T):
    result=[]
    for e  in  x:
        if e>T_in or e<-T_in:
            result.append(0)
        else:
            result.append(0.54+0.46*np.cos(np.pi*e/T_in))
    return result
index=1+index
paint_time(Hamming_time_fun,len,range_x,"Hamming_time",index)
#频谱
def Hamming_fre_fun(x,T_in=T):#当T=0.5时候等于np.sinc
    result=[]
    for e  in  x:
        if e==0:
            result.append(2*T_in*(0.54-0.08*np.power(2*T_in*e,2))/(1-np.power(2*T_in*e,2)))
        else:
            #方法1
            # result.append(0.5*rectangle_fre_fun(list([e]),T_in)[0]+0.25*rectangle_fre_fun(list([e-1/(2*T_in)]),T_in)[0]+0.25*rectangle_fre_fun(list([e+1/(2*T_in)]),T_in)[0])
            #方法2,简化
            result.append(np.sin(2*np.pi*e*T_in)/(np.pi*e)*(0.54-0.08*np.power(2*T_in*e,2))/(1-np.power(2*T_in*e,2)))
    return result
index=1+index
paint_time(Hamming_fre_fun,len,range_fre,"Hamming_fre",index)
dx=(2.0*range_x)/len #采样间隔
fN=[-1.0/((len+1)*dx),1.0/((len+1)*dx)]
y=[0,0]
# paint_scatter(fN,y,index)

#五、帕曾视窗，参数T=0.5
#时域
beta=150.0 #4-7范围
def Pz_time_fun(x,T_in=T):
    result=[]
    for e  in  x:
        if np.abs(e)>T_in:
            result.append(0)
        elif np.abs(e)<=T_in and np.abs(e)>T_in/2:
            result.append(2*np.power(1-np.abs(e)/T_in,3))
        else:
            result.append(1-6*np.power(np.abs(e)/T_in,2)+6*np.power(np.abs(e)/T_in,3))
    return result
index=1+index
paint_time(Pz_time_fun,len,range_x,"Pz_time",index)
#频谱
def Pz_fre_fun(x,T_in=T):#当T=0.5时候等于np.sinc
    result=[]
    for e  in  x:
        if e==0:
            result.append(3.0/4.0)
        else:
            #方法1
            # result.append(0.5*rectangle_fre_fun(list([e]),T_in)[0]+0.25*rectangle_fre_fun(list([e-1/(2*T_in)]),T_in)[0]+0.25*rectangle_fre_fun(list([e+1/(2*T_in)]),T_in)[0])
            #方法2,简化
            result.append((3.0/4.0)*np.power(np.sin(np.pi*e*T_in/2)/(np.pi*e*T_in/2),4))
    return result
index=1+index
paint_time(Pz_fre_fun,len,range_fre,"Pz_fre",index)
dx=(2.0*range_x)/len #采样间隔
fN=[-1.0/((len+1)*dx),1.0/((len+1)*dx)]
y=[0,0]
# paint_scatter(fN,y,index)


#六、丹尼尔视窗，参数T=0.5
#时域
beta=150.0 #4-7范围
def dne_time_fun(x,T_in=T):
    result=[]
    for e  in  x:
        if np.abs(e)>T_in:
            result.append(0)
        elif e==0:
            result.append(1)
        else:
            result.append(np.sin(np.pi*e/T_in)/(np.pi*e/T_in))
    return result
index=1+index
paint_time(dne_time_fun,len,range_x,"dne_time",index)
#频谱
def dne_fre_fun(x,T_in=T):#当T=0.5时候等于np.sinc
    result=[]
    for e  in  x:
        if np.abs(e)<=1/(2*T_in):
            result.append(T_in)
        else:
            result.append(0)
    return result
index=1+index
paint_time(dne_fre_fun,len,range_fre,"dne_fre",index)
dx=(2.0*range_x)/len #采样间隔
fN=[-1.0/((len+1)*dx),1.0/((len+1)*dx)]
y=[0,0]
# paint_scatter(fN,y,index)

#七、blackman视窗，参数T=0.5
#时域
beta=150.0 #4-7范围
def blackman_time_fun(x,T_in=T):
    result=[]
    for e  in  x:
        if np.abs(e)>T_in:
            result.append(0)
        else:
            result.append(0.42+0.5*np.cos(np.pi*e/T_in)+0.08*np.cos(2.0*np.pi*e/T_in))
    return result
index=1+index
paint_time(blackman_time_fun,len,range_x,"blackman_time",index)
#频谱
def blackman_fre_fun(x,T_in=T):#当T=0.5时候等于np.sinc
    result=[]
    for e  in  x:
        result.append(0.42*rectangle_fre_fun(list([e]),T_in)[0]\
                  +0.25*rectangle_fre_fun(list([e-1/(2*T_in)]),T_in)[0]+0.25*rectangle_fre_fun(list([e+1/(2*T_in)]),T_in)[0]\
                  +0.04*rectangle_fre_fun(list([e-1/T_in]),T_in)[0]+0.04*rectangle_fre_fun(list([e+1/T_in]),T_in)[0])
    return result
index=1+index
paint_time(blackman_fre_fun,len,range_fre,"blackman_fre",index)
dx=(2.0*range_x)/len #采样间隔
fN=[-1.0/((len+1)*dx),1.0/((len+1)*dx)]
y=[0,0]
# paint_scatter(fN,y,index)


#八、Kaiser视窗，参数T=0.5
#时域
beta=150.0 #4-7范围
def blackman_time_fun(x,T_in=T):
    result=[]
    for e  in  x:
        if np.abs(e)>T_in:
            result.append(0)
        else:
            result.append(0.42+0.5*np.cos(np.pi*e/T_in)+0.08*np.cos(2.0*np.pi*e/T_in))
    return result
index=1+index
paint_time(blackman_time_fun,len,range_x,"Kaiser_time",index)
#频谱
def blackman_fre_fun(x,T_in=T):#当T=0.5时候等于np.sinc
    result=[]
    for e  in  x:
        result.append(0.42*rectangle_fre_fun(list([e]),T_in)[0] \
                      +0.25*rectangle_fre_fun(list([e-1/(2*T_in)]),T_in)[0]+0.25*rectangle_fre_fun(list([e+1/(2*T_in)]),T_in)[0] \
                      +0.04*rectangle_fre_fun(list([e-1/T_in]),T_in)[0]+0.04*rectangle_fre_fun(list([e+1/T_in]),T_in)[0])
    return result
index=1+index
paint_time(blackman_fre_fun,len,range_fre,"Kaiser_fre",index)
dx=(2.0*range_x)/len #采样间隔
fN=[-1.0/((len+1)*dx),1.0/((len+1)*dx)]
y=[0,0]
# # plt.show()

# 九、四个理想滤波器图像
len=500
range_x=10
index=index+1
#理想滤波器
f1=1.0
f2=2.5
range_s=4
#理想低通滤波器
def lowpass_fre_fun(x,f1=f1,range_s=range_s):
    result=[]
    for e  in  x:
        if np.abs(e)<f1:
            result.append(1.0)
        else:
            result.append(0.0)
    return result

#理想高通滤波器
def highpass_fre_fun(x,f1=f1,range_s=range_s):
    result=[]
    for e  in  x:
        if np.abs(e)>range_s:
            result.append(0.0)
        elif np.abs(e)<=range_s and np.abs(e)>f1:
            result.append(1.0)
        else:
            result.append(0.0)
    return result
#理想带通滤波器
def band_fre_fun(x,f1=f1,f2=f2,range_s=range_s):
    result=[]
    for e  in  x:
        if np.abs(e)<=f2 and np.abs(e)>=f1:
            result.append(1.0)
        else:
            result.append(0.0)
    return result

def band_time_fun(x,f1=f1,f2=f2,range_s=range_s):
    result=[]
    for e in x:
        if e==0:
          result.append(2.0*(f2-f1))
        else:
           result.append(2*np.sin((f2-f1)*np.pi*e)*np.cos(np.pi*(f1+f2)*e)/(np.pi*e))
    return np.array(result)
#理想阻带滤波器
def stopband_fre_fun(x,f1=f1,f2=f2,range_s=range_s):
    result=[]
    for e  in  x:
        if np.abs(e)<f1:
            result.append(1.0)
        elif np.abs(e)<range_s and np.abs(e)>f2:
            result.append(1.0)
        else:
            result.append(0.0)
    return result
#绘制频谱上的函数图
plotopen=True
index=index+1
y_low_time,x,dx=paint_time(lowpass_fre_fun,len,range_x,"low_pass_fre",index,plotopen)
index=index+1
y_high_time,x,dx=paint_time(highpass_fre_fun,len,range_x,"high_pass_fre",index,plotopen)
index=index+1
y_band_time,x,dx=paint_time(band_fre_fun,len,range_x,"band_pass_fre",index,plotopen)
index=index+1
y_stopband_time,x,dx=paint_time(stopband_fre_fun,len,range_x,"stopband_pass_fre",index,plotopen)
#绘制频域上的函数图
index=index+1
y_sinc,x,dx=paint_time(np.sinc,len,range_x,"low_pass_time",index,True)
index=index+1
y_band,x,dx=paint_time(band_time_fun,len,range_x,"band_pass_time",index,True)


#计算离散信号下的连续频谱函数
def fre(y_time,x,dx,fre_len,range_fre,text,index,fltopen=True):
    #x是在时域上离散采样序列，dx时时序的采样间隔
    #range_fre是频域上的范围
    print(text+"在频域上的周期:",1/dx)
    s=[]
    d_fre=2*range_fre/fre_len
    for i in range(fre_len+1):
        s.append(-1.0*range_fre+i*d_fre)
    a=np.zeros(x.__len__())#构造复数
    fre_y=[] #频域上的近似连续频谱的离散频域值（绘图用）
    for i in range(s.__len__()):
        b=-2.0*np.pi*x*s[i]
        y_fre=np.exp(a+1j*b)
        fre_y.append(np.sum(y_time*y_fre).real*dx)
    plt.figure(index) # 创建图表1
    plt.title(text)
    plt.plot(s,fre_y)
    plt.savefig("./pitcture/"+text+".jpg")
index=index+1
fre(y_sinc,x,dx,2000,10,"windows_Of_rectangle_to_lowpass_fre",index)
index=index+1
fre(y_band,x,dx,2000,10,"windows_Of_rectangle_to_band_fre",index)


#十、使用其他窗口对原理想信号进行加窗
def add_windows_in_time(func,win,len,range_x,text="",index=1):
    if len%2==0:
        pass
    else:
        print("len must be 偶数")
        return -1
    x=[]
    dx=(2.0*range_x)/len #采样间隔
    print("采样间隔:",dx)
    for i in range(len+1):
        x.append(-1.0*range_x+i*dx)
    y=func(x)
    y_win=win(x)
    y=y*y_win
    # plt.savefig("./pitcture/"+text+".jpg")
    return np.array(y),np.array(x),dx

#三角时窗对低筒滤波器加窗
index=index+1
y_sinc,x,dx=add_windows_in_time(np.sinc,triangle_time_fun,len,range_x,"",index)
fre(y_sinc,x,dx,2000,10,"windows_Of_tectangle_to_lowpass_fre",index)

#blackman时窗对低通滤波器加窗
index=index+1
y_sinc,x,dx=add_windows_in_time(np.sinc,blackman_time_fun,len,range_x,"",index)
fre(y_sinc,x,dx,2000,10,"windows_Of_blackman_to_lowpass_fre",index)

#dan时窗对带通滤波器加窗
index=index+1
y_sinc,x,dx=add_windows_in_time(np.sinc,dne_time_fun,len,range_x,"",index)
fre(y_sinc,x,dx,2000,10,"windows_Of_dne_to_lowpass_fre",index)

#三角时窗对带通滤波器加窗
index=index+1
y_band,x,dx=add_windows_in_time(band_time_fun,triangle_time_fun,len,range_x,"",index)
fre(y_band,x,dx,2000,10,"windows_Of_tectangle_to_band_fre",index)

#blackman时窗对带通滤波器加窗
index=index+1
y_band,x,dx=add_windows_in_time(band_time_fun,blackman_time_fun,len,range_x,"",index)
fre(y_band,x,dx,2000,10,"windows_Of_blackman_to_band_fre",index)

#dan时窗对带通滤波器加窗
index=index+1
y_band,x,dx=add_windows_in_time(band_time_fun,dne_time_fun,len,range_x,"",index)
fre(y_band,x,dx,2000,10,"windows_Of_dne_to_band_fre",index)
plt.show()


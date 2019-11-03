#该模型是基于lstm预测值与实际值偏差在【avg-std*3，avg+std*3】范围以外作为异常值得判断，每BATCH_SIZE一组计算方差和均值，其中范围标准差倍数需要设置
from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
#os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] ='4'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
#tf.debugging.set_log_device_placement(True)
#tf.config.set_soft_device_placement(True)
import matplotlib.pyplot as plt

#os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT']='4'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import contextlib
# 构建包含上下文管理器的函数，使其可以在with中使用
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}: {}'.format(error_class, e))
    except Exception as e:
        print('Got unexpected exception \n  {}: {}'.format(type(e), e))
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(
            error_class))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import math
print(tf.__version__)

#离散傅里叶和傅里叶变换的关系，由于离散傅里叶实际上等于在时序上对原信号进行了一次采样，
# 然后通过傅里叶变换，采样函数的傅里叶变换以后，在频域上形成了一个周期函数，
# 这次又进行了一次在频域上的采样，为了保证只取一个周期里面的样本，所以对周期1/T,进行了一次切割，
# 按照取样个数 M,把周期切割成了 1/T/M,份，所以 u=1/T/M，2/T/M，直到（M-1)/T/M ,m=1,2...M-1,
# 带入到在时域上第一次取样后的傅里叶变换后的函数，就有了经典的离散傅里叶公式

#离散傅里叶*周期T 近似=原连续的信号的连续傅里叶变换

#实验表明对于离散采样点，傅里叶可以进行插值计算完美还原模型信号，但是代价是采样点足够多，采样频率足够高
#原始曲线
x=np.arange(-10,10,0.01)
a=np.sin(3.1415926*x)

#按周期为T进行的采样，并用傅里叶重构的函数
T=2.5  #采样周期不能刚好是2*pi的倍数，不然sin（2*pi*x）=0.采样无意义
sample_num=800
f_resample=[]
for i in range(2000):
    t=x[i]
    n=np.arange(-sample_num,sample_num,1)
    f_resample.append(np.dot(np.sin(3.1415926*n*T),np.sinc((t-n*T)/T)))
y_resample=np.array(f_resample)
plt.plot(x,y_resample,'r') #原曲线
plt.plot(x,a,'b')          #傅里叶插值复原函数：sum(f(n*T)*sinc((t-nT)/T))
plt.show()
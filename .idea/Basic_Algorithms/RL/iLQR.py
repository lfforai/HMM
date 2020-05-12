from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import gym
import numpy as np
import random
import os
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import layers

# test hessians矩阵
x = tf.random.normal((100,2),mean=2.0, stddev=3.0)
@tf.function
def compute_b(xy):
    #     y                x                    y
    z = xy[1]*tf.math.pow(xy[0],2)+tf.math.pow(xy[1],2)
    hes = tf.hessians(z,xy)
    return hes
# for i in range(100):
#     print(compute_b(x[i,:]))

#构造神经网
env_world=gym.make("CartPole-v1") #

class Linear(layers.Layer):
    def __init__(self, name="Linear",units=32,training=True,**kwargs):
        super(Linear, self).__init__(name=name,**kwargs)
        self.units = units
        self.training=training

    def build(self, input_shape):#只有在运行完call以后可以构造
        self.w = self.add_weight(name="w",shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=self.training)

        self.b = self.add_weight(name="b",shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=self.training)
        # super(Linear, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

#跳转st+1=f(st，at)
#难点在于st+1的输出有限制范围
#自定义loss
# class selfLoss(tf.keras.losses.Loss):
#     def __init__(self):
#         super(selfLoss,self).__init__()
#
#     def __call__(self,y_true,y_pred):
#         loss=tf.reduce_mean(tf.math.pow(tf.reshape(y_true-y_pred,(-1,)),2.0))
#         return loss

def selfLoss(y_true,y_pred):
    loss=tf.reduce_mean(tf.math.pow(tf.reshape(y_true-y_pred,(-1,)),2.0))
    return loss

class function_o(tf.keras.Model):#计算V(S）
    def __init__(self,state_shape,action_shape,state_init,state_range,name="critic",training=True,**kwargs):
        super(function_o, self).__init__(name=name,**kwargs)
        self.state_shape=state_shape
        self.state_init=state_init
        self.state_range=state_range
        self.action_shape=action_shape
        self.training=training
        self.block_1 = Linear(name="critic_linear1",units=62,training=training)
        self.block_2 = Linear(name="critic_linear2",units=62,training=training)
        self.block_3 = Linear(name="critic_linear3",units=32,training=training)
        self.block_4 = Linear(name="critic_linear3",units=state_shape,training=training)

    @tf.function
    def call(self,input):#此处不能是__call__
        #组合为(state,action)
        # states=input[:,:-1]
        # action=tf.reshape(input[:,-1],(-1,1))
        # action=tf.abs(action)/(1.0+tf.abs(action))
        # #action是被隐射到0到1之间
        # x=tf.concat([states,action],axis=1)
        x = self.block_1(input)
        # x = tf.nn.relu(x)
        x = self.block_2(x)
        # x = tf.nn.relu(x)
        x = self.block_3(x)
        # x = tf.nn.relu(x)
        x = self.block_4(x)
        x=tf.multiply(tf.sigmoid(x),self.state_range)+self.state_init
        return x

    def train(self,x_train,y_ture,size):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1.5e-3)
        self.compile(optimizer, loss=selfLoss)
        self.load_weights('./ckpt/')
        self.fit(x_train, y_ture, epochs=30,batch_size=size,verbose=0)
        self.save_weights('./ckpt/')

    def get_config(self):
        config = super(function_o, self).get_config()
        return config

#成本函数
class reward_o():
      def __init__(self,fun_o):
          self.fun=fun_o #跳转方程
          self.env=env_world

      @tf.function
      def __call__(self,input):
          state_next=self.fun(input)
          x=state_next[:,0]
          x_dot=state_next[:,1]
          theta=state_next[:,2]
          theta_dot=state_next[:,3]
          r1 = (self.env.x_threshold - tf.abs(x))/self.env.x_threshold - 0.8
          r2 = (self.env.theta_threshold_radians - tf.abs(theta))/self.env.theta_threshold_radians - 0.5
          t=r1+r2
          return -1.0*t #由于是成本函数所以用负数

#求hessian矩阵
@tf.function
def compute_hessians(z,xy):
    y=z(xy)
    hes=tf.hessians(y,xy)
    return hes

class iLQR():
      def __init__(self,fun_t,T=5,N=10):
          self.env=env_world
          self.fun=fun_t #构造跳转函数
          self.reward_fun=reward_o(self.fun) #构造
          from collections import deque
          self.sample=deque([])#有效利用历史样本V（s）
          self.N_sample=2000
          self.N_sample_remove=500
          self.N_sample_batch=1000
          self.action=0
          self.T=T #向后进行plan的周期
          self.N=N #在执行N次后，重新拟合一次
          self.r=0.9
          self.beta=0.0001

      def sample_random(self):#由于critic计算V（s）可以利用历史采样样本
          d=self.sample.__len__()
          if  d>=self.N_sample:#buffer满腾出空间，补充新样本
              print("sample_critic已满，扩充样本")
              for e in range(self.N_sample_remove):
                  self.sample.popleft()
              d=self.sample.__len__()
          while d<self.N_sample:
              state_now = self.env.reset()
              sum_loss=0
              while True:
                  # self.env.render()
                  action_now=self.env.action_space.sample()
                  state_next,reward,done,info = self.env.step(action_now)
                  x, x_dot, theta, theta_dot = state_next
                  r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                  r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                  reward = r1 + r2
                  if  done:
                      d=d+1
                      self.sample.append((state_now,state_next,action_now,reward))
                      sum_loss+=reward
                      if d%5==0:
                          print("critic sample total reward:",sum_loss)
                      break
                  else:
                      d=d+1
                      self.sample.append((state_now,state_next,action_now,reward))
                      sum_loss+=reward
                      state_now=state_next

      #训练跳转方差st+1=f(st,at)
      def function_train(self):
          fun_sample=list(self.sample).copy()
          random.shuffle(fun_sample)
          fun_sample=fun_sample[:int(self.N_sample_batch)]
          state_next=[]
          state_now=[]
          action_now=[]
          for e in fun_sample:
              state_next.append(e[1])
              state_now.append(e[0])
              action_now.append(e[2])
          y=tf.constant(np.array(state_next),tf.float32)
          state_now=tf.constant(np.array(state_now),tf.float32)
          action_now=tf.reshape(tf.constant(np.array(action_now),tf.float32),(-1,1))
          x=tf.concat([state_now,action_now],axis=1)
          self.fun.train(x,y,self.N_sample_batch)

      #max trace
      def max_sample_trace(self,state=[],trace_num=5):
          trace_list=[]
          temp_list=[]
          i=0
          if state.__len__()>0:
             state_now=np.array(state)
             self.env.env.state=state_now
          else:
             state_now=self.env.env.reset()
          while i<trace_num:
              # self.env.env.render()
              action=self.env.env.action_space.sample()
              state_next,reward,done,info=self.env.env.step(action)
              x, x_dot, theta, theta_dot = state_next
              r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
              r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
              reward=r1+r2
              if done==True:
                 temp_list.append((state_now,state_next,action,reward))
                 if state.__len__()>0:
                    state_now=np.array(state)
                    self.env.env.state=state_now
                 else:
                    state_now=self.env.env.reset()
                 trace_list.append(temp_list)
                 temp_list=[]
                 i=i+1
              else:
                 temp_list.append((state_now,state_next,action,reward))
                 state_now=state_next
          max_value=-99999999
          max_list=[]
          max_len=-1
          for e in trace_list:
              temp_len=list(e).__len__()
              temp_list=list(e)
              sum_value=0
              for i in range(temp_len):
                  sum_value=sum_value+np.power(self.r,i)*temp_list[i][3]
              if sum_value>max_value:
                 max_len=temp_len
                 max_list=list(e)
                 max_value=sum_value
          return max_list,max_len,max_value

      #求hessian矩阵
      def hessian_matrix(self,f,list_sa=[],shape_num=5):
          result=compute_hessians(f,list_sa)
          return result

      #求雅可比矩阵
      def jacobian_matrix(self,f,list_sa=[],shape_num=5):
          with tf.GradientTape() as tape:
               tape.watch(list_sa)
               value=f(list_sa)
          result=tf.reduce_sum(tape.jacobian(value,list_sa),axis=0)
          return result

      #对当前state，plan出最大reward的动作
      def plan_action(self,state_now=[]):
          sdim=4
          adim=1
          max_list,max_len,max_value=self.max_sample_trace(state=state_now)
          if max_len>self.T:
             max_list=max_list[:self.T]
             max_len=self.T
          for _ in range(3):
              state_action=[]
              for e in max_list:
                  s=tf.constant(e[0],tf.float32)
                  a=tf.reshape(tf.constant(e[2],tf.float32),(-1,))
                  state_action.append(tf.concat([s,a],axis=0))
              Ct_list=self.hessian_matrix(self.reward_fun,state_action)
              ct_list=self.jacobian_matrix(self.reward_fun,state_action)
              Ft_list=self.jacobian_matrix(self.fun,state_action)
              Kt_list=[]
              kt_list=[]
              #计算LQR
              #计算T时刻
              Qt=tf.reshape(Ct_list[max_len-1],(sdim+adim,sdim+adim))
              qt=tf.reshape(ct_list[max_len-1],(-1,1))
              Qxx=Qt[:sdim,:sdim]
              Qux=Qt[sdim:,:sdim]
              Quu=Qt[sdim:,sdim:][0][0]
              Qxu=Qt[:sdim,sdim:]
              qu=qt[sdim:,][0][0]
              qx=qt[:sdim,]
              Kt=(-1.0)/Quu*Qux
              kt=(-1.0)/Quu*qu
              Vt=Qxx+tf.matmul(Qxu,Kt)+tf.matmul(tf.transpose(Kt),Qux)+tf.matmul(tf.transpose(Kt)*Quu,Kt)
              vt=qx+Qxu*kt+tf.transpose(Kt)*qu+tf.transpose(Kt)*Quu*kt
              Kt_list.append(Kt)
              kt_list.append(kt)
              for t in range(max_len-1):
                  Ft=tf.reshape(Ft_list[max_len-2-t],(sdim,sdim+adim))
                  Ct=tf.reshape(Ct_list[max_len-2-t],(sdim+adim,sdim+adim))
                  ct=tf.reshape(ct_list[max_len-2-t],(-1,1))
                  Qt=Ct+tf.matmul(tf.matmul(tf.transpose(Ft),Vt),Ft)
                  qt=ct+tf.matmul(tf.transpose(Ft),vt)
                  Qxx=Qt[:sdim,:sdim]
                  Qux=Qt[sdim:,:sdim]
                  Quu=Qt[sdim:,sdim:][0][0]
                  Qxu=Qt[:sdim,sdim:]
                  qu=qt[sdim:,][0][0]
                  qx=qt[:sdim,]
                  Kt=(-1.0)/Quu*Qux
                  kt=(-1.0)/Quu*qu
                  Vt=Qxx+tf.matmul(Qxu,Kt)+tf.matmul(tf.transpose(Kt),Qux)+tf.matmul(tf.transpose(Kt)*Quu,Kt)
                  vt=qx+Qxu*kt+tf.transpose(Kt)*qu+tf.transpose(Kt)*Quu*kt
                  Kt_list.append(Kt)
                  kt_list.append(kt)
              Kt_list.reverse() #1...T
              kt_list.reverse()

              #修正得到新的state和新的action
              #当t==0
              max_list_new=[]
              e=max_list[0]
              new_action=self.beta*kt_list[0]+tf.constant(e[2],tf.float32)
              if np.abs(new_action-1.0)<np.abs(new_action):
                 new_action=1
              else:
                 new_action=0
              s_a=tf.constant(list(e[0])+list([new_action]),tf.float32)
              max_list_new.append([e[0],self.fun([s_a])[0],new_action])#new_next_state
              #按新的动作，调整样本状态
              for t in range(max_len-1):
                  now_s=max_list[t+1]
                  last_s=max_list_new[t]
                  dx=tf.reshape(tf.constant(last_s[1]-now_s[0],tf.float32),(-1,1))
                  new_action=(tf.matmul(Kt_list[t+1],dx)+self.beta*kt_list[t+1]+now_s[2]).numpy()[0][0]
                  if  np.abs(new_action-1.0)<np.abs(new_action):
                      new_action=tf.constant(1.0,tf.float32)
                  else:
                      new_action=tf.constant(0.0,tf.float32)
                  s_a=tf.constant(list(now_s[0])+list([new_action.numpy()]),tf.float32)
                  max_list_new.append([now_s[0],self.fun([s_a])[0],new_action])#new_next_state
              max_list=max_list_new
          action_max=max_list[0][2]
          return action_max

      def train_iLQR(self):
          self.sample_random()
          self.function_train()#训练st+1=f(st,at)
          i=0
          state_now = self.env.env.reset()
          d=0
          while i<1000000:
                # self.env.render()
                if i!=0 and i%self.N==0:
                   self.function_train()#训练st+1=f(st,at)
                action=self.plan_action(state_now)
                state_next,reward,done,info=self.env.env.step(action)
                if done:
                   if self.sample.__len__()>=self.N_sample:#buffer满腾出空间，补充新样本
                       for e in range(self.N_sample_remove):
                           self.sample.popleft()
                   self.sample.append((state_now,state_next,action,reward))
                   if i%1==0:
                      print("sample total step:",d)
                   state_now=self.env.env.reset()
                   d=0
                else:
                   d=d+1
                   self.sample.append((state_now,state_next,action,reward))
                   state_now=state_next
                i=i+1

# print(env_world.observation_space.low)
fun=function_o(4,1,tf.constant([-2.0,-0.30943951023931953,-1.5,-1.4],tf.float32),tf.constant([4.0,0.30943951023931953*2,3.0,2.8],tf.float32))
iLQR_a=iLQR(fun)
iLQR_a.train_iLQR()



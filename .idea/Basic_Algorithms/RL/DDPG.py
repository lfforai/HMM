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
deta=0.02
from tensorflow.keras import layers

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

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

    def assign_pra(self,other,deta):#deta=0.001为更新，=1.0为完全替换
        self.w.assign(other.w*deta+(1.0-deta)*self.w)
        self.b.assign(other.b*deta+(1.0-deta)*self.b)

class critic(layers.Layer):
    def __init__(self, name="critic",training=False,**kwargs):
        super(critic, self).__init__(name=name,**kwargs)
        self.training=training
        self.block_1 = Linear(name="critic_linear1",units=256,training=training)
        self.block_2 = Linear(name="critic_linear2",units=256,training=training)
        self.block_3 = Linear(name="critic_linear3",units=1,training=training)

    def call(self, inputs_state,input_action):
        x=tf.concat([inputs_state,input_action],-1)
        x = self.block_1(x)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x = tf.nn.relu(x)
        x = self.block_3(x)
        return x

    def get_config(self):
        config = super(critic, self).get_config()
        return config

    def assign_pra(self,other,deta):
        self.block_1.assign_pra(other.block_1,deta)
        self.block_2.assign_pra(other.block_2,deta)
        self.block_3.assign_pra(other.block_3,deta)

class actor(layers.Layer):
    def __init__(self, name="actor",training=False,**kwargs):
        super(actor, self).__init__(name=name,**kwargs)
        self.training=training
        self.block_1 = Linear(name="actor_linear1",units=256,training=training)
        self.block_2 = Linear(name="actor_linear2",units=256,training=training)
        self.block_3 = Linear(name="actor_linear3",units=1,training=training)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x = tf.nn.relu(x)
        x = self.block_3(x)
        x=tf.nn.tanh(x)*2.0
        return x

    def get_config(self):
        config = super(actor, self).get_config()
        return config

    def assign_pra(self,other,deta):
        self.block_1.assign_pra(other.block_1,deta)
        self.block_2.assign_pra(other.block_2,deta)
        self.block_3.assign_pra(other.block_3,deta)

class critic_eval_actor_eval_Net(tf.keras.Model):#critic网络,确定动作值函数
    def __init__(self,**kwargs):
        super(critic_eval_actor_eval_Net, self).__init__(**kwargs)
        self.actor_eval=actor(name="actor_eval",training=False)
        # self.actor_training=actor(name="actor_training",training=True)
        self.critic_eval=critic(name="critic_eval",training=False)
        # self.critic_training=critic(name="critic_training",training=True)
        # self.if_add_normal=False

    def call(self,input):
        action=self.actor_eval(input)
        x=self.critic_eval(input,action)
        return x

    def param_copy(self,critic_o,actor_o,deta):#参数的复制
        if self.built==True:
            self.critic_eval.assign_pra(critic_o,deta)
            self.actor_eval.assign_pra(actor_o,deta)
        else:
            print("没有进行build(input_shape=(None,inits)),参数没有实列化无法进行copy操作")

class critic_train_Net(tf.keras.Model):#critic网络,确定状态动作值函数
    def __init__(self,**kwargs):
        super(critic_train_Net, self).__init__(**kwargs)
        self.critic_train=critic(name="critic_train",training=True)

    def call(self,inputs):
        x=self.critic_train(inputs[0],inputs[1])
        return x

    def train(self,x_train,action_train,y_ture,size):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.fit(x=[x_train,action_train], y=y_ture, epochs=1,batch_size=size,verbose=0)

    def param_copy(self,critic,deta):#参数的复制
        if self.built==True:
            self.critic_train.assign_pra(critic,deta)
        else:
            print("没有进行build(input_shape=(None,inits)),参数没有实列化无法进行copy操作")

from tensorflow.keras import backend as K
def actor_loss(y_true, y_pred):
    return tf.constant(-1.0)*K.mean(y_pred)
class actor_train_Net(tf.keras.Model):#actor网络,确定动作函数
    def __init__(self,**kwargs):
        super(actor_train_Net, self).__init__(**kwargs)
        self.actor_train=actor(name="actor_train",training=True)
        self.critic_eval=critic(name="critic_eval",training=False)

    def call(self,input):
        action=self.actor_train(input)
        x=self.critic_eval(input,action)
        return x

    def train(self,x_train,y_ture,size):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.compile(optimizer, loss=actor_loss)
        self.fit(x_train, y_ture, epochs=1,batch_size=size,verbose=0)

    def param_copy(self,critic_o,actor_o,deta):#参数的复制
        if self.built==True:
            self.critic_eval.assign_pra(critic_o,deta)
            self.actor_train.assign_pra(actor_o,deta)
        else:
            print("没有进行build(input_shape=(None,inits)),参数没有实列化无法进行copy操作")

class DDPG:
    def __init__(self,filepath="/root/model2/",**kwargs):
        self.filepath=filepath
        # self.env=gym.make("MountainCarContinuous-v0")
        from collections import deque
        self.memory=deque([])
        self.r=0.9
        self.var=3.0
        self.var_increase=0.95
        self.var_mini=0.001
        self.mini_memory_len=1000
        self.size=100
        self.N=100
        self.env=gym.make("Pendulum-v0") #
        # self.env=gym.make("MountainCarContinuous-v0")
        self.Qvalue_eval=critic_eval_actor_eval_Net()
        self.Qvalue_train=critic_train_Net()
        self.actor_train=actor_train_Net()
        if not os.path.exists(self.filepath):
            print("存储参数")
            os.makedirs(self.filepath)
            self.Qvalue_eval.build(input_shape=(None,self.env.observation_space.shape[0]))
            # print(self.Qvalue_eval.weights)
            # exit()
            self.Qvalue_train.build(input_shape=[(None,self.env.observation_space.shape[0]),(None,1)])
            self.actor_train.build(input_shape=(None,self.env.observation_space.shape[0]))
            #初始值参数统一
            self.Qvalue_eval.param_copy(self.Qvalue_train.critic_train,self.actor_train.actor_train,deta=1.0)#替换
            self.actor_train.param_copy(self.Qvalue_train.critic_train,self.Qvalue_eval.actor_eval,deta=1.0)#替换
        else:
            print("加载参数")
            self.Qvalue_eval.build(input_shape=(None,self.env.observation_space.shape[0]))#必须先构造出来才能
            self.Qvalue_eval.load_weights(self.filepath+"Qvalue_eval_1.0")
            self.Qvalue_train.build(input_shape=[(None,self.env.observation_space.shape[0]),(None,1)])
            self.Qvalue_train.load_weights(self.filepath+"Qvalue_train_1.0")
            self.actor_train.build(input_shape=(None,self.env.observation_space.shape[0]))
            self.actor_train.load_weights(self.filepath+"actor_train_1.0")


    def action(self,state):
        # print(self.actor_train.actor_train(np.array([state]))[0].numpy())
        result= np.clip(np.random.normal(loc=self.actor_train.actor_train(np.array([state]))[0].numpy(),scale=self.var), -2.0, 2.0)#返回当前状态所确定的动作，使用可训练参数
        # print("result",result)
        # print("--------------")
        return result

    def sample(self):
        for c in range(10):
            sum_reward=0
            d=0
            state_now = self.env.reset()
            while True:
                self.env.render()
                action_now=self.action(state_now)
                state_next,reward,done,info = self.env.step(action_now)
                sum_reward+=reward
                if done:
                    if reward>0:
                        print(state_next,reward)
                    d=d+1
                    if c%1==0:
                        print("执行步数:%d"%(d))
                        print("累计回报：%f"%(sum_reward))
                        print("方差：%f"%(self.var))
                    break
                else:
                    if reward>0:
                        print(state_next,reward)
                    d=d+1
                    state_now=state_next
        print("采样完毕，开始迭代")

    def DDPG_train(self):
        #更新样板：样本集合大小始终固定不变,样本数超过最大容量，替换掉第一个元素
        for e  in range(self.N):#进行迭代的次数，每次迭代进行
            # if self.var*self.var_increase>self.var_mini:
            #     self.var=self.var_increase*self.var
            # else:
            #     self.var=3.0
            self.var=0.01+np.random.rand()*2.9
            # self.var=0.01
            # self.sample()
            sum_reward=0
            state_now = self.env.reset()
            for T in range(500):
                if self.memory.__len__()<self.size:#如果样本量不足，循环采集
                    self.env.render()
                    if e==0:
                        action_now=self.env.action_space.sample()
                    else:
                        action_now=self.action(state_now)
                    state_next,reward,done,info = self.env.step(action_now)
                    sum_reward+=reward
                    # position,_=state_next
                    # reward=reward+(abs(position+0.5)/0.95-0.5)*0.01
                    if done:
                        self.memory.append([state_now,state_next,reward,action_now,done])
                        state_now = self.env.reset()
                    else:
                        self.memory.append([state_now,state_next,reward,action_now,done])
                        state_now=state_next
                elif self.memory.__len__()>=self.size:#当新样本满的时候替换掉队列中的老样本
                    if self.memory.__len__()>10000:
                        self.memory.popleft()#当新样本满的时候替换掉队列中的老样本
                    self.env.render()
                    action_now=self.action(state_now)
                    state_next,reward,done,info = self.env.step(action_now)
                    sum_reward+=reward
                    # position,_=state_next
                    # reward=reward+(abs(position+0.5)/0.95-0.5)*0.01
                    # print(action_now,reward,self.var)
                    if done:
                        self.memory.append([state_now,state_next,reward,action_now,done])
                        state_now = self.env.reset()#只有当一轮状态达到终止时候才进行样本环境重置
                    else:
                        self.memory.append([state_now,state_next,reward,action_now,done])
                        state_now=state_next
                    # print("开始训练")
                    temp_memory=list(self.memory).copy()#进行采样
                    random.shuffle(temp_memory)
                    temp_memory=temp_memory[:self.size]
                    x_train=[]
                    y_critic_train=[]
                    action_vector=[]
                    for v in temp_memory:
                        action_vector.append(v[3])
                        if v[4]:
                            y_critic_train.append(v[2]+self.r*self.Qvalue_eval([list(v[1])])[0][0].numpy())
                            x_train.append(v[0])
                        else:
                            y_critic_train.append(v[2]+self.r*self.Qvalue_eval([list(v[1])])[0][0].numpy())
                            x_train.append(v[0])
                    x_train=tf.constant(x_train,dtype=float)
                    y_critic_train=tf.reshape(tf.constant(y_critic_train,dtype=tf.float32),(-1,1))
                    action_vector=tf.reshape(tf.constant(action_vector,dtype=tf.float32),(-1,1))
                    self.Qvalue_train.train(x_train,action_vector,y_critic_train,self.size)
                    # print("Qvalue_train over!")
                    self.actor_train.param_copy(self.Qvalue_train.critic_train,self.actor_train.actor_train,deta=1.0)#替换
                    self.actor_train.train(x_train,y_critic_train,self.size)
                    # print("actor_train over!")
                    #初始值参数统一，柔性更新参数 A=(1-deta)*A+deta*B
                    self.Qvalue_eval.param_copy(self.Qvalue_train.critic_train,self.actor_train.actor_train,deta=deta)#更新参数
                    if T%20==0:
                        self.Qvalue_eval.save_weights(os.path.join(self.filepath,'Qvalue_eval_{}'.format(1.0)))
                        self.Qvalue_train.save_weights(os.path.join(self.filepath,'Qvalue_train_{}'.format(1.0)))
                        self.actor_train.save_weights(os.path.join(self.filepath,'actor_train_{}'.format(1.0)))
            print("累计回报：=",sum_reward)
ddpg=DDPG()
ddpg.DDPG_train()
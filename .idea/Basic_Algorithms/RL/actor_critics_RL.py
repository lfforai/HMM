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

#基于actor_critics和 actor_critics_GAE的算法

#构造神经网
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

class critic(tf.keras.Model):#计算V(S）
    def __init__(self, name="critic",training=True,**kwargs):
        super(critic, self).__init__(name=name,**kwargs)
        self.training=training
        self.block_1 = Linear(name="critic_linear1",units=62,training=training)
        self.block_2 = Linear(name="critic_linear2",units=62,training=training)
        self.block_3 = Linear(name="critic_linear3",units=1,training=training)

    @tf.function
    def call(self, inputs_state):
        x = self.block_1(inputs_state)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x = tf.nn.relu(x)
        x = self.block_3(x)
        return x

    def train(self,x_train,y_ture,size):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1.5e-3)
        self.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
        self.fit(x_train, y_ture, epochs=5,batch_size=size,verbose=0)

    def get_config(self):
        config = super(critic, self).get_config()
        return config

#loss:sum(A(at,st)*log(Pi(at|st)))
class selfLoss(tf.keras.losses.Loss):
    def __init__(self,len):
        super(selfLoss,self).__init__()
        self.len=len

                   #A(at,st) #log(Pi(at|st))
    def __call__(self,y_true,y_pred):
        y_pred=tf.math.log(tf.cast(y_pred,tf.float32))#Pi(at|st)
        y_true=tf.cast(y_true,tf.float32)#A(at,st)
        y_true=tf.reshape(y_true,(self.len,-1))#列
        y_pred=tf.reshape(y_pred,(1,-1))#行
        result=tf.matmul(y_pred,y_true)
        return tf.reshape(result,(-1,))

class actor(tf.keras.Model):#计算Pi(at|st)
    def __init__(self, name="actor",training=True,**kwargs):
        super(actor, self).__init__(name=name,**kwargs)
        self.training=training
        self.block_1 = Linear(name="actor_linear1",units=62,training=training)
        self.block_2 = Linear(name="actor_linear2",units=62,training=training)
        self.block_3 = Linear(name="actor_linear1",units=2,training=training)

    @tf.function
    def call(self, inputs_state):
        x = self.block_1(inputs_state)
        x = tf.nn.relu(x)
        x = self.block_2(x)
        x=tf.nn.relu(x)
        x=self.block_3(x)
        x = tf.nn.softmax(x)
        return x

    def get_config(self):
        config = super(actor, self).get_config()
        return config

#使用向量作为,用线性模型来拟合
class actor_done():
    def __init__(self,pra_num):
        self.pra_num=pra_num
        self.w1_action_lr=tf.cast(tf.Variable(np.random.normal(size=(self.pra_num+1)*2),dtype=tf.float32),tf.float32)
        temp_list=[]
        for g in range(int(self.pra_num)):
            a2=np.random.random()*4.0-2.0 #小车速度
            # a1=-0.25943951023931953+0.25943951023931953*2/(self.pra_num/2)*g
            a1=-0.30943951023931953+np.random.random()*0.30943951023931953*2 #小车位置
            b2=np.random.random()*3.0-1.5 #平衡杆角速度
            b1=-1.4+1.4/(self.pra_num/2)*g
            # b1=-1.4+np.random.random()*1.4*2 #平衡杆偏离垂直的角度
            temp_list.append([b1,b2,a1,a2])
        self.bf_function=np.array(temp_list,dtype=float)

    def chage_state(self,state,action):
            BFlist=[1.0]
            value=np.array(state,dtype=float)
            for e in list(self.bf_function):
                BFlist.append(np.exp(-np.dot((value-np.array(e,dtype=float)),(value-np.array(e,dtype=float)))/(2.0*0.25)))
            if action==0:
                result=np.array(list(BFlist)+[1.0]+list(np.zeros(self.pra_num,dtype=float)),dtype=float)
                return result
            else:
                result=np.array([1.0]+list(np.zeros(self.pra_num,dtype=float))+list(BFlist),dtype=float)
                return result

    def one_action(self,state):
        state_left=self.chage_state(state,0)
        state_right=self.chage_state(state,1)
        state_left =tf.cast(np.array(state_left),dtype=tf.float32)
        state_right=tf.cast(np.array(state_right),dtype=tf.float32)
        state_left=tf.reshape(state_left,(1,-1))
        state_right=tf.reshape(state_right,(1,-1))
        right_pro=tf.exp(tf.matmul(state_right,tf.reshape(self.w1_action_lr,(-1,1))))/ \
                  (tf.exp(tf.matmul(state_right,tf.reshape(self.w1_action_lr,(-1,1))))+tf.exp(tf.matmul(state_left,tf.reshape(self.w1_action_lr,(-1,1)))))
        return np.random.choice(a=[0,1],p=[1.0-right_pro.numpy()[0][0],right_pro.numpy()[0][0]])

    def vector_action(self,states):
        states_list=list(states)
        state_left=[]
        state_right=[]
        for e in states_list:
            state_left.append(self.chage_state(e,0))
            state_right.append(self.chage_state(e,1))
        state_left =tf.cast(np.array(state_left),dtype=tf.float32)
        state_right=tf.cast(np.array(state_right),dtype=tf.float32)
        right_pro=tf.exp(tf.matmul(state_right,tf.reshape(self.w1_action_lr,(-1,1))))/ \
                    (tf.exp(tf.matmul(state_right,tf.reshape(self.w1_action_lr,(-1,1))))+tf.exp(tf.matmul(state_left,tf.reshape(self.w1_action_lr,(-1,1)))))
        left_pro=tf.constant(1.0,tf.float32)-right_pro
        return tf.concat([left_pro,right_pro],axis=-1)

class actor_critics():
    def __init__(self,filepath="/root/natural/"):
        self.actor_net_vector=actor_done(400)
        self.filepath=filepath
        self.actor_net=actor()#计算actor=PI（at|st）
        self.critic_net=critic()#计算V（s）
        self.env=gym.make("CartPole-v1") #
        if not os.path.exists(self.filepath):
            print("存储参数")
            # os.makedirs(self.filepath)
            self.actor_net.build(input_shape=(None,self.env.observation_space.shape[0]))
            self.critic_net.build(input_shape=(None,self.env.observation_space.shape[0]))
        else:
            print("加载参数")
            self.actor_net.load_weights(self.filepath+"actor_1.0")
            self.critic_net.load_weights(self.filepath+"critic_1.0")

        from collections import deque
        self.sample_critic=deque([])#有效利用历史样本V（s）
        self.sample_actor=[]#无法利用历史样本actor=PI（at|st）
        self.N_critic=500#buffer的样本数
        self.N_actor=500#buffer的样本数
        self.N_critic_remove=200#V（s）buffer每次更新的样本数量
        self.N_critic_batch=200
        self.N_actor_batch=200#梯度下降的batchsize
        self.r=0.90#折扣系数
        self.bate=0.85#计算GEA的weight
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-3)

    def action(self,state):#[state],以Pi（at|st）选择
        state=np.array(state,dtype=float)
        left_pro,right_pro=self.actor_net(state)[0]
        left_pro=left_pro.numpy()
        right_pro=1.0-left_pro
        return np.random.choice(a=[0,1],p=[left_pro,right_pro])

    def coll_sample_critic(self):#由于critic计算V（s）可以利用历史采样样本
        d=self.sample_critic.__len__()
        if  d>=self.N_critic:#buffer满腾出空间，补充新样本
            print("sample_critic已满，扩充样本")
            for e in range(self.N_critic_remove):
                self.sample_critic.popleft()
            d=self.sample_critic.__len__()
        while d<self.N_critic:
            state_now = self.env.reset()
            sum_loss=0
            while True:
                self.env.render()
                action_now=self.action([state_now])
                state_next,reward,done,info = self.env.step(action_now)
                x, x_dot, theta, theta_dot = state_next
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                if  done:
                    d=d+1
                    self.sample_critic.append((state_now,state_next,reward,action_now,-1))
                    sum_loss+=reward
                    if d%5==0:
                        print("critic sample total reward:",sum_loss)
                    break
                else:
                    d=d+1
                    self.sample_critic.append((state_now,state_next,reward,action_now,1))
                    sum_loss+=reward
                    state_now=state_next

    def coll_sample_critic_vector(self):#由于critic计算V（s）可以利用历史采样样本
        d=self.sample_critic.__len__()
        if  d>=self.N_critic:#buffer满腾出空间，补充新样本
            print("sample_critic已满，扩充样本")
            for e in range(self.N_critic_remove):
                self.sample_critic.popleft()
            d=self.sample_critic.__len__()
        while d<self.N_critic:
            state_now = self.env.reset()
            sum_loss=0
            while True:
                self.env.render()
                action_now=self.actor_net_vector.one_action(state_now)
                state_next,reward,done,info = self.env.step(action_now)
                x, x_dot, theta, theta_dot = state_next
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                if  done:
                    d=d+1
                    self.sample_critic.append((state_now,state_next,reward,action_now,-1))
                    sum_loss+=reward
                    if d%5==0:
                        print("critic sample total reward:",sum_loss)
                    break
                else:
                    d=d+1
                    self.sample_critic.append((state_now,state_next,reward,action_now,1))
                    sum_loss+=reward
                    state_now=state_next

    def coll_sample_actor(self):#由于actor计算Pi（at|st）
        self.sample_actor=[]#由于不可使用历史样本，每次必须清空
        d=0
        list_num=0
        while d<self.N_actor:
            state_now = self.env.reset()
            sum_loss=0
            index=0#用于记录当前reward需要折扣回的时间t
            while True:
                self.env.render()
                action_now=self.action([state_now])
                state_next,reward,done,info = self.env.step(action_now)
                x, x_dot, theta, theta_dot = state_next
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                if  done:
                    index+=1
                    d=d+1
                    self.sample_actor.append((state_now,state_next,reward,action_now,index,-1,list_num))
                    sum_loss+=reward
                    if d%5==0:
                        print("actor sample total reward:",sum_loss)
                    list_num+=1
                    break
                else:
                    index+=1
                    d=d+1
                    #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]t折扣期,e[5]done
                    self.sample_actor.append((state_now,state_next,reward,action_now,index,1,list_num))
                    sum_loss+=reward
                    state_now=state_next

    def coll_sample_actor_vector(self):#由于actor计算Pi（at|st）
        self.sample_actor=[]#由于不可使用历史样本，每次必须清空
        d=0
        list_num=0
        while d<self.N_actor:
            state_now = self.env.reset()
            sum_loss=0
            index=0#用于记录当前reward需要折扣回的时间t
            while True:
                self.env.render()
                action_now=self.actor_net_vector.one_action(state_now)
                state_next,reward,done,info = self.env.step(action_now)
                x, x_dot, theta, theta_dot = state_next
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                if  done:
                    index+=1
                    d=d+1
                    self.sample_actor.append((state_now,state_next,reward,action_now,index,-1,list_num))
                    sum_loss+=reward
                    if d%5==0:
                        print("actor sample total reward:",sum_loss)
                    list_num+=1
                    break
                else:
                    index+=1
                    d=d+1
                    #e[0]当前s，e[1]跳转s,e[2]reward，e[3]当前action,e[4]t折扣期,e[5]done
                    self.sample_actor.append((state_now,state_next,reward,action_now,index,1,list_num))
                    sum_loss+=reward
                    state_now=state_next

    #训练模型
    def critic_train(self):
        s_critic=list(self.sample_critic).copy()
        random.shuffle(s_critic)
        s_critic=s_critic[:int(self.N_critic_batch)]
        y=[]
        x=[]
        for e in s_critic:
            if e[4]>0:#非终止状态
                x.append(e[0])
                y.append(e[2]+self.r*self.critic_net(np.array([e[1]]))[0])
            else:#终止状态
                x.append(e[0])
                y.append(e[2])
        x=np.array(x,dtype=float)
        y=np.array(y,dtype=float)
        self.critic_net.train(x,y,x.__len__())

    #普通actor算法
    def actor_train(self):
        s_actor=list(self.sample_actor).copy()
        random.shuffle(s_actor)
        s_actor=s_actor[:int(self.N_actor_batch)]
        actor_list=[]
        state=[]
        A=[]
        for e in s_actor:
            #计算当前action
            if e[3]==0:#left
                action_v=tf.constant([1.0,0.0],dtype=tf.float32)
            else:#right
                action_v=tf.constant([0.0,1.0],dtype=tf.float32)
            actor_list.append(action_v)

            #当前状态
            state.append(e[0])
            if e[5]>0:#非终止状态
                A.append(e[2]+self.r*self.critic_net(np.array([e[1]]))[0]-self.critic_net(np.array([e[0]]))[0])
            else:#终止状态
                A.append(e[2]-self.critic_net(np.array([e[0]]))[0])

        loss=selfLoss(A.__len__())
        actor_list=tf.cast(np.array(actor_list),tf.float32)
        mark=0
        while mark<10:
            with tf.GradientTape() as tape:
                action_pro=self.actor_net(np.array(state)) #由当前状态计算向左和向右的概率 pi(at|st)
                y_pre=tf.reduce_sum(tf.multiply(actor_list,action_pro),axis=1)
                z=-1.0*loss(np.array(A),y_pre)/tf.cast(A.__len__(),tf.float32)
            grads = tape.gradient(z,self.actor_net.trainable_weights)
            #利用偏导进行梯度下降调整
            self.optimizer.apply_gradients(zip(grads,self.actor_net.trainable_weights))
            mark=mark+1

    def actor_vector_train(self):
        s_actor=list(self.sample_actor).copy()
        random.shuffle(s_actor)
        s_actor=s_actor[:int(self.N_actor_batch)]
        actor_list=[]
        state=[]
        A=[]
        for e in s_actor:
            #计算当前action
            if e[3]==0:#left
                action_v=tf.constant([1.0,0.0],dtype=tf.float32)
            else:#right
                action_v=tf.constant([0.0,1.0],dtype=tf.float32)
            actor_list.append(action_v)

            #当前状态
            state.append(e[0])
            if e[5]>0:#非终止状态
                A.append(e[2]+self.r*self.critic_net(np.array([e[1]]))[0]-self.critic_net(np.array([e[0]]))[0])
            else:#终止状态
                A.append(e[2]-self.critic_net(np.array([e[0]]))[0])

        loss=selfLoss(A.__len__())
        actor_list=tf.cast(np.array(actor_list),tf.float32)
        mark=0
        while mark<5:
            with tf.GradientTape() as tape:
                action_pro=self.actor_net_vector.vector_action(state)#由当前状态计算向左和向右的概率 pi(at|st)
                y_pre=tf.reduce_sum(tf.multiply(actor_list,action_pro),axis=1)
                z=-1.0*loss(np.array(A),y_pre)/tf.cast(A.__len__(),tf.float32)
            grads = tape.gradient(z,[self.actor_net_vector.w1_action_lr])
            #利用偏导进行梯度下降调整
            self.optimizer.apply_gradients(zip(grads,[self.actor_net_vector.w1_action_lr]))
            mark=mark+1

    def actor_train_GAE_vector(self):
        s_actor=list(self.sample_actor).copy()
        s_actor=s_actor[:int(self.N_actor_batch)]

        actor_list=[]
        state=[]
        A=[]
        for e in s_actor:
            #计算当前action
            if e[3]==0:#left
                action_v=tf.constant([1.0,0.0],dtype=tf.float32)
            else:#right
                action_v=tf.constant([0.0,1.0],dtype=tf.float32)
            actor_list.append(action_v)
            #当前状态
            state.append(e[0])

        #把每个子过程分离开
        list_A=[]
        sub_list_index=0 #每个子过程的序列号1...T分离为一组
        temp_list=[]
        j=0
        for e in s_actor:
            if e[6]==sub_list_index:
                temp_list.append(e)
            else:
                list_A.append(temp_list)
                sub_list_index=e[6]
                temp_list=[]
                temp_list.append(e)
        list_A.append(temp_list)

        A=[]#计算A（at，st）
        for sub_list in list_A:
            sub_list=list(sub_list)
            len_o=sub_list.__len__()
            temp_list=[]
            for e in sub_list[:-1]:
                if e[5]>0:#非终止状态
                    temp_list.append((e[2]+self.r*self.critic_net(np.array([e[1]]))[0]-self.critic_net(np.array([e[0]]))[0]).numpy()[0])
                else:#终止状态
                    temp_list.append((e[2]-self.critic_net(np.array([e[0]]))[0]).numpy()[0])

            for i in range(len_o):
                sum1=0
                list_sub=temp_list[i:-1]
                for e in list_sub:
                    sum1=sum1+np.power(self.r*self.bate,i)*e
                A.append(sum1)

        loss=selfLoss(A.__len__())
        actor_list=tf.cast(np.array(actor_list),tf.float32)
        mark=0
        while mark<5:
            with tf.GradientTape() as tape:
                action_pro=self.actor_net_vector.vector_action(state)#由当前状态计算向左和向右的概率 pi(at|st)
                y_pre=tf.reduce_sum(tf.multiply(actor_list,action_pro),axis=1)
                z=-1.0*loss(np.array(A),y_pre)/tf.cast(A.__len__(),tf.float32)
            grads = tape.gradient(z,[self.actor_net_vector.w1_action_lr])
            #利用偏导进行梯度下降调整
            self.optimizer.apply_gradients(zip(grads,[self.actor_net_vector.w1_action_lr]))
            mark=mark+1

    def train(self,type_method="GAE"):
        for e  in range(100000):
            self.coll_sample_critic_vector()
            self.critic_train()
            self.coll_sample_actor_vector()
            # self.actor_train()
            #self.actor_train_GAE()
            self.actor_train_GAE_vector()
            if e%10==0:
               self.actor_net.save_weights(os.path.join(self.filepath,'actor_{}'.format(1.0)))
               self.critic_net.save_weights(os.path.join(self.filepath,'critic_{}'.format(1.0)))

# a=actor_critics()
# a.train()
print(np.argmin(np.array([5.0,2.0,3.0,4.0]),axis=0))
from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import gym
import numpy as np
import random
import os

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

#四、REINFORCE方法
one=True
if (one==True):
    class net:
        def __init__(self,filepath):
            self.filepath=filepath
            self.learning_rate=0.001
            self.inputs = keras.Input(shape=(4,), name='img')
            self.x = layers.Dense(64,activation='relu')(self.inputs)
            # self.x = layers.Dense(128, activation='sigmoid')(self.x)
            self.x = layers.Dense(64, activation='sigmoid')(self.x)
            self.outputs = layers.Dense(1,activation='linear')(self.x) #Qvalue
            self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name='DQL_model')
            self.model.compile(loss='mse',
                               optimizer='adam',
                               metrics=['accuracy'])
            if os.path.exists(filepath):
                self.model=keras.models.load_model(filepath)
            else:
                self.model.save(filepath)
                self.model.summary()

        def __call__(self, x_train, y_train):
            x=tf.cast(x_train,dtype=tf.float32)
            y=tf.cast(y_train,dtype=tf.float32)
            del self.model
            self.model=keras.models.load_model(self.filepath)
            history = self.model.fit(x, y,
                                     epochs=1,verbose=0)
            self.model.save(self.filepath)

        def forecast(self,state):
            return self.model.predict(state)

    class REINFORCE:
        def __init__(self):
            self.pra_num=10
            self.w1_action_lr=np.random.normal(size=(self.pra_num+1)*2)
            self.w1_action_left=np.random.normal(size=(self.pra_num+1))
            self.w1_action_right=np.random.normal(size=(self.pra_num+1))
            if os.path.exists("./w1_action_lr.txt"):
                self.w1_action_lr=np.array(np.loadtxt("./w1_action_lr.txt"),dtype=float)
            if os.path.exists("./w1_action_left.txt"):
                self.w1_action_left=np.array(np.loadtxt("./w1_action_left.txt"),dtype=float)
            if os.path.exists("./w1_action_right.txt"):
                self.w1_action_right=np.array(np.loadtxt("./w1_action_right.txt"),dtype=float)
            self.state_s=np.random.normal(size=(16))
            self.env = gym.make('CartPole-v0')
            self.deta=0.01
            self.deta_min=0.0001
            self.deta_decay = 0.955
            self.r=0.955

            self.greedy=0.05
            self.greedy_min=0.05
            self.greedy_decay = 0.975
            self.memory=[]

            if os.path.exists("./bf_function.txt"):
                import csv
                with open("./bf_function.txt",newline='',encoding='UTF-8') as csvfile:
                    rows=csv.reader(csvfile)
                    self.bf_function=np.array(list(rows),dtype=float)
                csvfile.close()
            else:
                temp_list=[]
                for g in range(int(self.pra_num)):
                    a2=np.random.random()*4.0-2.0 #小车速度
                    # a1=-0.25943951023931953+0.25943951023931953*2/(self.pra_num/2)*g
                    a1=-0.30943951023931953+np.random.random()*0.30943951023931953*2 #小车位置
                    b2=np.random.random()*3.0-1.5 #平衡杆角速度
                    # b1=-1.0+2.0/(self.pra_num/2)*g
                    b1=-2.4+np.random.random()*2.4*2 #平衡杆偏离垂直的角度
                    temp_list.append([b1,b2,a1,a2])
                self.bf_function=np.array(temp_list,dtype=float)
                np.savetxt('./bf_function.txt',self.bf_function, delimiter=',')  # 数组x
            self.net1=net("Acto-critic.h5")

        def chage_state(self,state,action):
            # result=[]
            # result.append([1.0,state[0],state[1],state[2],state[3],state[0]+state[1],state[0]+state[2],state[0]+state[3],state[1]+state[2], \
            #            state[1]+state[3],state[2]+state[3],state[0]*state[1],state[0]*state[2],state[0]*state[3],state[1]*state[2],state[1]*state[3], \
            #            state[2]*state[3],np.exp(state[0]),np.exp(state[1]),np.exp(state[2]),np.exp(state[3])])
            # return np.array(result)
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
            # result1=[1.0,state[0],state[1],state[2],state[3],state[0]+state[1],state[0]+state[2],state[0]+state[3],state[1]+state[2],\
            #                state[1]+state[3],state[2]+state[3],state[0]*state[1],state[0]*state[2],state[0]*state[3],state[1]*state[2],state[1]*state[3],\
            #                state[2]*state[3]]
            # result2=[-10.0,state[0],state[1],state[2],state[3],state[0]+state[1],state[0]+state[2],state[0]+state[3],state[1]+state[2], \
            #         state[1]+state[3],state[2]+state[3],state[0]*state[1],state[0]*state[2],state[0]*state[3],state[1]*state[2],state[1]*state[3], \
            #         state[2]*state[3]]
            # if action==0:
            #    return np.array(result1+list(np.zeros(self.pra_num+1)))
            # else:
            #    return np.array(list(np.zeros(self.pra_num+1))+result2)

        def partial_derivative_lr(self,state,action):
            state_left=self.chage_state(state,0).reshape(-1)
            state_right=self.chage_state(state,1).reshape(-1)
            lr_value=tf.constant(self.w1_action_lr)
            if action==0:#left
                with tf.GradientTape() as t:
                    t.watch(lr_value)
                    z = tf.math.log(tf.exp(tf.reduce_sum(state_left*lr_value))/ \
                                    (tf.exp(tf.reduce_sum(state_left*lr_value))+tf.exp(tf.reduce_sum(state_right*lr_value))))
                dz_left = t.gradient(z,lr_value)
                del t
                return  dz_left.numpy()
            else:
                with tf.GradientTape() as t:
                    t.watch(lr_value)
                    z = tf.math.log(tf.exp(tf.reduce_sum(state_right*lr_value))/ \
                                    (tf.exp(tf.reduce_sum(state_left*lr_value))+tf.exp(tf.reduce_sum(state_right*lr_value))))
                dz_right = t.gradient(z,lr_value)
                del t
                return  dz_right.numpy()

        def action_lr(self,state):
            state_left=self.chage_state(state,0)
            state_right=self.chage_state(state,1)
            left_pro  = tf.exp(tf.reduce_sum(state_left*self.w1_action_lr))/ \
                        (tf.exp(tf.reduce_sum(state_left*self.w1_action_lr))+tf.exp(tf.reduce_sum(state_right*self.w1_action_lr)))
            right_pro = tf.exp(tf.reduce_sum(state_right*self.w1_action_lr))/ \
                        (tf.exp(tf.reduce_sum(state_left*self.w1_action_lr))+tf.exp(tf.reduce_sum(state_right*self.w1_action_lr)))

            if np.random.rand()<self.greedy:
                action =  self.env.action_space.sample()
                return action

            # if np.random.rand()<self.greedy:#贪婪算法
            if np.random.rand()<right_pro:
                return 1
            else:
                return 0

        def sample_lr(self):
            self.memory=[]
            for c in range(5):
                d=0
                state_now = self.env.reset()
                while True:
                    self.env.render()
                    action_now=self.action_lr(state_now)
                    state_next,reward,done,info = self.env.step(action_now)
                    x, x_dot, theta, theta_dot = state_next
                    r1 = (0.5 - abs(x))/self.env.x_threshold - 0.8
                    r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                    reward = r1 + r2
                    if done:
                        d=d+1
                        self.memory.append((state_now,state_next,reward,action_now,-1))
                        if c%5==0:
                            print("执行步数:%d"%(d))
                            print("贪婪度：%f"%(self.greedy))
                            print("deta:%f"%(self.deta))
                        # if d>155 and self.mark==0:
                        #    self.deta= 0.001
                        #    self.mark=1
                        break
                    else:
                        d=d+1
                        self.memory.append((state_now,state_next,reward,action_now,1))
                        state_now=state_next

        def a3c_syschronous(self):
            for c in range(5000):
                if c%5==0:
                    np.savetxt('./w1_action_lr.txt',self.w1_action_lr, delimiter=',')  # 数组x

                if c%3==0:
                    self.sample_lr()

                if self.deta*self.deta_decay<self.deta_min:
                    self.deta=self.deta_min
                else:
                    if c%20==0:
                        self.deta=self.deta*self.deta_decay
                # if self.greedy*self.greedy_decay<self.greedy_min:
                #    self.greedy=self.greedy_min
                # else:
                #    if c%3==0:
                #       self.greedy=self.greedy*self.greedy_decay
                d=0
                state_now = self.env.reset()
                self.memory=[]
                while True:
                    self.env.render()
                    action_now=self.action_lr(state_now)
                    state_next,reward,done,info = self.env.step(action_now)
                    x, x_dot, theta, theta_dot = state_next
                    r1 = (0.5 - abs(x))/self.env.x_threshold - 0.8
                    r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                    reward = r1 + r2
                    if done:
                        d=d+1
                        self.memory.append((state_now,state_next,reward,action_now,-1))
                        if c%10==0:
                            print("执行步数:%d"%(d))
                            print("贪婪度：%f"%(self.greedy))
                            print("deta:",self.deta)
                        break
                    else:
                        d=d+1
                        self.memory.append((state_now,state_next,reward,action_now,1))
                        state_now=state_next

                self.memory.reverse()
                R=0
                for e in self.memory:
                    train_x=[]
                    train_y=[]
                    R=self.r*R+e[2]
                    lr=self.partial_derivative_lr(e[0],e[3])
                    A=R-self.net1.forecast([list(e[0])])[0]
                    self.w1_action_lr=self.w1_action_lr+self.deta*A*lr
                    train_x.append(e[0])
                    train_y.append(R)
                    self.net1(train_x,train_y)

    a=REINFORCE()
    a.a3c_syschronous()
    # i=10000
    # for e in range(i):
    #     a.train_A_all(e)
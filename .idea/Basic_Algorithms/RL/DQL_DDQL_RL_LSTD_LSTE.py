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

#一、将图像变为一维向量
one=False
if (one==True):
    inputs = keras.Input(shape=(784,), name='img')
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='sigmoid')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
    model.summary()

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    # 如果你的 targets 是 one-hot 编码，用 categorical_crossentropy
    #     one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]
    # 如果你的 tagets 是 数字编码 ，用 sparse_categorical_crossentropy
    # 　　数字编码：2, 0, 1
    #crossentropy
    #H(p=[1,0,0],q=[0.5,0.4,0.1])=−(1∗log0.5+0∗log0.4+0∗log0.1)≈0.3

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=5,
                        validation_split=0.2)

    model.save('./path_to_my_model.h5')
    del model

    model = keras.models.load_model('./path_to_my_model.h5')
    test_scores = model.evaluate(x_test, y_test, verbose=2)

    # print('Test loss:', test_scores[0])
    # print('Test accuracy:', test_scores[1])
    print(x_test[:1])
    print(model.predict(x_test[:1]))

# 二、2维度的卷积和池化
one=False
if (one==True):
    inputs = keras.Input(shape=(32, 32, 3), name='img')
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='toy_resnet')
    model.summary()
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#三、DQL深度强化学习例子
one=False
if (one==True):
    class net:
        def __init__(self):
            self.learning_rate=0.001
            self.inputs = keras.Input(shape=(4,), name='img')
            self.x = layers.Dense(128,activation='relu')(self.inputs)
            self.x = layers.Dense(64, activation='relu')(self.x)
            self.outputs = layers.Dense(2,activation='linear')(self.x) #move left or right
            self.model = keras.Model(inputs=self.inputs, outputs=self.outputs, name='DQL_model')
            self.model.compile(loss='mse',
                               optimizer='adam',
                               metrics=['accuracy'])
            if os.path.exists(r'path_to_my_model.h5'):
                print("加载模型")
                self.model=keras.models.load_model('./path_to_my_model.h5')
            else:
                self.model.save('./path_to_my_model.h5')
                self.model.summary()

        def __call__(self, x_train, y_train):
            x=tf.cast(x_train,dtype=tf.float32)
            y=tf.cast(y_train,dtype=tf.float32)
            del self.model
            self.model=keras.models.load_model('./path_to_my_model.h5')
            history = self.model.fit(x, y,
                                     epochs=1,verbose=0)
            self.model.save('./path_to_my_model.h5')

        def action(self,state):
            return self.model.predict(state)

    class gym_o:
        def __init__(self):
            self.env = gym.make('CartPole-v0')
            self.epsilon=0.1
            self.epsilon_min=0.03
            self.epsilon_decay = 0.955
            #记录样本
            self.memory=[]
            #初始化网络
            self.DQL_net=net()

        def new_done(self,state=[]):
            result=False
            if state[0]>self.env.observation_space.high[0] or state[0]<self.env.observation_space.low[0] or state[2]>0.20198621771937624 or state[2]<-0.20198621771937624:
                result=True
            return result

        def smaple(self,memory_num,N): #N是迭代次数
            self.memory=[]
            if self.epsilon >self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            print("贪婪随机值概率:",self.epsilon)
            sum_O=0
            num=0
            false_num=0
            d=0
            while True:
                state_now = self.env.reset()
                d=0
                num=num+1
                for t in range(200):
                    self.env.render()
                    #获取行动方式
                    # 贪婪算法
                    d=d+1
                    p=np.random.rand()#贪婪算法
                    action=0
                    if p<self.epsilon:
                        action = random.randrange(2)
                    else:
                        temp_now=self.DQL_net.action([list(state_now)])
                        temp_now=list(temp_now[0])
                        action=temp_now.index(max(temp_now)) #左=0,右=1
                    state_next, reward, done, info = self.env.step(action)
                    # done=self.new_done(state_next)
                    if done:
                        # print("第%d轮迭代,本次执行%d次,以后失败"%(N,d))
                        reward=-400
                        self.memory.append((state_now,state_next,reward,action))
                        memory_num=memory_num-1
                        false_num=false_num+1
                        break
                    else:
                        reward=1
                        self.memory.append((state_now,state_next,reward,action))
                        memory_num=memory_num-1
                        if memory_num<1:
                            sum_O=d+sum_O
                            self.env.close()
                            print("第%d轮迭代,总共全轨迹%d条(失败%d条),总执行%d次，每次路径平均执行次数%f次,获取样本%d"%(N,num,false_num,sum_O,(float(sum_O))/num,self.memory.__len__()))
                            return self.memory
                        state_now=state_next
                sum_O=d+sum_O
                if memory_num<1:
                    self.env.close()
                    print("第%d轮迭代,总共全轨迹%d条(失败%d条),总执行%d次，每次路径平均执行次数%f次,获取样本%d"%(N,num,false_num,sum_O,(float(sum_O))/num,self.memory.__len__()))
                    return self.memory
            return self.memory

        def train_net(self,probility_z=1.0,probility_f=1.0):#正样本采样的比例
            #负样本全部采样，正样本按比例采样
            # sample_from_memory_z=[] #正样本
            # sample_from_memory_f=[] #负样本
            #
            # for e in self.memory:
            #     if e[2]<0:
            #        p=np.random.rand()
            #        if p<probility_f:
            #           sample_from_memory_f.append(e)
            #     else:
            #        p=np.random.rand()
            #        if p<probility_z:
            #           sample_from_memory_z.append(e)
            #
            # #安装bellmen方程进行更新样本值
            # sample_from_memory=sample_from_memory_z+sample_from_memory_f
            # print("本次训练样本个数:",sample_from_memory.__len__())
            x_train=[]
            y_train=[]
            # pra=0.05
            # if N*(10.0/50.0)<0.95:
            #      pra=N*(10.0/50.0)
            # else:
            #      pra=0.95
            sample_from_memory=self.memory.copy()
            print("样本数:",sample_from_memory.__len__()*0.50)
            length=int(self.memory.__len__()*0.50)
            random.shuffle(sample_from_memory)
            sample_from_memory=sample_from_memory[:length]
            left_Qvalue=0.0
            right_Qvalue=0.0
            for e in sample_from_memory:
                if e[3]==0: #左移动作
                    right_Qvalue=self.DQL_net.action([list(e[0])])[0][1]
                    if e[2]>0:
                        # left_Qvalue=left_Qvalue*pra+(1-pra)*(e[2]+0.90*max_Qvalue_next)
                        max_Qvalue_next=np.amax(self.DQL_net.action([list(e[1])])[0])
                        left_Qvalue=e[2]+0.90*max_Qvalue_next
                    else:
                        left_Qvalue=e[2]
                else:#右移动作
                    left_Qvalue=self.DQL_net.action([list(e[0])])[0][0]
                    if e[2]>0:
                        #right_Qvalue=right_Qvalue*pra+(1-pra)*(e[2]+0.90*max_Qvalue_next)
                        max_Qvalue_next=np.amax(self.DQL_net.action([list(e[1])])[0])
                        right_Qvalue=e[2]+0.90*max_Qvalue_next
                    else:
                        right_Qvalue=e[2]
                #self.DQL_net([e[0]],[[left_Qvalue,right_Qvalue]])
                y_train.append([left_Qvalue,right_Qvalue])
                x_train.append(e[0])
            return  x_train,y_train
    g=gym_o()
    i=0
    while i<100:
        g.smaple(500,i)
        x,y=g.train_net()
        g.DQL_net(x,y)
        i=i+1

#三、DDQL深度强化学习例子
one=False
if (one==True):
    class net:
        def __init__(self,filepath):
            self.filepath=filepath
            self.learning_rate=0.001
            self.inputs = keras.Input(shape=(4,), name='img')
            self.x = layers.Dense(128,activation='relu')(self.inputs)
            self.x = layers.Dense(64, activation='relu')(self.x)
            self.outputs = layers.Dense(2,activation='linear')(self.x) #move left or right
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
                                     epochs=2,verbose=0)
            self.model.save(self.filepath)

        def forecast(self,state):
            return self.model.predict(state)

    class DDQN:
        def __init__(self):
            self.Qnet1=net("./Qnet1.h5")
            self.Qnet2=net("./Qnet2.h5")
            self.env = gym.make('CartPole-v0')
            self.r=0.90
            self.greedy=0.05
            self.greedy_min=0.0065
            self.greedy_decay = 0.955
            self.beta=0.70
            self.beta_min=0.05
            self.beta_decay = 0.95
            #记录样本
            self.memory=[]
            #初始化网络

        def action(self,state):
            action=0
            if np.random.rand()<self.greedy:#贪婪算法
                action = random.randrange(2)
            else:
                Qvlaue1=self.Qnet1.forecast([list(state)])
                Qvlaue2=self.Qnet2.forecast([list(state)])
                Qvlaue_left=Qvlaue1[0][0]+Qvlaue2[0][0]
                Qvlaue_right=Qvlaue1[0][1]+Qvlaue2[0][1]
                action=list([Qvlaue_left,Qvlaue_right]).index(max(list([Qvlaue_left,Qvlaue_right])))
            return action

        def maxQvalue1(self,state):
            Qvlaue1=self.Qnet1.forecast([list(state)])
            Qvlaue_left=Qvlaue1[0][0]
            Qvlaue_right=Qvlaue1[0][1]
            Qvalue=max(Qvlaue_left,Qvlaue_right)
            return Qvalue

        def maxQvalue2(self,state):
            Qvlaue2=self.Qnet2.forecast([list(state)])
            Qvlaue_left=Qvlaue2[0][0]
            Qvlaue_right=Qvlaue2[0][1]
            Qvalue=max(Qvlaue_left,Qvlaue_right)
            return Qvalue

        def sample(self,memory_num,N):

            if self.greedy*self.greedy_decay<self.greedy_min:
                self.greedy=self.greedy_min
            else:
                self.greedy=self.greedy*self.greedy_decay
            print("采样greedy:=%f"%(self.greedy))
            sum_O=0
            num=0
            false_num=0
            d=0
            self.memory=[]
            while True:
                d=0
                num=num+1
                state_now = self.env.reset()
                for t in range(200):
                    d=d+1
                    self.env.render()
                    action_now=self.action(state_now)
                    state_next,reward,done,info = self.env.step(action_now)
                    x, x_dot, theta, theta_dot = state_next
                    r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                    r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                    reward = r1 + r2
                    if done or x<-1.0 or x>1.0:
                        # reward=-400
                        self.memory.append((state_now,state_next,reward,action_now,-1))
                        memory_num=memory_num-1
                        false_num=false_num+1
                        break
                    else:
                        # reward=1
                        self.memory.append((state_now,state_next,reward,action_now,1))
                        memory_num=memory_num-1
                        if memory_num<1:
                            sum_O=d+sum_O
                            self.env.close()
                            print("第%d轮迭代,总共全轨迹%d条(失败%d条),总执行%d次，每次路径平均执行次数%f次,获取样本%d"%(N,num,false_num,sum_O,(float(sum_O))/num,self.memory.__len__()))
                            return self.memory
                        state_now=state_next
                sum_O=d+sum_O
                if memory_num<1:
                    self.env.close()
                    print("第%d轮迭代,总共全轨迹%d条(失败%d条),总执行%d次，每次路径平均执行次数%f次,获取样本%d"%(N,num,false_num,sum_O,(float(sum_O))/num,self.memory.__len__()))
                    return self.memory
            return self.memory

        def train_net(self,i):#正样本采样的比例
            #负样本全部采样，正样本按比例采样
            # sample_from_memory_z=[] #正样本
            # sample_from_memory_f=[] #负样本
            #
            # for e in self.memory:
            #     if e[2]<0:
            #        p=np.random.rand()
            #        if p<probility_f:
            #           sample_from_memory_f.append(e)
            #     else:
            #        p=np.random.rand()
            #        if p<probility_z:
            #           sample_from_memory_z.append(e)
            #
            # #安装bellmen方程进行更新样本值
            # sample_from_memory=sample_from_memory_z+sample_from_memory_f
            # print("本次训练样本个数:",sample_from_memory.__len__())
            if self.beta*self.beta_decay<self.beta_min:
                self.beta=self.beta_min
            else:
                self.beta=self.beta*self.beta_decay
            print("训练beta:=%f"%(self.beta))
            x_train_1=[]
            y_train_1=[]
            x_train_2=[]
            y_train_2=[]
            sample_from_memory=self.memory.copy()
            length=0
            if i<1000:
                length=int(self.memory.__len__()*0.5)
                print("样本数:",sample_from_memory.__len__()*0.5)
            else:
                length=int(self.memory.__len__()*0.5)
                print("样本数:",sample_from_memory.__len__()*0.5)
            random.shuffle(sample_from_memory)
            sample_from_memory=sample_from_memory[:length]
            y=0
            for e in sample_from_memory:
                if np.random.rand()<0.5:#选择net1采样，e为四元组
                    y=self.Qnet1.forecast([list(e[0])])[0]#用net1预测y
                    if e[4]<0:#当前状态位结束状态，更新y值
                        if i>1:
                            y[e[3]]=y[e[3]]+self.beta*(e[2]-y[e[3]])#采样差分算法
                        else:
                            y[e[3]]=e[2]#不采用差分算法
                    else:#当前状态位不为结束状态
                        if i>1:
                            y[e[3]]=y[e[3]]+self.beta*(e[2]+self.r*self.maxQvalue2(e[1])-y[e[3]])
                        else:
                            y[e[3]]=e[2]+self.r*self.maxQvalue2(e[1])
                    g.Qnet1([e[0]],[y]) #根据神经网计算值，更新样本y
                    y_train_1.append(y)
                    x_train_1.append(e[0])
                else:#选择net2采样
                    y=self.Qnet2.forecast([list(e[0])])[0]
                    if e[4]<0:
                        if i>1:
                            y[e[3]]=y[e[3]]+self.beta*(e[2]-y[e[3]])
                        else:
                            y[e[3]]=e[2]
                    else:
                        if i>1:
                            y[e[3]]=y[e[3]]+self.beta*(e[2]+self.r*self.maxQvalue1(e[1])-y[e[3]])
                        else:
                            y[e[3]]=e[2]+self.r*self.maxQvalue1(e[1])
                    g.Qnet2([e[0]],[y])
                    y_train_2.append(y)
                    x_train_2.append(e[0])
            return x_train_1,y_train_1,x_train_2,y_train_2

    g=DDQN()
    i=0
    while i<1000:
        if i<20:
            g.sample(300,i)
        else:
            g.sample(200,i)
        x1,y1,x2,y2=g.train_net(i)
        g.Qnet1(x1,y1)
        g.Qnet2(x2,y2)
        i=i+1

#四、REINFORCE方法
one=True
if (one==True):
    class net:
        def __init__(self,filepath):
            self.filepath=filepath
            self.learning_rate=0.001
            self.inputs = keras.Input(shape=(4,), name='img')
            self.x = layers.Dense(64,activation='relu')(self.inputs)
            self.x = layers.Dense(128, activation='sigmoid')(self.x)
            self.x = layers.Dense(64, activation='relu')(self.x)
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
                                     epochs=2,verbose=0)
            self.model.save(self.filepath)

        def forecast(self,state):
            return self.model.predict(state)

    class REINFORCE:
        def __init__(self):
            self.pra_num=10
            self.w1_action_lr=np.random.normal(size=(self.pra_num+1)*2)
            if os.path.exists("./w1_action_lr.txt"):
                self.w1_action_lr=np.array(np.loadtxt("./w1_action_lr.txt"),dtype=float)
            self.state_s=np.random.normal(size=(16))
            self.env = gym.make('CartPole-v0')
            self.deta=0.1
            self.deta_min=0.0001
            self.deta_decay = 0.955
            self.r=0.955

            self.greedy=0.05
            self.greedy_min=0.05
            self.greedy_decay = 0.975
            self.memory=[]
            self.mark=0

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
                    b1=-1.4+1.4/(self.pra_num/2)*g
                    # b1=-1.4+np.random.random()*1.4*2 #平衡杆偏离垂直的角度
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
            # left_pro  = tf.exp(tf.reduce_sum(state_left*self.w1_action_lr))/ \
            #             (tf.exp(tf.reduce_sum(state_left*self.w1_action_lr))+tf.exp(tf.reduce_sum(state_right*self.w1_action_lr)))
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

        def train_actor_critic_online(self):
            for c in range(5000):
                if c%5==0:
                    np.savetxt('./w1_action_lr.txt',self.w1_action_lr, delimiter=',')  # 数组x

                if c%3==0:
                    self.sample_lr()

                d=0
                state_now = self.env.reset()

                if self.deta*self.deta_decay<self.deta_min:
                    self.deta=self.deta_min
                else:
                    if c%10==0:
                        self.deta=self.deta*self.deta_decay
                # if self.greedy*self.greedy_decay<self.greedy_min:
                #    self.greedy=self.greedy_min
                # else:
                #     if c%3==0:
                #         self.greedy=self.greedy*self.greedy_decay
                R=self.r
                while True:
                    train_x=[]
                    train_y=[]
                    self.env.render()
                    action_now=self.action_lr(state_now)
                    state_next,reward,done,info = self.env.step(action_now)
                    x, x_dot, theta, theta_dot = state_next
                    r1 = (0.5 - abs(x))/self.env.x_threshold - 0.8
                    r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                    reward = r1 + r2
                    if done:
                        d=d+1
                        train_x.append(state_now)
                        train_y.append(reward)
                        self.net1(train_x,train_y)
                        lr=self.partial_derivative_lr(state_now,action_now)
                        A=reward-self.net1.forecast([list(state_now)])[0]
                        self.w1_action_lr=self.w1_action_lr+R*self.deta*A*lr
                        if c%5==0:
                            print("执行步数:%d"%(d))
                            print("贪婪度：%f"%(self.greedy))
                            print("deta:%f"%(self.deta))
                        break
                    else:
                        d=d+1
                        train_x.append(state_now)
                        train_y.append(reward+self.r*self.net1.forecast([list(state_next)])[0])
                        self.net1(train_x,train_y)
                        lr=self.partial_derivative_lr(state_now,action_now)
                        A=reward+self.r*self.net1.forecast([list(state_next)])[0]-self.net1.forecast([list(state_now)])[0]
                        self.w1_action_lr=self.w1_action_lr+R*self.deta*A*lr
                        R=self.r*R
                        state_now=state_next

    a=REINFORCE()
    a.train_actor_critic_online()
    # i=10000
    # for e in range(i):
    #     a.train_A_all(e)

#四、LSPD_Q
one=False
if (one==True):
    class car_gym:
        def __init__(self):
            self.pra_num=500
            self.pra=np.random.rand(self.pra_num+2).reshape((self.pra_num+2,1))
            rb=100
            self.i=0
            self.T_o=np.eye(int(self.pra_num+2),dtype=float)*rb
            self.A_o=np.zeros((int(self.pra_num+2),int(self.pra_num+2)),dtype=float)
            self.Z_o=np.zeros((int(self.pra_num+2),1),dtype=float)
            self.env = gym.make('CartPole-v0')
            self.deta=0.05
            self.deta_min=0.0001
            self.deta_decay = 0.955
            self.r=0.90

            self.greedy=0.05
            self.greedy_min=0.005
            self.greedy_decay = 0.955
            self.memory=[]
            temp_list=[]
            for g in range(int(self.pra_num/2)):
                a2=np.random.random()*4.0-2.0 #小车速度
                # a1=-0.25943951023931953+0.25943951023931953*2/(self.pra_num/2)*g
                a1=-0.30943951023931953+np.random.random()*0.30943951023931953*2 #小车位置
                b2=np.random.random()*3.0-1.5 #平衡杆角速度
                # b1=-1.0+2.0/(self.pra_num/2)*g
                b1=-1.5+np.random.random()*1.5*2 #平衡杆偏离垂直的角度
                temp_list.append([b1,b2,a1,a2])
            self.bf_function=np.array(temp_list,dtype=float)
            self.N=1

        def sigmoid(self,x):
            return 1.0/(1+np.exp(-x))

        def BF(self,state_now=[]):
            #left_active
            # BFlist=[np.power(state_now[0]+state_now[1]+state_now[2]+state_now[3],3),
            #                    np.power(state_now[0]+state_now[1],2),
            #                    np.power(state_now[0]+state_now[2],2),
            #                    np.power(state_now[0]+state_now[3],2),
            #                    np.power(state_now[1]+state_now[2],2),
            #                    np.power(state_now[1]+state_now[3],2),
            #                    np.power(state_now[2]+state_now[3],2),
            #                    self.sigmoid(state_now[0]+state_now[1]),
            #                    self.sigmoid(state_now[0]+state_now[2]),
            #                    self.sigmoid(state_now[0]+state_now[3]),
            #                    self.sigmoid(state_now[1]+state_now[2]),
            #                    self.sigmoid(state_now[1]+state_now[3]),
            #                    self.sigmoid(state_now[2]+state_now[3]),
            #                    self.sigmoid(state_now[0]*state_now[1]),
            #                    self.sigmoid(state_now[0]*state_now[2]),
            #                    self.sigmoid(state_now[0]*state_now[3]),
            #                    self.sigmoid(state_now[1]*state_now[2]),
            #                    self.sigmoid(state_now[1]*state_now[3]),
            #                    self.sigmoid(state_now[2]+state_now[3]),
            #             state_now[0],
            #             state_now[1],
            #             state_now[2],
            #             state_now[3],np.exp(state_now[0]),np.exp(state_now[1]),np.exp(state_now[2]),np.exp(state_now[3])]
            BFlist=[]
            value=np.array(state_now,dtype=float)
            for e in list(self.bf_function):
                BFlist.append(np.exp(-np.dot((value-np.array(e,dtype=float)),(value-np.array(e,dtype=float)))/(2.0*0.25)))
            return np.array(BFlist,dtype=float)

        def mulity(self,a,b):
            a1=a.reshape((self.pra.__len__(),1))
            b1=b.reshape((1,self.pra.__len__()))
            return np.dot(a1,b1)

        def active_random(self,state_now):
            if np.random.rand()<self.greedy:
                action =  self.env.action_space.sample()
                return action
            else:
                temp_array=self.BF(state_now)
                zeros_array=np.zeros(int(self.pra_num/2+1),dtype=float)
                #left
                left_value=np.dot(np.array(list([1.0])+list(temp_array)+list(zeros_array),dtype=float),self.pra.reshape(-1))
                #right
                right_value=np.dot(np.array(list(zeros_array)+list([1.0])+list(temp_array),dtype=float),self.pra.reshape(-1))
                if left_value>right_value:
                    return 0
                else:
                    return 1

        def active(self,state_now):
            temp_array=self.BF(state_now)
            zeros_array=np.zeros(int(self.pra_num/2+1),dtype=float)
            #left
            left_value=np.dot(np.array(list([1.0])+list(temp_array)+list(zeros_array),dtype=float),self.pra.reshape(-1))
            #right
            right_value=np.dot(np.array(list(zeros_array)+list([1.0])+list(temp_array),dtype=float),self.pra.reshape(-1))
            if left_value>right_value:
                return 0
            else:
                return 1

        def BF_vector(self,state,action):
            temp_array=self.BF(state)
            zeros_array=np.zeros(int(self.pra_num/2+1),dtype=float)
            if action==1:#right
                #right
                return np.array(list(zeros_array)+list([1.0])+list(temp_array),dtype=float)
            else:
                #left
                return np.array(list([1.0])+list(temp_array)+list(zeros_array),dtype=float)

        def sample_no_init(self,N):
            self.memory=[]
            for i in range(N):
                state_now = self.env.reset()
                d=0
                action_now=self.active_random(state_now)
                while True:
                    self.env.render()
                    state_next,reward,done,info = self.env.step(action_now)
                    x, x_dot, theta, theta_dot = state_next
                    r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold-0.8
                    r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians-0.5
                    reward = r1 + r2
                    if done or x<-1.0 or x>1.0:
                        d=d+1
                        self.memory.append((state_now,state_next,reward,action_now,self.active(state_next)))
                        # if i%200==0:
                        #     if self.greedy>self.greedy_min:
                        #        self.greedy=self.greedy*self.greedy_decay
                        if i%5==0:
                            print("执行步数:%d"%(d))
                            print("贪婪度:%f"%(self.greedy))
                        break
                    else:
                        d=d+1
                        self.memory.append((state_now,state_next,reward,action_now,self.active(state_next)))
                        action_now=self.active_random(state_next)
                        state_now=state_next

        def sample(self):
            for i in range(self.N):
                d=0
                state_now = self.env.reset()
                action_now=self.active_random(state_now)
                while True:
                    self.env.render()
                    state_next,reward,done,info = self.env.step(action_now)
                    # x, x_dot, theta, theta_dot = state_next
                    # r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                    # r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                    # reward = r1 + r2
                    if done or (state_next[0]<-0.20 or state_next[0]>0.20):
                        d=d+1
                        reward=-1
                        self.memory.append((state_now,state_next,reward,action_now,self.active(state_next)))
                        # if i%200==0:
                        #     if self.greedy>self.greedy_min:
                        #        self.greedy=self.greedy*self.greedy_decay
                        if i%50==0:
                            print("执行步数:%d"%(d))
                            print("贪婪度:%f"%(self.greedy))
                        break
                    else:
                        d=d+1
                        reward=0
                        self.memory.append((state_now,state_next,reward,action_now,self.active(state_next)))
                        action_now=self.active_random(state_next)
                        state_now=state_next

        def if_out(self,num):
            i=0
            for e  in self.memory:
                if self.active(e[0])==e[3]:
                    i=i+1
            rato=float(i/num)
            print("匹配正确数量：",rato)
            if  rato>0.9:
                return True
            else:
                return False

        def LSTD_Q(self):
            state_now = self.env.reset()
            action_now= self.active_random(state_now)
            d=0
            while True:
                self.env.render()
                state_next,reward,done,info = self.env.step(action_now)
                x, x_dot, theta, theta_dot = state_next
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                if done or x<-0.5 or x>0.5:
                    d=d+1
                    now=self.BF_vector(state_now,action_now)
                    next=self.BF_vector(state_next,self.active(state_next))
                    self.T_o=self.T_o+self.mulity(now,now)
                    self.A_o=self.A_o+self.mulity(now,next)
                    self.Z_o=self.Z_o+now.reshape((int(self.pra_num+2),1))*reward
                    self.i=self.i+1
                    self.pra=np.dot(np.linalg.inv((self.T_o-self.r*self.A_o)/self.i),self.Z_o/self.i)
                    if self.i%200==0:
                        if self.greedy>self.greedy_min:
                            self.greedy=self.greedy*self.greedy_decay
                    if self.i%5==0:
                        print("执行步数:%d"%(d))
                        print("贪婪度:%f"%(self.greedy))
                        # print(self.pra)
                    break
                else:
                    # reward=1
                    now=self.BF_vector(state_now,action_now)
                    next=self.BF_vector(state_next,self.active(state_next))
                    self.T_o=self.T_o+self.mulity(now,now)
                    self.A_o=self.A_o+self.mulity(now,next)
                    self.Z_o=self.Z_o+now.reshape((int(self.pra_num+2),1))*reward
                    self.i=self.i+1
                    d=d+1
                    action_now=self.active_random(state_next)
                    state_now=state_next

        def LSTD_Q_offline(self):
            for i  in range(10000):
                if i%5==0:
                    if self.greedy>self.greedy_min:
                        self.greedy=self.greedy*self.greedy_decay
                self.sample_no_init(5)
                d=0
                while True:
                    num=0
                    # self.T_o=np.eye(int(self.pra_num+2),dtype=float)*0.001
                    # self.Z_o=np.zeros((int(self.pra_num+2),1),dtype=float)
                    for e in self.memory:
                        now=self.BF_vector(e[0],e[3])
                        next=self.BF_vector(e[1],self.active(e[1]))
                        self.T_o=self.T_o-np.dot(np.dot(self.T_o,self.mulity(now,now-self.r*next)),self.T_o) \
                                 /(1.0+np.dot((now-self.r*next).reshape(-1),np.dot(self.T_o,now.reshape(self.pra_num+2,1)).reshape(-1)))
                        self.Z_o=self.Z_o+now.reshape((int(self.pra_num+2),1))*e[2]
                        num=num+1
                    self.pra=np.dot(num*self.T_o,self.Z_o/num)
                    if d>5:
                        print(self.pra)
                        print(num)
                        break
                    d=d+1

        def LSTE_Q_offline(self):
            for i  in range(1000):
                if i%5==0:
                    if self.greedy>self.greedy_min:
                        self.greedy=self.greedy*self.greedy_decay
                self.sample_no_init(10)
                d=0
                while True:
                    num=0
                    list_temp=[]
                    for e in self.memory:
                        value=self.active(e[1])
                        list_temp.append([e[0],e[1],e[2],e[3],value])
                    for e in list_temp:
                        now=self.BF_vector(e[0],e[3])
                        next=self.BF_vector(e[1],e[4])
                        self.T_o=self.T_o-np.dot(np.dot(self.T_o,self.mulity(now,now)),self.T_o) \
                                 /(1.0+np.dot(now,np.dot(self.T_o,now.reshape(self.pra_num+2,1)).reshape(-1)))
                        self.A_o=self.A_o+self.mulity(now,next)
                        self.Z_o=self.Z_o+now.reshape((int(self.pra_num+2),1))*e[2]
                        num=num+1
                        self.pra=0.5*self.pra+0.5*np.dot((num)*self.T_o,np.dot(self.A_o,self.pra.reshape(self.pra_num+2,1))*self.r/(num)+self.Z_o/(num))
                    if d>5:
                        # print(self.pra)
                        print(num)
                        break
                    d=d+1

        def LSTE_Q(self):
            state_now = self.env.reset()
            action_now= self.active_random(state_now)
            d=0
            temp1=self.pra
            temp2=0
            while True:
                self.env.render()
                state_next,reward,done,info = self.env.step(action_now)
                x, x_dot, theta, theta_dot = state_next
                r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                reward = r1 + r2
                if done or x<-0.25 or x>0.25:
                    d=d+1
                    reward=-4
                    now=self.BF_vector(state_now,action_now)
                    next=self.BF_vector(state_next,self.active(state_next))
                    self.T_o=self.T_o+self.mulity(now,now)
                    self.A_o=self.A_o+self.mulity(now,next)
                    self.Z_o=self.Z_o+now.reshape((self.pra_num+2,1))*reward
                    self.i=self.i+1
                    temp2=np.dot(np.linalg.inv(self.T_o/(self.i+1)),self.r*np.dot(self.A_o/(self.i+1),temp1)+self.Z_o/(self.i+1))
                    self.pra=temp2*0.5+temp1*0.5
                    # if i%200==0:
                    #     if self.greedy>self.greedy_min:
                    #        self.greedy=self.greedy*self.greedy_decay
                    if self.i%5==0:
                        print("执行步数:%d"%(d))
                        print("贪婪度:%f"%(self.greedy))
                        # print(self.pra)
                    break
                else:
                    # reward=1
                    now=self.BF_vector(state_now,action_now)
                    next=self.BF_vector(state_next,self.active(state_next))
                    self.T_o=self.T_o+self.mulity(now,now)
                    self.A_o=self.A_o+self.mulity(now,next)
                    self.Z_o=self.Z_o+now.reshape((self.pra_num+2,1))*reward
                    temp2=np.dot(np.linalg.inv(self.T_o/(self.i+1)),self.r*np.dot(self.A_o/(self.i+1),temp1)+self.Z_o/(self.i+1))
                    temp1=temp2*0.5+temp1*0.5
                    self.i=self.i+1
                    d=d+1
                    action_now=self.active_random(state_next)
                    state_now=state_next


        def TD(self):
            for i in range(10000):
                state_now = self.env.reset()
                d=0
                action_now=self.active_random(state_now)
                mark=0
                while True:
                    self.env.render()
                    state_next,reward,done,info = self.env.step(action_now)
                    now=self.BF_vector(state_now,action_now)
                    state_now=state_next
                    action_next=self.active_random(state_next)
                    action_now=action_next
                    x, x_dot, theta, theta_dot = state_next
                    r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
                    r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
                    reward = r1 + r2
                    if done or (state_next[0]<-1.0 or state_next[0]>1.0):
                        d=d+1
                        # reward=-1
                        self.pra=self.pra+self.deta*(reward-np.dot(now,self.pra.reshape(-1)))*now.reshape((int(self.pra_num+2),1))
                        if i%100==0:
                            if self.greedy>self.greedy_min:
                                self.greedy=self.greedy*self.greedy_decay
                            if self.deta>self.deta_min:
                                self.deta=self.deta*self.deta_decay
                        if i%25==0:
                            print("执行步数:%d"%(d))
                            print("贪婪度:%f"%(self.greedy))
                            print("执行步数:%f"%(self.deta))
                        break
                    else:
                        d=d+1
                        # reward=0
                        next_random=self.BF_vector(state_next,action_next)
                        self.pra=self.pra+self.deta*(reward+self.r*np.dot(next_random,self.pra.reshape(-1))- \
                                                     np.dot(now,self.pra.reshape(-1)))*now.reshape((int(self.pra_num+2),1))


        def train(self,method="LSTD_Q"):
            N=0
            while True:
                N=N+1
                if method=="LSTD_Q":
                    self.LSTD_Q()
                    # if N%10==0:
                    #     if self.greedy>self.greedy_min:
                    #        self.greedy=self.greedy*self.greedy_decay
                else:
                    self.LSTE_Q()
                if N>5000:
                    return 0
    car=car_gym()
    # car.train()
    # env = gym.make('CartPole-v0')
    # print(env.x_threshold)
    # print(env.theta_threshold_radians)
    # exit()
    # car.LSTD_Q_offline()
    # car.LSTE_Q_offline()
    car.TD()

    # print(env.action_space)
    # #> Discrete(2)
    # print(env.observation_space)
    # print(env.observation_space.high)
    # print(env.observation_space.low)

    # print(env.observation_space.high)
    # print(env.observation_space.low)
    # car.train()



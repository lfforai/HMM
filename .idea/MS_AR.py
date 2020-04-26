import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
# import tushare as ts

#生成HMM观察值
class MS_AR_observation:
    # pi初始分布，aij跳转概率分布
    def __init__(self,meanlist=[],stddevlist=[],T=100,pi_o=[],aij=[]):
        self.meanlist=meanlist
        self.stddevlist=stddevlist
        self.pra=list(zip(self.meanlist,self.stddevlist))
        self.pi=pi_o#初始概率
        self.aij=np.array(aij)#跳转概率
        self.T=int(T)#时间time
        self.N=self.pi.__len__()
        self.states=range(self.N)#转移状态
        self.result=[]
        self.record_state=[] #记录样本的状态跳转序列
        self.aij_sample=np.zeros((self.N,self.N))#由样本状态序列生成的频率跳转矩阵
        self.pi_sample=np.zeros((self.N))#样本的概率分布频率分布

    #计算生成样本的跳转概率矩阵aij
    def statistic_staterecords(self):
        len=self.record_state.__len__()
        for t in range(len-1):
            self.aij_sample[int(self.record_state[t]),int(self.record_state[t+1])]=self.aij_sample[int(self.record_state[t]),int(self.record_state[t+1])]+1
        aij_l=tf.reshape(tf.reduce_sum(tf.cast(self.aij_sample,dtype=tf.float32),axis=1),(-1,1))#列相加
        self.aij_sample=(tf.cast(self.aij_sample,tf.float32)/aij_l).numpy()
        for t in range(len):
            self.pi_sample[int(self.record_state[t])]=self.pi_sample[int(self.record_state[t])]+1
        pi_sum=tf.reduce_sum(self.pi_sample)
        self.pi_sample=self.pi_sample/pi_sum

    #生成观察数据
    def genetate(self):
        result=[]
        i=0
        for e in range(int(self.T)):
            if(i==0):#根据pi初始分布生成初始概率
                state_now=np.random.choice(a=self.states,p=self.pi) #初始分布
                result.append(tf.random.normal((1,),self.pra[int(state_now)][0],self.pra[int(state_now)][1]).numpy()[0])
                state_next=np.random.choice(a=self.states,p=self.aij[int(state_now),:])
                self.record_state.append(state_now)
                i=1
            else:#
                state_now=state_next
                self.record_state.append(state_now)
                result.append(tf.random.normal((1,),self.pra[int(state_now)][0],self.pra[int(state_now)][1]).numpy()[0])
                state_next=np.random.choice(a=self.states,p=self.aij[int(state_now),:])
        self.result=result
        self.statistic_staterecords()
        return result

    #画出时许图像
    def paint(self):
        x = np.linspace(0, self.T, self.T)
        y=np.array(self.result)
        plt.title('HMM_prices')
        plt.xlabel('time')
        plt.ylabel('prices')
        plt.plot(x,y)
        plt.show()

HMM_observation=MS_AR_observation([50.0,47.0],[1.5,1.8],200,[0.5,0.5],[[0.7,0.3],[0.6,0.4]])
HMM_observation.genetate()

#由于需要对pi,aij，正态分布mean，var，衰减系数beta等参数进行反向传导所以必须定义为tf.variable
#自己定义loss函数
class selfLoss(tf.keras.losses.Loss):
    def __init__(self,):
        super(selfLoss,self).__init__()

    def __call__(self,y_true,y_pred):
        return y_pred

class MS_AR(tf.keras.Model):
      def __init__(self,mean=[52.0,48.0],var=[1.5,1.8],pi=np.array([0.5,0.5]),aij=np.array([[0.3,0.7],[0.5,0.5]]),MS_AR_obs=HMM_observation,**kwargs):
          super(MS_AR, self).__init__(**kwargs)
          self.obs=np.array([0.0]+MS_AR_obs.result)
          self.obs_2_T=self.obs[1:] #从2-T
          self.obs_1_T_1=self.obs[:-1] #从1-T-1
          self.m=pi.__len__() #state的维度
          self.T=MS_AR_obs.T #观察值的数量
          self.optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
          self.mse_loss_fn=selfLoss()

          #需要训练的参数,都是可以训练的
          self.mean=tf.Variable(mean,dtype=tf.float32)
          self.var=tf.Variable(var,dtype=tf.float32)
          self.bate=tf.Variable(0.85,dtype=tf.float32)
          self.aij=tf.Variable(aij,dtype=tf.float32)
          self.pi=tf.Variable(pi,dtype=tf.float32)

          #中间变量
          self.  Eta=tf.zeros((self.m,self.m,self.T))   #AR(1)的正态分布概率矩阵,观测值在同状态下的概率和前一个观察者有关
          self.Gamma=tf.zeros((self.T,self.m,self.m))
          self.Epsilon=tf.zeros((self.T,self.m))

      def init_Eta(self):#初始化基于观察值的正态分布变量矩阵Eta
          #t-1时刻处于i状态，t时候跳转到j状态
          for i  in range(self.m):# t-1
              for j in range(self.m):# t
                  if(i==0 and j==0):
                      #self.Eta[i,j,:]=tf.exp(-1.0*(self.obs_2_T-(self.mean[j]+self.bate*(self.obs_1_T_1-self.mean[i])))/(2.0*self.var[j]))/tf.sqrt(3.1415926*2.0*self.var[j])
                      #由于tensorflow是不支持对张量进行局部赋值的，因为这样不是一个tf的标准op操作
                      #无法进行反向传导，所以我们用tf.contact进行变量之间的连接
                      self.Eta=tf.cast(tf.exp(-1.0*(self.obs_2_T-(self.mean[j]+self.bate*(self.obs_1_T_1-self.mean[i])))/(2.0*self.var[j]))/tf.sqrt(3.1415926*2.0*self.var[j]),dtype=tf.float32)
                  else:
                      self.Eta=tf.concat([self.Eta,tf.cast(tf.exp(-1.0*(self.obs_2_T-(self.mean[j]+self.bate*(self.obs_1_T_1-self.mean[i])))/(2.0*self.var[j]))/tf.sqrt(3.1415926*2.0*self.var[j]),dtype=tf.float32)],axis=0)
          self.Eta=tf.reshape(self.Eta,(self.m,self.m,self.T))

      #计算r_i_j(t)
      def init_Gamma(self):
          #计算t=1时刻r_i_j(1)
          for t in range(self.T):
              if t==0:#初始化
                 for i in range(self.m):
                     for j in range(self.m):
                         denominator=tf.constant(0.0,dtype=tf.float32) #分母
                         for k in range(self.m):
                             for l in range(self.m):
                                 denominator=denominator+self.Eta[k,l,0]*self.pi[k]*self.aij[k,l]
                         if(i==0 and j==0):
                           self.Gamma=tf.reshape(self.Eta[i,j,0]*self.pi[i]*self.aij[i,j]/denominator,(-1,))
                         else:
                           self.Gamma=tf.concat([self.Gamma,tf.reshape(self.Eta[i,j,0]*self.pi[i]*self.aij[i,j],(-1,))/denominator],axis=0)
                 # tf.Tensor(
                 #     [[0.0403519  0.2646596 ]
                 #      [0.20888762 0.48610085]], shape=(2, 2), dtype=float32)
                 self.Gamma=tf.reshape(self.Gamma,(self.m,self.m))

                 #计算Epsilonself
                 for j in range(self.m):
                     sum1=tf.reduce_sum(self.Gamma[:,j],axis=0)
                     if j==0:
                        self.Epsilon=tf.reshape(sum1,(-1,))
                     else:
                        self.Epsilon=tf.concat([self.Epsilon,tf.reshape(sum1,(-1,))],axis=0)
                 # tf.Tensor([0.24923952 0.75076044], shape=(2,), dtype=float32)
                 self.Epsilon=tf.reshape(self.Epsilon,(1,-1))
                 self.Gamma=tf.reshape(self.Gamma,(-1,))
                 # tf.Tensor(
                 #     [[[0.03844092 0.26882058]
                 #       [0.19899514 0.49374333]]], shape=(1, 2, 2), dtype=float32)
                 # tf.Tensor([[0.23743607 0.76256394]], shape=(1, 2), dtype=float32)
              else:#t=2...T
                 for i in range(self.m):
                      for j in range(self.m):
                          denominator=tf.constant(0.0,dtype=tf.float32) #分母
                          for k in range(self.m):
                              for l in range(self.m):
                                  denominator=denominator+self.Eta[k,l,t]*self.Epsilon[t-1,k]*self.aij[k,l]
                          if(i==0 and j==0):
                              Gammatemp=tf.reshape(self.Eta[i,j,t]*self.Epsilon[t-1,i]*self.aij[i,j]/denominator,(-1,))
                          else:
                              Gammatemp=tf.concat([Gammatemp,tf.reshape(self.Eta[i,j,t]*self.Epsilon[t-1,i]*self.aij[i,j],(-1,))/denominator],axis=0)
                          # tf.Tensor(
                          #     [[0.0403519  0.2646596 ]
                          #      [0.20888762 0.48610085]], shape=(2, 2), dtype=float32)
                          #计算Epsilonself

                 Gammatemp=tf.reshape(Gammatemp,(self.m,self.m))

                 for j in range(self.m):
                     sum1=tf.reduce_sum(Gammatemp[:,j],axis=0)
                     if j==0:
                         Epsilon_temp=tf.reshape(sum1,(-1,))
                     else:
                         Epsilon_temp=tf.concat([Epsilon_temp,tf.reshape(sum1,(-1,))],axis=0)

                 # tf.Tensor([0.24923952 0.75076044], shape=(2,), dtype=float32)
                 Epsilon_temp=tf.reshape(Epsilon_temp,(-1,))

                 #添加到Gamma和Epsilon中国
                 self.Epsilon=tf.reshape(self.Epsilon,(-1,))
                 self.Epsilon=tf.concat([self.Epsilon,Epsilon_temp],axis=0)
                 self.Epsilon=tf.reshape(self.Epsilon,(t+1,self.m))
                 self.Gamma=tf.concat([self.Gamma,tf.reshape(Gammatemp,(-1,))],axis=0)
          self.Gamma=tf.reshape(self.Gamma,(self.T,self.m,self.m))
          # print(self.Epsilon)
          # print(self.Gamma)

      #EM方法训练
      def __call__(self,iter=1):
          # self.Gamma=tf.zeros((self.T,self.m,self.m))
          # self.Eta=tf.zeros((self.m,self.m,self.T))
          i=0
          while(i<iter):
              print('trainable_weights::', self.trainable_weights)
              loss=tf.constant(0.0,dtype=tf.float32)
              for t in range(self.T):
                  loss=loss+tf.matmul(tf.reshape(self.Gamma[t,:,:],(1,-1)),tf.reshape(self.Eta[:,:,t],(-1,1)))
              loss=loss[0][0]
              print("向前传导loss成功::",loss)
              with tf.GradientTape() as tape:
                   loss=self.mse_loss_fn(np.ones(1),loss)
              grads = tape.gradient(loss, self.trainable_weights)
              print("向后传导loss失败grads::",list(grads))
              #输出为none,因为tensorflow不支持aij[i][j]或者tf.gather_nd和tf.slice等tf.varible局部变量反向传导
              #解决的办法是自己写局部varible向后传导的op
              self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
              i=i+1

a=MS_AR()
a.init_Eta()
a.init_Gamma()
a()

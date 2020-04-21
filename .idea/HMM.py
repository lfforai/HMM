import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import tushare as ts

#生成HMM观察值
class genetate_normal_HMM_observation_sequence:
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



#训练HMM模型
class HMM:
      #发射矩阵,(T,m)矩阵
      def  init_P(self,fact=22):
           matrix=np.array(self.HMM_obs.pra)
           for t in range(self.T):
               for j in range(self.m):
                   mean=tf.cast(matrix[j,0],tf.float32)
                   var=tf.cast(matrix[j,1],tf.float32)
                   #正态分布模拟
                   self.P[t,j]=(tf.exp(-1.0*np.power(self.obs[t]-mean,2.0)/(2.0*var))/(tf.math.sqrt(2.0*var*3.1415926))).numpy()
           self.P=np.array(self.P)*fact

      #向前
      def  init_Alpha(self,):
           #用递归算法从1..T求Alpha
           #计算a1..m(1),pi*P(x1)
           self.Alpha=[]
           pi_o=tf.cast(np.reshape(np.array(self.P[0,:]),(-1)),tf.float32)
           p_now=tf.cast(np.reshape(np.array(self.pi),(-1)),tf.float32)
           temp_now=tf.multiply(pi_o,p_now).numpy()
           self.Alpha.append(temp_now)
           #计算a1..m(t), #a（t+1）=a（t）*A*P（x（t+1））
           for t in range(self.T-1):#从2...T
               #P（xt）：对角概率矩阵
               P_eye_xi=tf.linalg.diag(tf.cast(np.reshape(np.array(self.P[t+1,:]),(-1)),tf.float32))
               #a（t+1）=a（t）*A*P（x（t+1））
               temp_now=tf.matmul(tf.matmul(tf.reshape(temp_now,(1,-1)),self.aij),P_eye_xi).numpy()[0]
               self.Alpha.append(temp_now)
           self.Alpha=np.array(self.Alpha)

      #向后
      def init_Beta(self,):
          #用递归算法从1..T求Alpha
          #计算b1..m(T),b(T)=1
          self.Beta=[]
          state_num=self.m
          temp_now=tf.ones((state_num,))
          self.Beta.append(temp_now.numpy())
          #计算a1..m(t), #a（t+1）=a（t）*A*P（x（t+1））
          for t in range(self.T-1):#从2...T
              #P（xt）：对角概率矩阵
              P_eye_xi=tf.linalg.diag(tf.cast(np.reshape(np.array(self.P[self.T-(t+1),:]),(-1)),tf.float32))
              #a（t+1）=a（t）*A*P（x（t+1））
              temp_now=tf.reshape(tf.matmul(tf.matmul(self.aij,P_eye_xi),tf.reshape(temp_now,(state_num,-1))),(-1,)).numpy()
              self.Beta.append(temp_now)
          self.Beta.reverse()#反转list
          self.Beta=np.array(self.Beta)

      def init_L(self,):
          a=tf.reshape(tf.multiply(tf.cast(self.pi,dtype=tf.float32),np.reshape(np.array(self.P[0,:]),(-1))),(1,-1))
          # print(tf.matmul(tf.reshape(self.Alpha[self.T-1,:],(1,-1)),tf.reshape(self.Beta[self.T-1,:],(-1,1))))
          self.L=tf.matmul(a,tf.reshape(self.Beta[0,:],(-1,1))).numpy()[0]


      #r，g(i)=sum（ri（t））
      def init_Gamma(self,):
          for  i in range(self.m):
               for t in range(self.T):
                   self.Gamma[i,t]=self.Alpha[t,i]*self.Beta[t,i]/self.L

      #e,h(i,j)=sum（rij（t））
      def init_Epsilon(self,):
          for  i in range(self.m):
              for j in range(self.m):
                  for t in range(self.T-1):
                      self.Epsilon[i,j,t]=self.Alpha[t,i]*self.aij[i,j]*self.P[t+1,j]*self.Beta[t+1,j]/self.L

      def  __init__(self,HMM_observation_o=[],\
                    mean_js=[],var_js=[],pi_js=[],aij_js=[]
                    ,scale=1.0):
           #初始化状态、转移矩阵、观测值，时间周期等参数
           self.scale=scale
           self.HMM_obs=HMM_observation_o#将HMM训练数据导入
           self.pi=np.array(pi_js.copy(),dtype=float)#假设的初始状态分布
           self.aij=np.array(tf.cast(np.array(aij_js),dtype=tf.float32))#假设的转移概率分布
           self.obs=np.array(self.HMM_obs.result,dtype=float)#观察数据
           self.m=self.HMM_obs.states.__len__()#状态数量 牛市、盘震，熊市
           self.T=int(self.HMM_obs.T)#时间长度

           #EM中需要训练的参数
           self.pra_EM=list(zip(mean_js,var_js)).copy()#EM中需要训练的参数3,4 正态分布的mean和var
           self.HMM_obs.pra=self.pra_EM.copy()#替换点真实的参数
           self.pi_EM=self.pi.copy()#EM中需要训练的参数1，初始状态分布
           self.aij_EM=self.aij.__copy__()#EM中需要训练的参数2 ，转移概率aij

           #迭代计算r，g，e，h等EM算法需要的中间参数
           self.Alpha=[]#向前递归求a（t）
           self.Beta=[]#向后递归求b(t)
           self.L=0;#极大拟然统计量
           self.Gamma=np.zeros((self.m,int(self.T)),dtype=float)#ri（t）计算在状态i上的期望次数
           self.Epsilon=np.zeros((self.m,self.m,int(self.T-1)),dtype=float)
           #状态m和时间期T,shape(m,T),Alpha=[[a1(1),a2(1)...am(1)],[a1(2),a2(2)...am(2)]....[a1(T),a2(T)...am(T)]]
           self.P=np.zeros((int(self.T),self.m),dtype=float)
           #由于P发射矩阵和aij转移矩阵的值都很小，
           # 当求Alpha，L等时候需要连续乘T个小矩阵很容就趋于0了
           #所以需要对P扩大scale倍，进行修正避免发射矩阵过小
           self.init_P(scale)
           self.init_Alpha()
           # print(np.round(self.Alpha,3))
           self.init_Beta()
           # print(np.round(self.Beta[1,:],4))
           self.init_L()#极大拟然值
           if (self.L==0.0):
              print("L 等于0，请重新调整 self.P参数")
              exit()
           self.init_Gamma()
           self.init_Epsilon()
           #其他参数初始化

      def init_again(self):
          self.HMM_obs.pra=list(self.pra_EM).copy()#使用新的mean和var
          self.pi=self.pi_EM.copy() #使用新的初始状态分布
          self.aij=self.aij_EM.copy()#使用新的跳转概率

          #重新计算所有中间参数
          self.init_P(self.scale)
          self.init_Alpha()
          # print(np.round(self.Alpha,3))
          self.init_Beta()
          # print(np.round(self.Beta[1,:],4))
          self.init_L()
          if (self.L==0.0):
              print("L 等于0，请重新调整 self.P参数")
              exit()
          self.init_Gamma()
          self.init_Epsilon()

      #判断是否收敛，跳出EM训练
      def convergence(self,rato_err=0.0001):
          #new-old 列
          aij_l=tf.reshape(self.aij_EM,(-1,1))-tf.reshape(self.aij,(-1,1))
          pi_l=tf.reshape(self.pi_EM,(-1,1))-tf.reshape(self.pi,(-1,1))
          pra_l=tf.reshape(tf.cast(list(self.pra_EM),dtype=tf.float32),(-1,1))-tf.reshape(tf.cast(list(self.HMM_obs.pra),dtype=tf.float32),(-1,1))

          #new-old 行
          aij_h=tf.reshape(aij_l,(1,-1))
          pi_h=tf.reshape(pi_l,(1,-1))
          pra_h=tf.reshape(pra_l,(1,-1))

          #计算精度
          a=(tf.math.sqrt((tf.matmul(aij_h,aij_l)))).numpy()[0][0]
          b=(tf.math.sqrt((tf.matmul(pi_h,pi_l)))).numpy()[0][0]
          c=(tf.math.sqrt((tf.matmul(pra_h,pra_l)))).numpy()[0][0]

          if a+b+c<rato_err:
             return True
          else:
             return False

      def EM(self,itera_num=500):
          index=0
          while(index<itera_num):
              #(1)计算初始分布的新估计值
              self.pi_EM=np.array(self.Gamma[:,0])
              #(2)计算转移概率的新估计值
              g=tf.reduce_sum(self.Gamma[:,:-1],axis=1).numpy()#从1到T-1
              h=tf.reduce_sum(self.Epsilon,axis=2).numpy()
              for i in range(self.m):
                  for j in range(self.m):
                      self.aij_EM[i,j]=h[i,j]/g[i]
              #（3）发射器参数评估
              #3.1计算正态分布
              #新mean
              g_all=tf.reduce_sum(self.Gamma,axis=1).numpy()#从1到T
              u_new=tf.cast(tf.reshape(tf.matmul(self.Gamma,tf.reshape(self.obs,(-1,1))),(-1,))/g_all,dtype=tf.float32).numpy()
              #新var
              var_list=[]
              for i in range(self.m):
                  x_T=tf.reshape(tf.cast(self.obs,dtype=tf.float32)-u_new[i],(-1,1))#列向量
                  x=tf.reshape(tf.cast(self.obs,dtype=tf.float32)-u_new[i],(1,-1))#行向量
                  w=tf.cast(tf.linalg.diag(tf.reshape(self.Gamma[i,:],(-1,))),dtype=tf.float32)
                  var_list.append((tf.cast(tf.matmul(x,tf.matmul(w,x_T))[0]/g_all[i],dtype=tf.float32).numpy())[0])
              self.pra_EM=np.array(list(zip(u_new,var_list)),dtype=float)
              index=index+1
              if  self.convergence()==True:
                  print("精度符合要求跳出：")
                  print("极大拟然估计值",self.L)#P和L很容易就非常小，无法继续计算
                  print("pi:",self.pi)
                  print("mean,var:",list(self.HMM_obs.pra))
                  print("aij:",list(self.aij))
                  print("--------------------------------")
                  print("样本跳转频率:",self.HMM_obs.aij_sample)
                  print("样本分布频率：",self.HMM_obs.pi_sample)
                  print("样本序列初始状体:",self.HMM_obs.record_state[0])
                  exit()
              self.init_again()
              self.scale=self.scale
              #输出结果检验
              print("极大拟然估计值",self.L)#P和L很容易就非常小，无法继续计算
              print("pi:",self.pi)
              print("mean,var:",list(self.HMM_obs.pra))
              print("aij:",list(self.aij))
              print("--------------------------------")
# class sharedate:
#       def __init__(self,nameshare='002253',start='2019-02-10',end='2020-04-19'):
#           e = ts.get_hist_data('002253',start=start,end=end)
# a=list(e['close'])
# print(a)
#生成HMM观察数据
HMM_observation=genetate_normal_HMM_observation_sequence([50.0,47.0],[1.5,1.8],200,[0.5,0.5],[[0.7,0.3],[0.6,0.4]])
HMM_observation.genetate()
# HMM_observation.paint()
b=HMM(HMM_observation,mean_js=[52.0,45.0],var_js=[1.2,2.3],pi_js=[0.3,0.7],aij_js=[[0.5,0.5],[0.5,0.5]]
      ,scale=14.095)
b.EM()
print("over")
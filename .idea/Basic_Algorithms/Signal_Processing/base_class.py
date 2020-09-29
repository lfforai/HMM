import matplotlib.pyplot as plt
from matplotlib import font_manager
my_font = font_manager.FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc")
import numpy as np

class paint_linear:
      def __init__(self,title=""):
          self.title=title

      def __call__(self, x=[],y=[],x_type="x_r_y_fun"):
          if x.__len__()!=y.__len__():
             print("erros!: x must be  same length with funs")
             return False
          if x_type=="x_r_y_fun":
             for i in range(len(x)):
                  x_o=np.linspace(x[i][0],x[i][1],x[i][2])
                  y_o=y[i][0](x_o)
                  plt.plot(x_o,y_o,label=y[i][1])
          elif x_type=="x_o_y_fun":
             for i in range(len(x)):
                  x_o=x[i]
                  y_o=y[i][0](x_o)
                  plt.plot(x_o,y_o,label=y[i][1])
          elif x_type=="x_o_y_o":
              for i in range(len(x)):
                  x_o=x[i]
                  y_o=y[i][0]
                  plt.plot(x_o,y_o,label=y[i][1])
          plt.title(self.title)
          plt.legend()
          plt.show()
          return True

# o=paint_linear([[np.sin,"sin(x)"],[np.cos,"cos(x)"]])
# o([[0,2*np.pi,50],[-np.pi,np.pi,50]])

class discrete2continuity:
      def __init__(self,xrange=[-2,2],num=1000,x_fun=np.sin,g_fun="sinc"):
          self.xrange=xrange #
          self.num=num
          self.x_o=x_fun  #被逼近的连续函数x(t)
          self.g_o=g_fun #逼近插值函数g(t)
          self.x_value=np.linspace(self.xrange[0],self.xrange[1],num)
          self.x_range=self.x_o(self.x_value)
          self.interval=(self.xrange[1]-self.xrange[0])/(num-1) #采样间隔
          print("interval；",self.interval)

      def __call__(self,t):#返回由离散sum(x（n*interval）*x(t-n*interval)的逼近方程x_(t)
          if self.g_o=="sinc":
             g_o_new=self.sinc_o
          elif self.g_o=="triangle":
             g_o_new=self.triangle
          elif self.g_o=="square":
              g_o_new=self.square

          if len(t)==1: #only one
             g_list=[]
             for i in range(self.num):
                  g_list.append(g_o_new(t[0]-self.x_value[i]))
             g_list=np.array(g_list)
             return np.dot(np.array(self.x_range),g_list)
          else:
              result=[]
              for e in t:
                  g_list=[]
                  for i in range(self.num):
                      g_list.append(g_o_new(e-self.x_value[i]))
                  g_list=np.array(g_list)
                  result.append(np.dot(np.array(self.x_range),g_list))
              return result

      def sinc_o(self,t):
        if t==0:
           return 1
        else:
           return np.sin(t*np.pi/self.interval)/(np.pi*t/self.interval)

      def triangle(self,t):
          if np.abs(t)>self.interval:
              return 0.0
          else:
              return 1.0-np.abs(t)/self.interval

      def square(self,t):
              if np.abs(t)>self.interval:
                  return 0.0
              else:
                  return np.abs(t)/self.interval

# o_d=discrete2continuity(xrange=[-20,20],num=100,x_fun=np.sin,g_fun="square")
# x_range=np.linspace(-20,20,200)
# y_fit=o_d(x_range)
# y_rel=list(np.sin(x_range))
# o=paint_linear()
# o([x_range,x_range],[[y_fit,"fit"],[y_rel,"sin"]],x_type="x_o_y_o")
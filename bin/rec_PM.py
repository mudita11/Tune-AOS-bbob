

from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
del absolute_import, division, print_function, unicode_literals
import random
import math
import csv
from numpy.linalg import inv
import shutil
import cec2005real

verbose = 1
print("lion100")

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass


class AOS(object):
    
    def __init__(self,chunk,F1,F,dim,u,X,index,f_min,x_min,best_so_far,best_so_far1,test,n_improvements=0,strategy=4):
        #print("AOS init")
        self.aa=[0,1,2,3]; self.a_list=list(self.aa)
        self.chunk=chunk
        self.F1=F1
        self.F=F
        self.dim=dim
        self.u=u
        self.X=X
        self.index=index
        self.f_min=f_min
        self.x_min=x_min
        self.best_so_far=best_so_far
        self.best_so_far1=best_so_far1
        
        self.n_improvements=n_improvements
        self.strategy=strategy
        
        self.p=np.zeros(int(self.strategy))
        self.Q=np.zeros(int(self.strategy))
        self.reward=np.zeros(int(self.strategy))
        self.tran_matrix=np.random.rand(4,4)
        #self.tran_matrix=[[1 for j in range(int(self.strategy))] for i in range(int(self.strategy))]
        self.tran_matrix[:]=self.tran_matrix/np.sum(self.tran_matrix,axis=1)[:,None]
        
        self.n=np.zeros(int(self.chunk))
        self.opu=np.zeros(int(chunk))
        self.choice=np.zeros(int(self.strategy))
        self.window_delta_f=np.zeros(int(W))
        self.operator=np.zeros(int(W))
        self.window_delta_f1=np.zeros(int(W))
        self.operator1=np.zeros(int(W))
        self.rank=np.zeros(int(W))
        self.area=np.zeros(int(self.strategy))
        self.N=np.zeros(int(self.strategy))
        
    
    def AOSUpdate(self,SI):
        #print("AOS update")
        self.n_improvements=0;
        for i in range(self.chunk):
            if self.F1[i]<=self.F[i]:
                self.n_improvements=self.n_improvements+1
                self.n[i]=(self.best_so_far/self.F1[i])*math.fabs(self.F[i]-self.F1[i])
                delta_f=self.F1[i]#math.fabs(F1[i]-F[i])
                if np.any(self.window_delta_f==np.inf):
                    for value in range(W-1,-1,-1):
                        if self.window_delta_f[value]==np.inf:
                            self.window_delta_f[value]=delta_f;
                            self.operator[value]=self.opu[i];
                            break;
                else:
                    for nn in range(W-1,-1,-1):
                        #print("Inside2,opu[i]:",opu[i])
                        if self.operator[nn]==self.opu[i]:
                            for nn1 in range(nn,0,-1):
                                self.window_delta_f[nn1]=self.window_delta_f[nn1-1]
                                self.operator[nn1]=self.operator[nn1-1];
                            self.window_delta_f[0]=delta_f;
                            self.operator[0]=self.opu[i];
                            #print("opu[i],Inside",opu[i])
                            break;
                        elif nn==0 and self.operator[nn]!=self.opu[i]:
                            if delta_f<np.max(self.window_delta_f):
                                zy=np.argmax(self.window_delta_f) # argmin gives position of value in list that is minimum
                                self.window_delta_f[zy]=delta_f;
                                self.operator[zy]=self.opu[i];
                for j in range(self.dim):
                    self.X[i][j]=self.u[i][j]
                self.F[i]=self.F1[i]
        
        self.index=np.argmin(self.F)
        if self.f_min is None or self.F[self.index] < self.f_min:
            self.x_min, self.f_min = self.X[self.index], self.F[self.index]
        self.best_so_far1=self.f_min
        if self.best_so_far1<self.best_so_far:
            self.best_so_far=self.best_so_far1

        self.Reward(SI)
        self.Quality()
        self.Probability()

class F_AUC(AOS):
    def __init__(self,chunk,F1,F,dim,u,X,index,f_min,x_min,best_so_far,best_so_far1,n_improvements=0):
        super(F_AUC,self).__init__(chunk,F1,F,dim,u,X,index,f_min,x_min,best_so_far,best_so_far1,n_improvements,strategy=4)
        for i in range(self.strategy):
            self.Q[i]=np.Inf
        for i in range(self.chunk):
            self.opu[i]=np.inf
        for i in range(W):
            self.window_delta_f[i]=np.inf
        for i in range(W):
            self.operator[i]=np.inf

    def Selection(self):
        if a_list:
            SI=random.choice(a_list)
            a_list.remove(SI)
            #print("a after removing ",a_list)
        else:
            SI=np.argmax(self.choice);
            self.N=np.zeros(int(self.strategy))
        return SI
    
    def Reward(self,SI):
        # Sort window in descending order
        self.operator1=list(self.operator);
        self.window_delta_f1=list(self.window_delta_f);
        a,b=zip(*sorted(zip(self.window_delta_f1,self.operator1)))#,reverse=True))
        self.window_delta_f1=list(a)
        self.operator1=list(b); #print(window_delta_f1,operator1)
        
        #Application of one strategy might affect the others
        #CreditAssignment.GetReward(i)
        self.rank[0]=1
        for i in range(1,W):
            if round(self.window_delta_f1[i],1)==round(self.window_delta_f1[i-1],1):
                self.rank[i]=self.rank[i-1]
            else:
                self.rank[i]=i+1
        #print("rank,Q,operator1,window_delta_f1,rank,SI",rank,Q,operator1,window_delta_f1,rank,SI)

    def AUC(self,operator1,rank,ii):
        x1=0;y1=0;area=0;delta_r1=0;rr=0;rr1=0;
        #loop on window (just one window for all operators)
        #print("operator1,rank,ii",operator1,rank,ii)
        for r in range(1,W+1): # r:rank-position
            tiesX=0;tiesY=0;
            #print("r: ",r)
            #delta_r1=math.pow(DD,r)*(W-r) # calculate weight of rank position in the area
            #print("delta_r",delta_r)
            delta_r1=(math.pow(2,W-r)-1)/(math.log(1+r))  #adapted_NDCG
            i=0;
            while i<W:
                if operator1[i]==ii: #number of rewards equal to reward ranked r given by op-ii
                    rr=rank[i];
                    flag=False;
                    for dd in range(i+1,W):
                        if rank[dd]==rr:
                            tiesY=tiesY+1
                            xyz=dd;
                            i=xyz;
                            flag=True
                    if not flag:
                        i=i+1
                else:
                    i=i+1;

            i=0;
            while i<W:
                if operator1[i]!=ii:  # rewards equal to reward ranked r given by others
                    rr1=rank[r-1];
                    flag=False;
                    for dd1 in range(i+1,W):
                        if rank[dd1]==rr1:
                            tiesX=tiesX+1;
                            xyz1=dd1
                            i=xyz1;
                            flag=True;
                    if not flag:
                        i=i+1
                else:
                    i=i+1;

            #print("tiesX,tiesY :",tiesX,tiesY)
            if tiesX+tiesY>0:
                for s in range(r+1,r+tiesX+tiesY):
                    #print("DD,s,W",DD,s,W)
                    #delta_r1=delta_r1+(math.pow(DD,s)*(W-s))/(tiesX+tiesY) #sum weights of tied ranks, divided by number of ties
                    delta_r1=delta_r1+((math.pow(2,W-r)-1)/(math.log(1+r)))/(tiesX+tiesY)
                #x1=x1+(tiesX*delta_r)
                area=area+(y1*tiesX*delta_r1);  #print("area",area);#sum the rectangle below
                y1=y1+(tiesY*delta_r1);
                area=area+(0.5*math.pow(delta_r1,2)*tiesX*tiesY); #print("area1",area); #sum the triangle below slanted line
                r=r+tiesX+tiesY-1;
            elif operator1[r-1]==ii: #if op generated r, vertical segment
                y1=y1+delta_r1
            else: #if another operator generated r, horizontal segment
                #x1=x1+delta_r
                area=area+(y1*delta_r1)
        return area

    def Quality(self):
        for ii in range(self.strategy):
            self.Q[ii]=self.AUC(self.operator1,self.rank,ii)
        cat=np.sum(self.Q)
        for ii in range(self.strategy):
            #print("strategy,Q[ii],cat,Q[ii]/cat",strategy,Q[ii],cat,Q[ii]/cat);
            if cat!=0:
                self.Q[ii]=self.Q[ii]/cat # credit = normalized area

    def Probability(self):
        for i in range(self.strategy):
            # N[i]=0;
            for it in range(W):
                if self.operator1[it]==i:
                    self.N[i]=self.N[i]+1;  #the number of times each operator appears in the sliding window
        #for ss in range(strategy):
            #N_sum=N_sum+N[ss]
        for z in range(self.strategy):
            if self.N[z]!=0:
                self.choice[z]=self.Q[z]+C*math.sqrt((2*math.log(np.sum(self.N)))/(self.N[z]))

class PM(AOS):
    
    def __init__(self,chunk,F1,F,dim,u,X,index,f_min,x_min,best_so_far,best_so_far1,n_improvements=0):
        #print("PM update")
        #print("n_improvements",n_improvements)
        super(PM,self).__init__(chunk,F1,F,dim,u,X,index,f_min,x_min,best_so_far,best_so_far1,n_improvements,strategy=4)
        for i in range(self.strategy):
            self.p[i]=1.0/self.strategy
        for i in range(self.strategy):
            self.Q[i]=1.0
        #print("Finished PM-init")

    def Selection(self):
        #print("PM selection")
        if self.a_list:
            #print("a_list",self.a_list)
            SI=random.choice(self.a_list)
            self.a_list.remove(SI)   #print("a after removing ",a_list)
        else:
            SI=np.argmax(np.random.multinomial(1,self.p,size=1))
        return SI

    def Reward(self,SI):
        #print("PM reward")
        for i in range(self.strategy):
            if i==SI and self.n_improvements!=0:
                self.reward[i]=np.sum(self.n)/self.n_improvements;
            elif i==SI and n_improvements==0:
                self.reward[i]=0
            else:
                self.reward[i]=0
    
    def Quality(self):
        self.Q=(1-alpha)*self.Q+alpha*self.reward

    def Probability(self):
        #print("PM probability")
        for i in range(self.strategy):
            if np.sum(self.Q)!=0:
                self.p[i]=p_min+(1-self.strategy*p_min)*(self.Q[i]/np.sum(self.Q))
        #print("p in pm: ",self.p)


class Rec_PM(PM):
    def __init__(self,chunk,F1,F,dim,u,X,index,f_min,x_min,best_so_far,best_so_far1,n_improvements=0):
        #print("Rec-PM init")
        super(Rec_PM,self).__init__(chunk,F1,F,dim,u,X,index,f_min,x_min,best_so_far,best_so_far1,n_improvements=0)
        #print("Finished rec_PM-init")
    
    def Selection(self):
        #print("Rec-PM selection")
        SI=PM.Selection(self)
        return SI
    
    def Reward(self,SI):
        #print("Rec-PM reward")
        self.reward *= 0.5
        self.reward[SI]+=(self.n_improvements/self.chunk);
        #print("End of reward calculation: ",reward2)

    def Quality(self):
        #print("Rec-PM Quality")
        self.Q=np.matmul(np.linalg.pinv(np.array((1-alpha*np.array(self.tran_matrix)))),np.array(self.reward))
        #print("Q: ",self.Q)
        self.Q=self.Q-np.max(self.Q)
        #print("Q1: ",self.Q)
        self.Q=np.exp(self.Q)
        #print("Q2: ",self.Q)
        sum_Q=np.sum(self.Q);self.Q=self.Q/sum_Q
        #print("Q3: ",self.Q)
        #print("Finished calculating Quality")

    def Probability(self):
        #print("Rec-PM Probability")
        last_p=np.copy(self.p)
        PM.Probability(self)
        for i in range(self.strategy):
            for j in range(self.strategy):
                self.tran_matrix[i][j]=self.p[i]+self.p[j]
        self.tran_matrix[:]=self.tran_matrix/np.sum(self.tran_matrix,axis=1)[:,None]
        #print("Finished calculating Probability")
        #print("probability: ",self.p)


def DE(fun, lbounds, ubounds, budget):

    iteration = 0
    
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    
    chunk=NP
    X=lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
    F=[fun(x) for x in X];
    budget-=chunk
    
    u=[[0 for j in range(int(dim))] for i in range(int(chunk))];#print(u)
    F1=np.zeros(int(chunk));
    
    index = np.argmin(F);
    if f_min is None or F[index] < f_min:
        x_min, f_min = X[index], F[index];
    best_so_far=f_min
    best_so_far1=best_so_far

    aos_method = Rec_PM(chunk,F1,F,dim,u,X,index,f_min,x_min,best_so_far,best_so_far1,n_improvements=0)
    
    while budget>0:
        #print("DE")
        SI=aos_method.Selection();
        for i in range(aos_method.chunk):
            r1=random.randint(0,aos_method.chunk-1); #print("r1",r1)
            r2=random.randint(0,aos_method.chunk-1); #print("r2",r2)
            r3=random.randint(0,aos_method.chunk-1); #print("r3",r3)
            r4=random.randint(0,aos_method.chunk-1); #print("r4",r4)
            r5=random.randint(0,aos_method.chunk-1); #print("r5",r5)
            best=np.argmin(aos_method.F);#print("best",best)
            while r1==i:
                r1=random.randint(0,aos_method.chunk-1)
            while r2==i or r2==r1:
                r2=random.randint(0,aos_method.chunk-1)
            while r3==i or r3==r1 or r3==r2:
                r3=random.randint(0,aos_method.chunk-1)
            while r4==i or r4==r1 or r4==r2 or r4==r3:
                r4=random.randint(0,aos_method.chunk-1)
            while r5==i or r5==r1 or r5==r2 or r5==r3 or r5==r4:
                r5=random.randint(0,aos_method.chunk-1)
            jrand = random.randint(0,aos_method.dim-1)
            for j in range(aos_method.dim):
                if random.random()<CR or j==jrand:
                    if SI==0:
                        aos_method.u[i][j]=aos_method.X[r1][j]+FF*(aos_method.X[r2][j]-aos_method.X[r3][j]) # DE/rand/1
                    elif SI==1:
                        aos_method.u[i][j]=aos_method.X[r1][j]+FF*(aos_method.X[r2][j]-aos_method.X[r3][j])+FF*(aos_method.X[r4][j]-aos_method.X[r5][j]) # DE/rand/2
                    elif SI==2:
                        aos_method.u[i][j]=aos_method.X[r1][j]+FF*(aos_method.X[best][j]-aos_method.X[r1][j])+FF*(aos_method.X[r2][j]-aos_method.X[r3][j])+FF*(aos_method.X[r4][j]-aos_method.X[r5][j]) # DE/rand-to-best/2
                    elif SI==3:
                        aos_method.u[i][j]=aos_method.X[i][j]+FF*(aos_method.X[r1][j]-aos_method.X[i][j])+FF*(aos_method.X[r2][j]-aos_method.X[r3][j]) # DE/current-to-rand/1
                else:
                    aos_method.u[i][j]=aos_method.X[i][j]
        aos_method.F1=[fun(x) for x in aos_method.u]

        aos_method.AOSUpdate(SI)

        aos_method.index=np.argmin(aos_method.F)
        if aos_method.f_min is None or aos_method.F[aos_method.index] < aos_method.f_min:
            aos_method.x_min, aos_method.f_min = aos_method.X[aos_method.index], aos_method.F[aos_method.index]
        aos_method.best_so_far1=aos_method.f_min;
        if aos_method.best_so_far1<aos_method.best_so_far:
            aos_method.best_so_far=aos_method.best_so_far1
        
        if iteration%1000==0:
            print("iteration",iteration)
            print("r: ",aos_method.reward,np.shape(aos_method.reward))
            print("Q: ",aos_method.Q)
            print("probability: ",aos_method.p)
            print("tran matrix: ",aos_method.tran_matrix)
        iteration=iteration+1
        budget-=aos_method.chunk
        #print("iteration :",iteration)
    cost = aos_method.best_so_far;
    print("\n",cost)
    return aos_method.best_so_far
    

#print("lion1", sys.argv[0])
FF = float(sys.argv[1]); print(FF)
NP = int(sys.argv[2]); print(NP)
CR = float(sys.argv[3]); print(CR)
alpha = float(sys.argv[4]); print(alpha)
p_min = float(sys.argv[5]); print(p_min)
dim = int(sys.argv[6]); print("D= ",dim)
instance = int(sys.argv[7]); print(instance)
#print("lion2")
instance = [instance]; #print("inst2",instance)
budget = 1e4
W=50;C=0.5

fbench = Function(instance, dim)
info = fbench.info(); print(info)
fun = fbench.get_eval_function()
lbounds = np.repeat(info['lower'], dim); lbounds = np.array(lbounds)
ubounds = np.repeat(info['upper'], dim); ubounds = np.array(ubounds)


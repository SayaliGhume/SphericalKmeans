# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:22:31 2017

@author: mahat
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.matlib as npm
import scipy as sp
import sys
from numpy.linalg import svd
import itertools

def randVMF (N, mu ,k):
    if(np.linalg.norm(mu,2)<(1-0.0001) or np.linalg.norm(mu,2)>(1+0.0001)):
        sys.exit('Mu must be unit vector')
    else:
        p = len(mu)
        tmpMu = [1]
        tmpMu = np.append(tmpMu, np.zeros([1,(p-1)]))
        t = randVMFMeanDir(N,k,p)
        RandSphere = randUniformSphere(N,p-1)
        temp1 = np.zeros([N,1])
        temp = np.concatenate((temp1,RandSphere), axis=1)       
        RandVMF = npm.repmat(t,1,p)*npm.repmat(tmpMu,N,1)+npm.repmat((1-t**2)**0.5,1,p)*temp 
        Otho = nullspace(mu)
        tmu = np.array(mu)[np.newaxis]
        Rot = np.concatenate((tmu.T,Otho), axis=1)
        RandVMF = np.transpose(np.dot(Rot,np.transpose(RandVMF)))                            
        return RandVMF
    
    
def randVMFMeanDir (N, k, p):
    min_thresh = 1/(5*N)
    xx = np.arange(-1,1,0.000001)    
    yy = VMFMeanDirDensity(xx,k,p)
    cumyy = np.cumsum(yy)*(xx[2]-xx[1])
    for i in range (0,len(cumyy)):
        if cumyy[i]>min_thresh:
            leftbound = xx[i]
            break
        else:
            continue
    xx = np.linspace(leftbound,1.0,num = 1000)
    xx = list(xx)
    yy = VMFMeanDirDensity(xx,k,p)
    M = max(yy)
    t = np.zeros([N,1])
    for i in range (0,N):
        while(1):
            x = np.random.random()*(1-leftbound)+leftbound
            x = [x]
            h = VMFMeanDirDensity(x,k,p)
            draw = np.random.random()*M
            if(draw <= h):
                break
        t[i] = x    
    return t


def VMFMeanDirDensity(x, k, p):  
    
    for i in range (0,len(x)):
        if(x[i]<-1.0 or x[i]>1.0):
            sys.exit('Input of x must be within -1 to 1')
    coeff = (k/2)**(p/2-1) * (sp.special.gamma((p-1)/2)*sp.special.gamma(1/2)*sp.special.iv(p/2-1,k))**(-1);
    y = coeff * np.exp(k*x)*np.power((1-np.square(x)),((p-2)/2))
    return y


def randUniformSphere(N,p):
    randNorm = np.random.normal(np.zeros([N,p]),1,size = [N,p])
    RandSphere = np.zeros([N,p])
    for r in range(0,N):
        RandSphere[r,:] = randNorm[r,:]/np.linalg.norm(randNorm[r,:])
    return RandSphere

#http://scipy-cookbook.readthedocs.io/items/RankNullspace.html
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns
    

##############################  MAIN FUNCTION  ################################ 

data = randVMF(1000,[0,0,0,1],1)
    
#mention the number of clusters
no_clusters=len(data[0])
    
##clusture the data using the von-mises fisher distribution
#getting the size of the data
H=len(data)
W=len(data[0])
    
#calculating the mean to remove the dc component
sum_sample=sum(data)
norm_sample=np.linalg.norm(sum_sample)
mean_sample=(sum_sample)/(norm_sample)

#calculating global mean to get centroids of the clusters
deviation=0.01
mean_global = np.zeros([W,W])
for i in range (0,no_clusters):
    random_sample=np.random.rand(1,W)-0.5
    random_norm=deviation*(np.random.rand())
    random_sample2=(random_norm*random_sample)/np.linalg.norm(random_sample)
    temp = mean_sample+random_sample2
    mean_global[i,:] = temp/np.linalg.norm(temp)
    
#calculating mean from spherical kmeans

difference=1
epsilon=0.01
number=100
iteration=0



while (difference>epsilon):
    #check
    iteration=iteration+1
    number2=number
    
    #computing the nearest neighbour and assigning the points
    mean_global2 = np.transpose(mean_global)
    value=np.dot(data, mean_global2)
    value_max=value.max(1)
    clusters=value.argmax(1)
    
    #computing value of the function
    number=sum(value_max)    
    
    sum_sample3 = np.zeros([4,3])
    #computing centroids for the clusters
    for i in range(0,no_clusters):
        sum_sample3=sum(data[np.where(clusters==i)])
        temp=sum_sample3/np.linalg.norm(sum_sample3)
        mean_global[i,:] = temp
        
    difference=abs(number-number2)
    
#displaying the clusters
plt.plot(clusters, marker = '.', linestyle = '')
#Plot first 3 colums of data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = itertools.cycle(["r", "b", "g", "k", "m", "y"])
marker = itertools.cycle(["o", "*", "+", "x", "D", "s", "^"])

for i in range (0,no_clusters):
    a = np.zeros([1,W])
    j=0
    for p in range (0,H):
        if clusters[p] == i:
            a = np.vstack((a,data[p,:]))
            j=j+1
    a = np.delete(a,(0),axis=0)
    ax.scatter(a[:,0],a[:,1],a[:,2], c=next(colors), marker=next(marker))
plt.show()
    
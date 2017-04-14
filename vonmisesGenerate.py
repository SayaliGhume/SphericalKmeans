# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:20:37 2017

@author: Bhavana
"""


import numpy as np
import numpy.matlib as npm
import sys
from numpy.linalg import svd
import mpmath

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
    min_thresh = (1/(5*N))
    xx = np.arange(-1,1,0.000001)    
    yy = VMFMeanDirDensity(xx,k,p)
    cumyy = (np.cumsum(yy)*(xx[2]-xx[1]))
    for i in range (0,len(cumyy)):
        #print(cumyy[0])
        if cumyy[i]>min_thresh:
            leftbound = xx[i]
            #print(i)
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
    coeff = float((k/2)**(p/2-1) * (mpmath.gamma((p-1)/2)*mpmath.gamma(1/2)*mpmath.besseli(p/2-1,k))**(-1));#sp.special.gamma((p-1)/2) becoming infinity
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
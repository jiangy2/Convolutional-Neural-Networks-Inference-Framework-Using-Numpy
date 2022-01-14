# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 00:47:47 2019

@author: admin
"""

import numpy as np


#def maxout(z,cof):
#    
#    z_temp = z.ravel()
#    z_group = [z_temp[i:i+cof] for i in range(0,len(z_temp),cof)]
#    out =  np.random.random_sample(size=(len(z_group))).astype(np.float32)
#
#    
#    for i in range(len(z_group)):
#
#        out[i] = z_group[i].max()
#    
#    return out.reshape(1,-1)

def maxout(x,max_out):

    out_shape = (1,int(x.shape[1] / max_out))
    y = np.zeros(out_shape)
    y = np.max(x.reshape(-1, 2), 1)
    return y.reshape(1,-1)


def forward_step(a,n,k):
    maxout_cof = 2
    dim = maxout_cof*n
    out = np.random.random_sample(size=(1,dim)).astype(np.float32)
    
    for i in range(k):
        W = np.random.random_sample(size=(a.shape[1],int(dim/k))).astype(np.float32)
        out[0,i*int(dim/k):(i+1)*int(dim/k)] = np.ravel(a.dot(W))
        
    return out

def forward(a,n):
    W = np.random.random_sample(size=(a.shape[1],n)).astype(np.float32)
    return a.dot(W)

def forward_step_alex(a,dim,k):

    out = np.random.random_sample(size=(1,dim)).astype(np.float32)
    
    for i in range(k):
        W = np.random.random_sample(size=(a.shape[1],int(dim/k))).astype(np.float32)
        out[0,i*int(dim/k):(i+1)*int(dim/k)] = np.ravel(a.dot(W))
        
    return out    
    


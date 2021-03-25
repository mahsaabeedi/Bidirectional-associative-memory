# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:12:24 2019

@author: Mahsa
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:52:11 2019

@author: sakin
"""

# =============================================================================
# get weight matrix
# =============================================================================
import numpy as np
a=np.array([[-1,1,-1],[1,-1,1],[1,1,1],[1,-1,1],[1,-1,1]])
b=np.array([[1,1,-1],[1,-1,1],[1,1,-1],[1,-1,1],[1,1,-1]])
c=np.array([[-1,1,1],[1,-1,-1],[1,-1,-1],[1,-1,-1],[-1,1,1]])
a_noisy=np.array([[1,1,-1],[1,-1,1],[-1,1,1],[1,-1,1],[1,-1,1]])
b_noisy=np.array([[1,-1,-1],[1,-1,1],[1,1,-1],[-1,-1,1],[1,1,1]])
c_noisy=np.array([[1,1,-1],[1,1,-1],[-1,-1,-1],[1,-1,-1],[-1,-1,1]])
a_noisyt=np.reshape(a_noisy,(1,15))
b_noisyt=np.reshape(b_noisy,(1,15))
c_noisyt=np.reshape(c_noisy,(1,15))
a_to=np.array([1,1,-1])
a_to=np.reshape(a_to,(1,3))
b_to=np.array([1,1,1])
b_to=np.reshape(b_to,(1,3))
c_to=np.array([-1,1,1])
c_to=np.reshape(c_to,(1,3))
sa=np.reshape(a,(15,1))
sb=np.reshape(b,(15,1))
sc=np.reshape(c,(15,1))
# =============================================================================
# taranahade
# =============================================================================
sat=np.reshape(sa,(1,15))
sbt=np.reshape(sb,(1,15))
sct=np.reshape(sc,(1,15))

wa=np.dot(sa,a_to)
wb=np.dot(sb,b_to)
wc=np.dot(sc,c_to)
w=wa+wb+wc

# =============================================================================
# verification for original pattern
# =============================================================================
a_to_new=np.dot(a_noisyt,w)
b_to_new=np.dot(b_noisyt,w)
c_to_new=np.dot(a_noisyt,w)
# =============================================================================
# apply activation function
# =============================================================================
for i in range(3):
    if a_to_new[0][i]>0:
        a_to_new[0][i]=1
    else:
        a_to_new[0][i]=-1
    if b_to_new[0][i]>0:
        b_to_new[0][i]=1
    else:
        b_to_new[0][i]=-1
    if c_to_new[0][i]>0:
        c_to_new[0][i]=1
    else:
        c_to_new[0][i]=-1

if (np.array_equal(a_to,a_to_new) and np.array_equal(b_to,b_to_new) and np.array_equal(c_to,c_to_new)):
    print("weight matrix is correct")
    
# =============================================================================
# output vector a and b are corrrect, but the output of c is incorrect and do bam algorithm 
# =============================================================================
x=np.array([1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1])
x=np.reshape(x,(1,15))
y=np.array([1,1,-1])
y=np.reshape(y,(1,3))
yin=np.zeros(1*3)
yin=np.reshape(yin,(1,3))
xin=np.zeros(1*15)
xin=np.reshape(xin,(1,15))
for k in range(1):
    for j1 in range(3):
        sum=0
        for i1 in range(15):
            sum=sum+w[i1][j1]*x[0][i1]
        yin[0][j1]=sum
        if yin[0][j1]>0:
            y[0][j1]=1
        else:
            y[0][j1]=-1
    for i2 in range(15):
        sum=0
        for j2 in range(3):
            sum=sum+w[i2][j2]*y[0][j2]
        xin[0][i2]=sum
        if xin[0][i2]>0:
            x[0][i2]=1
        else:
            x[0][i2]=-1
l1=np.array_equal(x,sct)
l2=np.array_equal(y,c_to)
if l1 and l2:
    print("finish")
    print("epoch:",k+1)
    
    
# =============================================================================
# input [-1,1,0]
# =============================================================================
x=sct
y=np.array([-1,1,0])
y=np.reshape(y,(1,3))
yin=np.zeros(1*3)
yin=np.reshape(yin,(1,3))
xin=np.zeros(1*15)
xin=np.reshape(xin,(1,15))
for k in range(1):
    for j1 in range(3):
        sum=0
        for i1 in range(15):
            sum=sum+w[i1][j1]*x[0][i1]
        yin[0][j1]=sum
        if yin[0][j1]>0:
            y[0][j1]=1
        else:
            y[0][j1]=-1
    for i2 in range(15):
        sum=0
        for j2 in range(3):
            sum=sum+w[i2][j2]*y[0][j2]
        xin[0][i2]=sum
        if xin[0][i2]>0:
            x[0][i2]=1
        else:
            x[0][i2]=-1
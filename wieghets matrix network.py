# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:52:11 2019

@author: Mahsa
"""

# =============================================================================
# get weight matrix
# =============================================================================
import numpy as np
a=np.array([[-1,1,-1],[1,-1,1],[1,1,1],[1,-1,1],[1,-1,1]])
b=np.array([[1,1,-1],[1,-1,1],[1,1,-1],[1,-1,1],[1,1,-1]])
c=np.array([[-1,1,1],[1,-1,-1],[1,-1,-1],[1,-1,-1],[-1,1,1]])
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
a_to_new=np.dot(sat,w)
b_to_new=np.dot(sbt,w)
c_to_new=np.dot(sct,w)
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
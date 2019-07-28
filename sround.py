#!/usr/bin/env python
# coding: utf-8

# # Stochastic Rounding Tutorial 

# ## 1. Unoptimised implementation

# First import numpy and timing libraries

# In[1]:


import numpy as np
from timeit import timeit

# computes elementwise relative error of elements of lists x1 and x2
def relative_error(x1, x2):
    ret = []
    for (i, r) in zip(x1, x2):
        ret.append(abs(i - r) / r)
    return ret


# Unoptimised implementation of stochastic rounding. 

# In[2]:


# from https://medium.com/@minghz42/what-is-stochastic-rounding-b78670d0c4a
# Generates s randomly rounded values of x. The mean should be x.
def rstoc(x, s):
    r = []
    for i in range(s):
        decimal = abs(x - np.trunc(x))
        random_selector = np.random.random_sample()
        if (random_selector < decimal):
            adjustor = 1
        else:
            adjustor = 0
        if(x < 0):
            adjustor = -1 * adjustor
        r.append(np.trunc(x) + adjustor)
    return r

# For each element x of the list v, compute the mean of rstoc(x, s). Return as another list
def E_seq(v, s):
    return map(lambda x : np.mean(rstoc(x, s)), v)


# Run stochastic rounding for different values of s (and a fixed v). The output is the relative error compared with the expected value.

# In[6]:


spower = 7 	# samples to try (execution time grows exponentiall)
v = [-1.234, 0.1, 0.5, 0.6789]

# this function runs 
def run_rstoc():
    for i in range(spower):
        print(10 ** i, list(relative_error(E_seq(v, 10 ** i), v)))
t_rstoc = timeit(run_rstoc, number=1)
print("Execution time rstoc()", t_rstoc, '\n')


# ## 2. Optimised implementation
# 
# In this part, change frstoc() to be as fast as possible (while computing the correct result)

# Here is a faster implementation in which the s random samples are generated in frstoc

# In[4]:


# replace with your optimised implementation 
def frstoc(x, s):  
    return 0

# For each element x of the list v, compute the mean of rstoc(x, s). Return as another list
def E_vec(v, s):
    return map(lambda x : np.mean(frstoc(x, s)), v)

def run_vec():
    for i in range(spower):
        print(10 ** i, list(relative_error(E_vec(v, 10 ** i), v)))


# Run and check speed

# In[5]:


t_frstoc = timeit(run_vec, number=1)
print("Execution time vec_f_rstoc()", t_frstoc)
print("Speedup", t_rstoc / t_frstoc)


# In[ ]:





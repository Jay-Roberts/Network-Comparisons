import tensorflow as tf 
#import numpy as np 
from blocks import blocks

def f_1(x):
    return x
def f_2(x,y):
    return x+y

d = {(1,1):f_1}
print(d[(1,1)].__name__)
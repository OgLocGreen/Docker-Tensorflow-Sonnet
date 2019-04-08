import numpy as np
import time

for i in range(10) :
    print("test" , i)
    a = np.zeros(5)
    print(a)
    time.sleep(3)

#import tensorflow as tf
#tf.enable_eager_execution()
#print(tf.reduce_sum(tf.random_normal([1000, 1000])))

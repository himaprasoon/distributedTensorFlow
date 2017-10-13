# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
a = tf.constant(3.0, name="A")
b = tf.constant(4.0, name="B")
sum_ab = tf.add(a, b)

print("A :", a)
print("B :", b)

print("sum_ab :", sum_ab)

# For launching tensorboard
logs_path = "logdir"
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

sess = tf.Session() # Creates a Session : This connects the graph to an execution engine
print(sess.run([a, b, sum_ab]))
sess.close()





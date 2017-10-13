# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
# line equation y = M * X
X_DATA = [1.0, 0.0, 2.0, 4.0]
Y_DATA = [3.0, 0.0, 6.0, 12.0]

epoch = 15

x_place_holder = tf.placeholder(dtype=tf.float32, shape=[])
y_place_holder = tf.placeholder(dtype=tf.float32, shape=[])

m = tf.Variable(100.0)

Y = tf.multiply(m, x_place_holder)

cost = tf.abs(Y - y_place_holder)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#     for x, y in zip(X_DATA, Y_DATA):
#         print("\nx, y is :", x, y)
#         print("Output ", sess.run(Y, feed_dict={x_place_holder: x}))


with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        for x,y in zip(X_DATA, Y_DATA):
            print("\nx, y is :", x, y)
            # print("before opt",sess.run(Y, feed_dict={x_place_holder: x}))
            sess.run(optimizer, feed_dict={x_place_holder: x, y_place_holder: y})
            print("after optimisation m ", sess.run(m))


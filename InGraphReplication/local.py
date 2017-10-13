import tensorflow as tf



A = tf.Variable(0, name="Var_A")
o_add = tf.assign_add(A, 1, name="add1_A")


# B = tf.Variable(100,name="Var_B")
t_add = tf.assign_add(A, 1, name="add1_B")

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        input("haha")
        print(sess.run([o_add,t_add]))
        # print(sess.run([t_add]))
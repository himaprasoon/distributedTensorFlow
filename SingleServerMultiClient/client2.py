import tensorflow as tf
with tf.device("/job:tf_job/task:0"):
    A = tf.Variable(0, name="Var_A")
    o_add = tf.assign_add(A, 1, name="add1_A")

with tf.Session("grpc://localhost:2223") as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        input("Press Enter")
        print(sess.run([o_add]))
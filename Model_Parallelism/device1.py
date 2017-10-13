import tensorflow as tf
cluster = tf.train.ClusterSpec({"tf_job": ["localhost:2222", "localhost:2223", "localhost:2224"]})
server = tf.train.Server(cluster, job_name="tf_job", task_index=0)

with tf.device("/job:tf_job/task:0"):
    x_place_holder = tf.placeholder(dtype=tf.float32, shape=[])
    Op1 = tf.add(x_place_holder, 1, name="Op1")

with tf.device("/job:tf_job/task:1"):
    Op2 = tf.add(Op1, 2, name="Op2")

with tf.device("/job:tf_job/task:2"):
    Op3 = tf.add(Op1, 3, name="Op4")

with tf.device("/job:tf_job/task:0"):
    Output = tf.add(Op2, Op3, name="Output")

print("Server.target is",server.target)

with tf.Session(server.target) as sess:
    for i in range(5):
        data_in =float(input("Enter input"))
        print(sess.run(Output, feed_dict={x_place_holder: data_in}))


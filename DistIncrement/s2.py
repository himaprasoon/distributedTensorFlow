import tensorflow as tf
cluster = tf.train.ClusterSpec({"tf_job": ["localhost:2222","localhost:2223"]})
server = tf.train.Server(cluster, job_name="tf_job", task_index=1)


with tf.device("/job:tf_job/task:0"):
    A = tf.Variable(0, name="Var_A")
    o_add = tf.assign_add(A, 1, name="add1_A")

with tf.device("/job:tf_job/task:1"):
    B = tf.Variable(100,name="Var_B")
    t_add = tf.assign_add(B, 1, name="add1_B")

logs_path = "tf_job_log"
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

with tf.Session(server.target,config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        input("Press Enter")
        print(sess.run([t_add,o_add]))
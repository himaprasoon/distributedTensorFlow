import tensorflow as tf
cluster = tf.train.ClusterSpec({"local": ["localhost:2222"], "ml":["localhost:2223"]})
server = tf.train.Server(cluster, job_name="ml", task_index=0)
print("/job:local/task:%d" % 0)

with tf.device("/job:local/task:0"):
    weights_1 = tf.Variable(1, name="hima1")
    weights_t = tf.Variable(1, name="himat")
    o_add = tf.assign(weights_1, tf.add(weights_1, weights_t, name="add1"))

with tf.device("/job:ml/task:0"):
    weights_2 = tf.Variable(2, name="hima2")
    t_add = tf.assign(weights_2, tf.add(weights_2, 10, name="add2"))
logs_path="ml"
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
with tf.Session("grpc://localhost:2222", config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        input("haha")
        print(sess.run([t_add,o_add]))
# cluster = tf.train.ClusterSpec({ "ps":["localhost:2222"],"worker": ["localhost:2223","localhost:2224"]})
# print(len(cluster.as_dict()["worker"]))


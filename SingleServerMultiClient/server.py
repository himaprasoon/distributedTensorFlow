import tensorflow as tf
cluster = tf.train.ClusterSpec({"tf_job": ["localhost:2223"]})
server = tf.train.Server(cluster, job_name="tf_job", task_index=0)
print(server.target)
server.join()
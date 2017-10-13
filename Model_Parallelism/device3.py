import tensorflow as tf
cluster = tf.train.ClusterSpec({"tf_job": ["localhost:2222", "localhost:2223", "localhost:2224"]})
server = tf.train.Server(cluster, job_name="tf_job", task_index=2)
server.join()


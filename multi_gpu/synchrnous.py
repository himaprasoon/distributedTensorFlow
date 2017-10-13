import tensorflow as tf
import argparse
import sys
import numpy as np
n = 100
learning_rate = 0.001
batch_size = 100
display_step = 10
epochs = 3000

h,l,w =224, 224, 3
def get_data():
    train_data = np.random.rand(n, h*l*w)
    train_labels = np.random.rand(n, 1)
    return train_data,train_labels

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x):
    weights = {}
    biases = {}

    with tf.name_scope(name="ffn2"):
        weights['wd2'] = tf.Variable(tf.random_uniform([h*l*w, 10]))
        biases['bd2'] = tf.Variable(tf.random_uniform([10]))
        fc2 = tf.nn.tanh(tf.add(tf.matmul(x, weights['wd2']), biases['bd2']))

    with tf.name_scope(name="output"):
        weights['out'] = tf.Variable(tf.random_uniform([10, 1]))
        biases['out'] = tf.Variable(tf.random_uniform([1]))
        out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


def main(_):

    cluster = tf.train.ClusterSpec({ "ps":["localhost:2222"], "worker": ["localhost:2223", "localhost:2224"]})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":
        print(FLAGS.task_index, "task index")

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d/device:gpu:%d" % (FLAGS.task_index,FLAGS.task_index),
                cluster=cluster)):

            x = tf.placeholder(tf.float32, [None, 150528])
            y = tf.placeholder(tf.float32, [None, 1])
            logits = conv_net(x)

            pred = tf.nn.sigmoid(logits)

            # Define loss and optimizer
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.contrib.framework.get_or_create_global_step()

        opt = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=2,
                                           total_num_replicas = 2)
        train_op = opt.minimize(cost,global_step=global_step)

        logs_path = FLAGS.job_name+str(FLAGS.task_index)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        sync_replicas_hook = opt.make_session_run_hook(FLAGS.task_index == 0)
        # Read data
        train_data, train_labels = get_data()
        train_data_minibatches = [train_data[k:k + batch_size] for k in range(0, len(train_data), batch_size)]
        train_label_minibatches = [train_labels[k:k + batch_size] for k in range(0, len(train_labels), batch_size)]
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),hooks=[sync_replicas_hook],
                                               checkpoint_dir=None,
                                               ) as mon_sess:
            counter = epochs
            while not mon_sess.should_stop() and counter > 0 :
                counter -= 1
                print("Epoch :: ", epochs-counter)
                for batch_x, batch_y in zip(train_data_minibatches, train_label_minibatches):
                    mon_sess.run(cost, feed_dict={x: batch_x, y: batch_y})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--job_name",
        type=str,
        default="worker",
        help="One of 'ps', 'worker'"
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print("\nParameters:")
    print(FLAGS.job_name)
    tf.app.run(main=main, argv=[sys.argv[0]])
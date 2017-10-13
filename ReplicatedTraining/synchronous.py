import tensorflow as tf
import argparse
import sys


def read_data(worker_task_index):
    if worker_task_index == 0:
        return 2.0,20.0
    return 2.0,6.0

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
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            x_place_holder = tf.placeholder(dtype=tf.float32, shape=[])
            y_place_holder = tf.placeholder(dtype=tf.float32, shape=[])

            m = tf.Variable(10.0)

            Y = tf.multiply(m, x_place_holder)
            cost = tf.abs(Y - y_place_holder)
            global_step = tf.contrib.framework.get_or_create_global_step()

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

        opt = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=2,
                                           total_num_replicas = 2)
        train_op = opt.minimize(cost,global_step=global_step)

        logs_path = FLAGS.job_name+str(FLAGS.task_index)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        sync_replicas_hook = opt.make_session_run_hook(FLAGS.task_index == 0)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),hooks=[sync_replicas_hook],
                                               checkpoint_dir=None,
                                               ) as mon_sess:
            counter = 9000
            while not mon_sess.should_stop() and counter > 0 :
                counter -= 1
                input("Inside mon_Sess.should_stop. Press enter")
                data = read_data(FLAGS.task_index)
                print(mon_sess.run([train_op],feed_dict={x_place_holder: data[0], y_place_holder:data[1]}))
                print(mon_sess.run([m,global_step]))
                # print(mon_sess.run[])


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
    FLAGS, unparsed = parser.parse_known_args()
    print("\nParameters:")
    print(FLAGS.job_name)
    tf.app.run(main=main, argv=[sys.argv[0]])
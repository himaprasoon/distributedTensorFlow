import tensorflow as tf
import argparse
import sys
import time
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.framework import ops


class MyCustomHook(session_run_hook.SessionRunHook):
    def __init__(self, server_name, graph_node_name):
        self.server_name = server_name
        self.graph_node_name = graph_node_name

    def begin(self):
        print("begin in HimasHook", self.server_name)
        self._global_step_tensor = training_util.get_global_step()
        self.my_var = ops.get_default_graph().get_collection(self.graph_node_name)
        print(len(self.my_var))
        if len(self.my_var) == 1:
            self.my_var = self.my_var[0]
        print("my_var",self.my_var)
        print(ops.get_default_graph())

    def before_run(self, run_context):
        print("Begin_run in MyCustomHook", self.server_name, run_context)
        return SessionRunArgs(self.my_var)

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        print("After run in MyCustomHook", self.server_name, global_step, run_values, run_context)

def create_start_queue(i,no_of_workers):
    """Queue used to signal death for i'th ps shard. Intended to have
    all workers enqueue an item onto it to signal doneness."""
    with tf.device("job:worker/task:" + str(i)):
        queue = tf.FIFOQueue(no_of_workers, tf.int32, shared_name="start_queue" +
                                                      str(i))
        enq = queue.enqueue(1)
        deq = queue.dequeue()
        return {"enque":enq , "deque":deq}


def create_start_queues(num_of_workers):
    return [create_start_queue(i,num_of_workers) for i in range(num_of_workers)]

def main(_):
    cluster = tf.train.ClusterSpec({ "ps":["localhost:2222"], "worker": ["localhost:2223", "localhost:2224"]})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        # time.sleep(10)
        queue =  tf.FIFOQueue(2, tf.int32, shared_name="done_queue" +
                                                             str(FLAGS.task_index))
        deq = queue.dequeue()
        print("queue.name",queue.name)

        with tf.Session(server.target) as sess:
            print("Inside session")
            for i in range(2):
                sess.run(deq)
                print("ps %d received done %d" % (FLAGS.task_index, i))
            print("ps %d: quitting" % FLAGS.task_index)
        print("Both Workers Finished execution")
    elif FLAGS.job_name == "worker":
        print(FLAGS.task_index, "task index")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            a = tf.Variable(20, dtype=tf.float32,name="ahah")
            b = tf.Variable(100.0, dtype=tf.float32,name="local")
            c = tf.assign(b, tf.add(2.0, b))
            global_step = tf.contrib.framework.get_or_create_global_step()
            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                a, global_step=global_step)
            # train_op = tf.train.AdagradOptimizer(0.01).minimize(
            #     loss, global_step=global_step)
        with tf.device("job:ps/task:" + str(0)):
            print("Creating graph")
            queue = tf.FIFOQueue(2, tf.int32, shared_name="done_queue" +
                                                          str(0))
            print("queue.name", queue.name)
            # input("jja")
            enq = queue.enqueue(1)
            deq = queue.dequeue()

        qs = create_start_queues(2)
        this_queue  = qs[FLAGS.task_index]["enque"]


        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_epochs)]

        logs_path = FLAGS.job_name+str(FLAGS.task_index)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir=None,
                                               hooks=hooks) as mon_sess:
            counter = 9000
            print("enquing curent worker queue")
            for i in range(2):
                # enque current worker queue
                mon_sess.run(this_queue)
            print("Dequing ")
            mon_sess.run([q["deque"] for q in qs])
            while not mon_sess.should_stop() and counter >0:
                counter -= 1
                print("Inside mon_sees.should stop")
                print(mon_sess.run([train_op, a, global_step, b]))
            mon_sess.run(enq)


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
        "--num_epochs",
        type=int,
        default=50000,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print("\nParameters:")
    print(FLAGS.job_name)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
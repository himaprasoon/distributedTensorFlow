import tensorflow as tf
import os
import ast

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


def create_done_queue(i,no_of_workers):
    """Queue used to signal death for i'th ps shard. Intended to have
    all workers enqueue an item onto it to signal doneness."""
    with tf.device("job:ps/task:" + str(i)):
        queue = tf.FIFOQueue(no_of_workers, tf.int32, shared_name="done_queue" +
                                                      str(i))
        enq = queue.enqueue(1)
        deq = queue.dequeue()
        return {"enque":enq , "deque":deq}


def create_done_queues(num_of_ps,num_of_workers):
    return [create_done_queue(i,num_of_workers) for i in range(num_of_ps)]


def main(_):

    POD_NAME = os.environ.get('POD_NAME')
    job_name, task_id, _ = POD_NAME.split('-', 2)
    task_id = int(task_id)

    CLUSTER_CONFIG = os.environ.get('CLUSTER_CONFIG')
    cluster_def = ast.literal_eval(CLUSTER_CONFIG)
    cluster = tf.train.ClusterSpec(cluster_def)
    no_of_ps = len(cluster_def["ps"])
    no_of_workers = len(cluster_def["worker"])
    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=task_id)
    if job_name == "ps":
        deq = create_done_queue(task_id,no_of_workers)["deque"]

        with tf.Session(server.target) as sess:
            for i in range(no_of_workers):
                sess.run(deq)
                print("ps %d received done from worker %d" % (task_id, i))
            print("ps %d: quitting" % (task_id))
        print("All Workers Finished execution")
    elif job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_id,
                cluster=cluster)):

            a = tf.Variable(20, dtype=tf.float32,name="ahah")
            b = tf.Variable(100.0, dtype=tf.float32,name="local")
            c = tf.assign(b, tf.add(2.0, b))
            global_step = tf.contrib.framework.get_or_create_global_step()
            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                a, global_step=global_step)

        enques = [enq["enque"] for enq in create_done_queues(num_of_ps=no_of_ps,num_of_workers=no_of_workers)]
        # The StopAtStepHook handles stopping after running given steps.
        # HimasHook(FLAGS.job_name + str(FLAGS.task_index), graph_node_name="local")
        # hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_epochs)]
        hooks = None
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        start_queues = create_start_queues(2)
        this_queue = start_queues[task_id]["enque"]

        logs_path = POD_NAME+str(task_id)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_id == 0),
                                               checkpoint_dir=None,
                                               hooks=hooks) as mon_sess:
            counter = 9000
            print("enquing curent worker queue")
            for i in range(no_of_workers):
                # enque current worker queue
                mon_sess.run(this_queue)
            print("Dequing ")
            mon_sess.run([q["deque"] for q in start_queues])
            while not mon_sess.should_stop() and counter >0:
                counter -= 1
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                # input("waiting")
                print("Inside mon_sees.should stop")
                print(mon_sess.run([train_op, a, c, b]))
            mon_sess.run(enques)


if __name__ == "__main__":

    tf.app.run(main=main)
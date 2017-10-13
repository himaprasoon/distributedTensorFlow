from __future__ import print_function

import tensorflow as tf
import time

import os
import numpy as np

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 10
epochs = 10

# def get_data():
#     img_data = np.load('2000samples.npy')
#     random.shuffle(img_data)
#     train_data = [d[0:-1] for d in img_data]
#     train_labels = [[d[-1]] for d in img_data]
#
#     return train_data, train_labels
gpu_options = tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(gpu_options=gpu_options)
n = 10000
h,l,w =224, 224, 3
def get_data():
    train_data = np.random.rand(n, h*l*w)
    train_labels = np.random.rand(n, 1)
    return train_data,train_labels

x = tf.placeholder(tf.float32, [None, h*l*w])
y = tf.placeholder(tf.float32, [None, 1])

train_data, train_labels = get_data()
print(len(train_data))

save_path = os.getcwd() + '/alex_net_session/'
#
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# save_path = save_path + 'alex_net_session.sess'


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


# Construct model
with tf.device('/cpu:0'):
    logits = conv_net(x)

    pred = tf.nn.sigmoid(logits)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.greater(logits, 0.5), tf.greater(y, 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    train_data_minibatches = [train_data[k:k + batch_size] for k in range(0, len(train_data), batch_size)]
    train_label_minibatches = [train_labels[k:k + batch_size] for k in range(0, len(train_labels), batch_size)]

# saver = tf.train.Saver()
# tf.add_to_collection('pred', pred)
# tf.add_to_collection('x', x)
# tf.add_to_collection('y', y)

epoch_time_list = []
with tf.Session(config=config) as sess:
    sess.run(init)
    # tf.train.write_graph(tf.get_default_graph(), '/tmp/alexnet_puretf/', 'train.pb')
    # sess.run(optimizer, feed_dict={x: train_data_minibatches[0], y: train_label_minibatches[0]})
    for i in range(epochs):
        print("Epoch :: ", i)
        time_list = []
        for batch_x, batch_y in zip(train_data_minibatches, train_label_minibatches):
            start_time = int(round(time.time() * 1000))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            time_list.append(int(round(time.time() * 1000)) - start_time)
        epoch_time_list.append(np.asarray(time_list).sum())

    print("Optimization Finished!")
    # save_path = saver.save(sess=sess, save_path=save_path, write_meta_graph=True)

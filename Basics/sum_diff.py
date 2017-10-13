import tensorflow as tf
a = tf.constant(3.0, name="A")
b = tf.constant(4.0, name="B") # also tf.float32 implicitly
print("A", a)
print("B", b)

sum_ab = tf.add(a, b)

print("sum_ab", sum_ab)
# init_op = tf.global_weight_initializer()
sess = tf.Session() # Creates a Session : This connects the graph to an execution engine
print(sess.run([a, b, sum_ab]))

diff_ab = tf.subtract(a, b)
print(sess.run([diff_ab]))

sess.close() # Closes the session
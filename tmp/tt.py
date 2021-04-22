import tensorflow as tf

hello = tf.constant("dd")
sess = tf.compat.v1.Session()
print(sess.run(hello))
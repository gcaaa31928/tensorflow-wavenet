import tensorflow as tf

input_data = [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1]
input_samples = tf.placeholder(tf.complex64)
fft = tf.fft(input_samples)
sess = tf.Session()
sess.run(fft, feed_dict={input_samples: input_data})

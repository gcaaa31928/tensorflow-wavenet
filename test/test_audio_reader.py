from time import sleep

import tensorflow as tf
from sklearn.metrics import mean_squared_error
from wavenet import AudioReader

DATA_DIR = '../input'
DATA_OUT_DIR = '../output'
SAMPLE_RATE = 16000
SAMPLE_SIZE = 5000
SILENCE_THRESHOLD = 0.3


class TestAudioReader(tf.test.TestCase):

    def setUp(self):
        self.coord = tf.train.Coordinator()
        self.reader = AudioReader(
            DATA_DIR,
            DATA_OUT_DIR,
            self.coord,
            sample_rate=SAMPLE_RATE,
            sample_size=SAMPLE_SIZE,
            silence_threshold=SILENCE_THRESHOLD)

    def testAudioThread(self):
        max_allowed_mse = 1.0
        with self.test_session() as sess:
            threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)
            self.reader.start_threads(sess)
            input_batch = self.reader.dequeue(1)
            input_audio, output_audio = sess.run(input_batch)

            print(input_audio)
            test = self.reader.dequeue(1)

            print(sess.run(test)[0])
            test = self.reader.dequeue(1)
            print(sess.run(test)[0])
            test = self.reader.dequeue(1)
            print(sess.run(test)[0])
            test = self.reader.dequeue(1)
            print(sess.run(test)[0])
            test = self.reader.dequeue(1)
            print(sess.run(test)[0])
            mse = mean_squared_error(input_audio.flatten(), output_audio.flatten())
            self.assertLess(mse, max_allowed_mse)
            self.coord.request_stop()



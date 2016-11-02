import tensorflow as tf
import numpy as np

from wavenet import WaveNetModel
from wavenet import mu_law_encode

np.set_printoptions(threshold=np.nan)
SAMPLE_RATE_HZ = 2000.0  # Hz
SAMPLE_DURATION = 0.5  # Seconds
F1 = 155.56  # E-flat frequency in hz
F2 = 196.00  # G frequency in hz
F3 = 233.08  # B-flat frequency in hz
F4 = 1760.00  # D#4/Eb4
F5 = 2093.00  # F#4/Gb4
F6 = 2489.02  # A4


def make_sine_waves():
    """Creates a time-series of audio amplitudes corresponding to 3
    superimposed sine waves."""
    sample_period = 1.0 / SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)

    amplitudes = (np.sin(times * 2.0 * np.pi * F1) / 3.0 +
                  np.sin(times * 2.0 * np.pi * F2) / 3.0 +
                  np.sin(times * 2.0 * np.pi * F3) / 3.0)

    output_amplitudes = (np.sin(times * 2.0 * np.pi * F4) / 3.0 +
                         np.sin(times * 2.0 * np.pi * F5) / 3.0 +
                         np.sin(times * 2.0 * np.pi * F6) / 3.0)

    return amplitudes, output_amplitudes


class TestOwn(tf.test.TestCase):
    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                skip_channels=32)

    def test1(self):
        audio, output_audio = make_sine_waves()
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        output_audio_tensor = tf.convert_to_tensor(output_audio, dtype=tf.float32)

        input_batch = mu_law_encode(audio_tensor, 256)
        output_batch = mu_law_encode(output_audio_tensor, 256)
        encoded = self.net._one_hot(input_batch)
        output_encoded = self.net._one_hot(output_batch)
        shifted = tf.slice(output_encoded, [0, 1, 0],
                           [-1, tf.shape(output_encoded)[1] - 1, -1])
        # shifted = tf.pad(shifted, [[0, 0], [0, 1], [0, 0]])
        raw_output = self.net._create_network(encoded)
        out = tf.reshape(raw_output, [-1, self.net.quantization_channels])
        # Cast to float64 to avoid bug in TensorFlow
        proba = tf.cast(
            tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
        last = tf.slice(
            proba,
            [tf.shape(proba)[0] - 1, 0],
            [1, self.net.quantization_channels])
        lasted = tf.reshape(last, [-1])
        # shifted = tf.pad(shifted, [[0, 0], [0, 1], [0, 0]])
        # slice = tf.reshape(shifted, [-1, self.net.quantization_channels])
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            print(sess.run(out).shape)
            print(sess.run(proba)[1])
            print(sess.run(proba)[0])
            print(sess.run(last).shape)
            print(sess.run(lasted).shape)

    def test2(self):
        matrix = [[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]]
        shaped = tf.shape(matrix)
        sliced = tf.slice(matrix, [0, 1, 0], [-1, tf.shape(matrix)[1] - 1, -1])
        shifted = tf.pad(sliced, [[0, 0], [0, 1], [0, 0]])
        with self.test_session() as sess:
            print(sess.run(shaped))
            print(sess.run(shifted))


if __name__ == '__main__':
    tf.test.main()

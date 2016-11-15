import tensorflow as tf
import numpy as np
from generate import create_seed
from wavenet import (WaveNetModel, time_to_batch, batch_to_time, causal_conv,
                     optimizer_factory, mu_law_decode)
from wavenet.model import create_variable
import librosa


SAMPLE_RATE_HZ = 2000.0  # Hz
TRAIN_ITERATIONS = 6000
SAMPLE_DURATION = 0.5  # Seconds
SAMPLE_PERIOD_SECS = 1.0 / SAMPLE_RATE_HZ
MOMENTUM = 0.95
MOMENTUM_SCALAR_INPUT = 0.9
GENERATE_SAMPLES = 1000
QUANTIZATION_CHANNELS = 256
WINDOW_SIZE = 1000
F1 = 155.56  # E-flat frequency in hz
F2 = 196.00  # G frequency in hz
F3 = 233.08  # B-flat frequency in hz
F4 = 500.00  # D#4/Eb4
F5 = 600.00  # F#4/Gb4
F6 = 700.02  # A4


class TestScalarInput(tf.test.TestCase):

    def generate_waveform(self, sess):
        samples = tf.placeholder(tf.int32)
        next_sample_probs = self.net.predict_proba_all(samples)
        operations = [next_sample_probs]

        waveform = []
        seed = create_seed("sine_train.wav",
                           SAMPLE_RATE_HZ,
                           QUANTIZATION_CHANNELS,
                           window_size=WINDOW_SIZE,
                           silence_threshold=0)
        input_waveform = sess.run(seed).tolist()
        decode = mu_law_decode(samples, QUANTIZATION_CHANNELS)
        slide_windows = 256
        for slide_start in range(0, len(input_waveform), slide_windows):
            if slide_start + slide_windows >= len(input_waveform):
                break
            input_audio_window = input_waveform[slide_start:slide_start + slide_windows]

            # Run the WaveNet to predict the next sample.
            all_prediction = sess.run(operations, feed_dict={samples: input_audio_window})[0]
            all_prediction = np.asarray(all_prediction)
            output_waveform = get_all_output_from_predictions(all_prediction)
            print("Prediction {}".format(output_waveform))
            waveform.extend(output_waveform)

        waveform = np.array(waveform[:])
        decoded_waveform = sess.run(decode, feed_dict={samples: waveform})
        return decoded_waveform

    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                use_biases=True,
                                skip_channels=32)
        self.optimizer_type = 'sgd'
        self.learning_rate = 0.02
        self.generate = True
        self.momentum = MOMENTUM

    def testEndToEndTraining(self):
        audio, output_audio = make_sine_waves()
        np.random.seed(42)
        librosa.output.write_wav('sine_train.wav', audio, int(SAMPLE_RATE_HZ))
        librosa.output.write_wav('sine_expected_answered.wav', output_audio, int(SAMPLE_RATE_HZ))

        input_samples = tf.placeholder(tf.float32)
        output_samples = tf.placeholder(tf.float32)

        loss = self.net.loss(input_samples, output_samples)
        optimizer = optimizer_factory[self.optimizer_type](
            learning_rate=self.learning_rate, momentum=self.momentum)
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss, var_list=trainable)
        init = tf.initialize_all_variables()

        generated_waveform = None
        max_allowed_loss = 0.1
        slide_windows = 256
        slide_start = 0
        with self.test_session() as sess:
            sess.run(init)
            for i in range(TRAIN_ITERATIONS):
                if slide_start + slide_windows >= min(len(audio), len(output_audio)):
                    slide_start = 0
                    print("slide from beginning...")
                input_audio_window = audio[slide_start:slide_start + slide_windows]
                output_audio_window = output_audio[slide_start:slide_start + slide_windows]
                slide_start += 1
                loss_val, _ = sess.run([loss, optim], feed_dict={input_samples: input_audio_window,
                                                                 output_samples: output_audio_window})
                if i % 10 == 0:
                    print("i: %d loss: %f" % (i, loss_val))
            # saver.save(sess, '/tmp/sine_test_model.ckpt', global_step=i)
            if self.generate:
                # Check non-incremental generation
                generated_waveform = self.generate_waveform(sess)
                check_waveform(self.assertGreater, generated_waveform)


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

def check_waveform(assertion, generated_waveform):
    librosa.output.write_wav('sine_test.wav',
                             generated_waveform,
                             int(SAMPLE_RATE_HZ))
    power_spectrum = np.abs(np.fft.fft(generated_waveform)) ** 2
    freqs = np.fft.fftfreq(generated_waveform.size, SAMPLE_PERIOD_SECS)
    indices = np.argsort(freqs)
    indices = [index for index in indices if freqs[index] >= 0 and
               freqs[index] <= 500.0]
    power_spectrum = power_spectrum[indices]
    freqs = freqs[indices]
    # plt.plot(freqs[indices], power_spectrum[indices])
    # plt.show()
    power_sum = np.sum(power_spectrum)
    f1_power = find_nearest(freqs, power_spectrum, F4)
    f2_power = find_nearest(freqs, power_spectrum, F5)
    f3_power = find_nearest(freqs, power_spectrum, F6)
    expected_power = f1_power + f2_power + f3_power
    # print("Power sum {}, F1 power:{}, F2 power:{}, F3 power:{}".
    #        format(power_sum, f1_power, f2_power, f3_power))

    # Expect most of the power to be at the 3 frequencies we trained
    # on.
    assertion(expected_power, 0.9 * power_sum)

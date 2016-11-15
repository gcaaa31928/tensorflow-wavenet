"""Unit tests for the WaveNet that check that it can train on audio data."""

import json
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
# import matplotlib.pyplot as plt
# import librosa
from tensorflow.python.client import timeline

from generate import create_seed
from wavenet import (WaveNetModel, time_to_batch, batch_to_time, causal_conv,
                     optimizer_factory, mu_law_decode)
from wavenet.model import create_variable

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


def get_all_output_from_predictions(predictions):
    samples = []
    for predit in predictions:
        sample = np.random.choice(
            np.arange(QUANTIZATION_CHANNELS), p=predit)
        samples.append(sample)
    return samples


def generate_waveform(sess, net, fast_generation, wav_seed=False):
    samples = tf.placeholder(tf.int32)
    if fast_generation:
        next_sample_probs = net.predict_proba_incremental(samples)
        sess.run(net.init_ops)
        operations = [next_sample_probs]
        operations.extend(net.push_ops)
    else:
        next_sample_probs = net.predict_proba(samples)
        operations = [next_sample_probs]

    waveform = [128]
    if wav_seed:
        seed = create_seed("sine_train.wav",
                           SAMPLE_RATE_HZ,
                           QUANTIZATION_CHANNELS,
                           window_size=WINDOW_SIZE,
                           silence_threshold=0)
        input_waveform = sess.run(seed).tolist()
    decode = mu_law_decode(samples, QUANTIZATION_CHANNELS)
    for i in range(GENERATE_SAMPLES):
        print("=====================================================")
        if fast_generation:
            window = waveform[-1]
            if wav_seed and i < len(input_waveform):
                window = input_waveform[i]
        else:
            if len(waveform) > 256:
                window = waveform[-256:]
            else:
                window = waveform
            if wav_seed:
                if i >= len(input_waveform):
                    break
                if i - 256 >= 0:
                    f_window = input_waveform[i - 256:i]

                else:
                    f_window = input_waveform[:i]
                    print("Input {}".format(f_window))
                    # print(window)
                if len(f_window) == 0:
                    continue
                    # print(window)

        # Run the WaveNet to predict the next sample.
        all_prediction = sess.run([net.predict_proba_all(samples)], feed_dict={samples: input_waveform})[0]
        all_prediction = np.asarray(all_prediction)
        output_waveform = get_all_output_from_predictions(all_prediction)
        print("Prediction {}".format(output_waveform))
        decoded_waveform = sess.run(decode, feed_dict={samples: output_waveform})
        return decoded_waveform
        # prediction = sess.run(operations, feed_dict={samples: f_window})[0]
        # sample = np.random.choice(
        #     np.arange(QUANTIZATION_CHANNELS), p=prediction)
        # waveform.append(sample)


        # print("Generated {} of {}: {}".format(i, GENERATE_SAMPLES, sample))
        # sys.stdout.flush()

    # Skip the first number of samples equal to the size of the receptive
    # field.
    waveform = np.array(waveform[:])
    decoded_waveform = sess.run(decode, feed_dict={samples: waveform})
    return decoded_waveform


def find_nearest(freqs, power_spectrum, frequency):
    # Return the power of the bin nearest to the target frequency.
    index = (np.abs(freqs - frequency)).argmin()
    return power_spectrum[index]


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


class TestSeed(tf.test.TestCase):
    def testVariableSeed(self):
        tensor = create_variable(
            'test',
            [2, 2, 2])
        init = tf.initialize_all_variables()
        with self.test_session() as sess:
            sess.run(init)
            matrix = sess.run(tensor)
            self.assertAlmostEqual(-0.45200047, matrix[0][0][0])
            self.assertAlmostEqual(0.72815341, matrix[0][0][1])


class TestMoveNet(tf.test.TestCase):

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


class TestNet(tf.test.TestCase):
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

    # Train a net on a short clip of 3 sine waves superimposed
    # (an e-flat chord).
    #
    # Presumably it can overfit to such a simple signal. This test serves
    # as a smoke test where we just check that it runs end-to-end during
    # training, and learns this waveform.

    def testEndToEndTraining(self):
        audio, output_audio = make_sine_waves()
        np.random.seed(42)
        librosa.output.write_wav('sine_train.wav', audio, int(SAMPLE_RATE_HZ))
        librosa.output.write_wav('sine_expected_answered.wav', output_audio, int(SAMPLE_RATE_HZ))
        # if self.generate:
        #
        #    power_spectrum = np.abs(np.fft.fft(audio))**2
        #    freqs = np.fft.fftfreq(audio.size, SAMPLE_PERIOD_SECS)
        #    indices = np.argsort(freqs)
        #    indices = [index for index in indices if freqs[index] >= 0 and
        #                                             freqs[index] <= 500.0]
        #    plt.plot(freqs[indices], power_spectrum[indices])
        #    plt.show()
        run_metadata = tf.RunMetadata()

        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        output_audio_tensor = tf.convert_to_tensor(output_audio, dtype=tf.float32)
        loss = self.net.loss(audio_tensor, output_audio_tensor)
        optimizer = optimizer_factory[self.optimizer_type](
            learning_rate=self.learning_rate, momentum=self.momentum)
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss, var_list=trainable)
        init = tf.initialize_all_variables()
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE)
        generated_waveform = None
        max_allowed_loss = 0.1
        loss_val = max_allowed_loss
        initial_loss = None
        with self.test_session() as sess:
            sess.run(init)
            initial_loss = sess.run(loss)
            for i in range(TRAIN_ITERATIONS):
                loss_val, _ = sess.run([loss, optim], run_metadata=run_metadata)
                if i % 10 == 0:
                    print("i: %d loss: %f" % (i, loss_val))
                    tl = timeline.Timeline(run_metadata.step_stats)
                    timeline_path = os.path.join('.', 'timeline.trace')
                    # with open(timeline_path, 'w') as f:
                    #     f.write(tl.generate_chrome_trace_format(show_memory=True))

            # Sanity check the initial loss was larger.
            # self.assertGreater(initial_loss, max_allowed_loss)

            # Loss after training should be small.
            # self.assertLess(loss_val, max_allowed_loss)

            # Loss should be at least two orders of magnitude better
            # than before training.
            # self.assertLess(loss_val / initial_loss, 0.01)

            # saver = tf.train.Saver(var_list=tf.trainable_variables())
            # saver.save(sess, '/tmp/sine_test_model.ckpt', global_step=i)
            if self.generate:
                # Check non-incremental generation
                generated_waveform = generate_waveform(sess, self.net, False, wav_seed=True)
                check_waveform(self.assertGreater, generated_waveform)

                # Check incremental generation
                # generated_waveform = generate_waveform(sess, self.net, True, wav_seed=True)
                # plt.plot(generated_waveform)
                # plt.show()
                # check_waveform(self.assertGreater, generated_waveform)


class TestNetWithBiases(TestNet):
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
        self.generate = False
        self.momentum = MOMENTUM


class TestNetWithRMSProp(TestNet):
    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                skip_channels=256)
        self.optimizer_type = 'rmsprop'
        self.learning_rate = 0.001
        self.generate = True
        self.momentum = MOMENTUM


class TestNetWithScalarInput(TestNet):
    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64,
                                           1, 2, 4, 8, 16, 32, 64],
                                filter_width=2,
                                residual_channels=32,
                                dilation_channels=32,
                                quantization_channels=256,
                                use_biases=True,
                                skip_channels=32,
                                scalar_input=True,
                                initial_filter_width=4)
        self.optimizer_type = 'rmsprop'
        self.learning_rate = 0.001
        self.generate = False
        self.momentum = MOMENTUM_SCALAR_INPUT


if __name__ == '__main__':
    tf.test.main()

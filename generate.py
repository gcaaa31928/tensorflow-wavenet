from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader

SAMPLES = 1000
TEMPERATURE = 1.0
LOGDIR = './logdir'
WINDOW = int(1e9)
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = 50
SILENCE_THRESHOLD = 0.1
STEP_LENGTH = 100
OUTPUT_FILE='./output/output.wav'

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError('Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
             'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,
        help='The number of past samples to take into '
             'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=False,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--step_length',
        type=int,
        default=STEP_LENGTH)
    return parser.parse_args()


def get_all_output_from_predictions(predictions, quantization_channels):
    samples = []
    for predit in predictions:
        sample = np.random.choice(
            np.arange(quantization_channels), p=predit)
        samples.append(sample)
    return samples


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size=WINDOW,
                silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    # audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]


def mse_with_output(waveform, filename, sample_rate):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    min_len = min(len(waveform), len(audio))
    waveform = waveform[:min_len]
    audio = audio[:min_len]
    error_array = np.square(np.subtract(waveform, audio))
    return np.average(error_array)


def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'])

    samples = tf.placeholder(tf.int32)

    next_sample = net.predict_proba_all(samples)

    # if args.fast_generation:
    #     sess.run(tf.initialize_all_variables())
    #     sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

    quantization_channels = wavenet_params['quantization_channels']
    seed = create_seed(args.wav_seed,
                       wavenet_params['sample_rate'],
                       quantization_channels)
    input_waveform = sess.run(seed).tolist()
    waveform = []
    print('waveform seed length from {}'.format(len(input_waveform)))
    print('samples {}'.format(args.samples))
    last_sample_timestamp = datetime.now()
    for slide_start in range(0, len(input_waveform), args.step_length):
        if slide_start + args.samples >= len(input_waveform):
            break
        input_audio_window = input_waveform[slide_start:slide_start + args.samples]

        outputs = [next_sample]
        # Run the WaveNet to predict the next sample.
        all_prediction = sess.run(outputs, feed_dict={samples: input_audio_window})[0]
        all_prediction = np.asarray(all_prediction)
        output_waveform = get_all_output_from_predictions(all_prediction, net.quantization_channels)

        if len(waveform) > 0:
            overlap_waveform = waveform[slide_start:len(waveform)]
            output_overlap_waveform = output_waveform[:-args.step_length]
            print(len(overlap_waveform), len(output_overlap_waveform), len(waveform))
            result = np.divide(np.add(output_overlap_waveform, overlap_waveform), 2.0)
            waveform[slide_start:len(waveform)] = result
            waveform.extend(output_waveform[-args.step_length:])

        else:
            waveform = output_waveform

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(slide_start + 1, args.samples),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

        # If we have partial writing, save the result so far.
        if (args.wav_out_path and args.save_every and
                        (slide_start + 1) % args.save_every == 0):
            out = sess.run(decode, feed_dict={samples: waveform})
            write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)
            print("current step is {}".format(slide_start))

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as an audio summary.
    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.train.SummaryWriter(logdir)
    tf.audio_summary('generated', decode, wavenet_params['sample_rate'])
    summaries = tf.merge_all_summaries()
    summary_out = sess.run(summaries,
                           feed_dict={samples: np.reshape(waveform, [-1, 1])})
    writer.add_summary(summary_out)

    # Save the result as a wav file.
    if args.wav_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})
        print("The error between expected and actual is {}".format(mse_with_output(out, OUTPUT_FILE, wavenet_params['sample_rate'])))
        write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()

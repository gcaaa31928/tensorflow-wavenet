import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename


def load_vctk_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the VCTK dataset, and
    additionally the ID of the corresponding speaker.'''
    files = find_files(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        matches = speaker_re.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]
        yield audio, speaker_id


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 audio_output_dir,
                 coord,
                 sample_rate,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=256):
        self.audio_dir = audio_dir
        self.audio_output_dir = audio_output_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.output_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'float32'],
                                         shapes=[(None, 1), (None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, self.output_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(audio_dir):
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        output_buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        iterator = load_generic_audio(self.audio_dir, self.sample_rate)
        output_iterator = load_generic_audio(self.audio_output_dir, self.sample_rate)
        while not stop:
            for (audio, filename), (output_audio, output_filename) in zip(iterator, output_iterator):
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)

                    output_audio = trim_silence(output_audio[:, 0], self.silence_threshold)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))

                print("audio length is {}".format(len(audio)))
                print("output audio length is {}".format(len(output_audio)))

                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, audio)
                    output_buffer_ = np.append(output_buffer_, output_audio)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        output_piece = np.reshape(output_buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece, self.output_placeholder: output_piece})
                        buffer_ = buffer_[1:]
                        output_buffer_ = output_buffer_[1:]
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})
        self.coord.request_stop()

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

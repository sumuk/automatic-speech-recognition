import os
import sys
import glob
import scipy.io.wavfile as wav
import numpy as np
from string import punctuation
from collections import OrderedDict
import re
import json
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops import array_ops
import math
import warnings
#warnings.simplefilter("error")
config_path=os.path.join(os.getcwd(),"config.json")
assert os.path.isfile(config_path),"Config file not found in {0}".format(config_path)

with open(config_path,"r") as file:
    config=json.loads(file.read())

num_features = config["num_features"]
num_filters = config["num_filters"]
num_hidden = config["num_hidden"]
num_layers = config["num_layers"]
batch_size = config["batch_size"]
num_classes = config["num_classes"]
dropout = config["dropout"]
initial_lr = config["lr"]
model_path=config["model_path"]
graph_path=config["graph_path"]
data_root=config["data_root"]
try:
    data_folder=config["data_folder"].split(',')
except:
    data_folder=config["data_folder"]


class dataloader():
    def __init__(self, batch, path):
        self.batch = batch
        self.path = path
        self.count = 0
        self.files = glob.iglob(os.path.join(self.path, "wav", "*.wav"))
        self.SPACE_TOKEN = '<space>'
        self.SPACE_INDEX = 0
        self.FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
        self.char_list = [str(i) for i in range(10)]
        self.char_list_c = [chr(32) if i == 0 else chr(i+96)
                            for i in range(27)]
        self.char_list_c.extend(self.char_list)
        self.char2int = OrderedDict((val, i)
                                    for i, val in enumerate(self.char_list_c))
        self.int2char = OrderedDict((i, val)
                                    for i, val in enumerate(self.char_list_c))

    def next(self):
        data = []
        length = []
        target = []
        original_text = []
        i = 0
        while i < self.batch:
            file = next(self.files)
            fs, audio = wav.read(file)
            print(file)
            if(audio.shape[0] > 0):
                
                filename = os.path.basename(file)
                file_name, ext = os.path.splitext(filename)
                try:
                    inputs = self.compute_linear_specgram(audio, fs)
                except ArithmeticError:
                    continue  
                
                i += 1
                data.append(inputs)
                length.append(inputs.shape[0])
                with open(os.path.join(self.path, "txt", file_name+".txt")) as f:
                    target_text = f.readline()
                    target_text = re.sub('[^a-zA-Z0-9]+', " ", target_text)
                    # target_text=re.sub("|".join(self.char_list),'',target_text)
                    original = ' '.join(target_text.strip().lower().split(' '))
                    # print("original:",original)
                    original_text.append(original)
                    targets = original.replace(' ', '  ')
                    targets = targets.split(' ')

                    # Adding blank label
                    targets = np.hstack(
                        [self.SPACE_TOKEN if x == '' else list(x) for x in targets])

                    # Transform char into index
                    targets = np.asarray(
                        [self.SPACE_INDEX if x == self.SPACE_TOKEN else self.char2int[x] for x in targets])

                    target.append(targets)
        max_len = np.max(length)
        for i in range(self.batch):
            if(data[i].shape[0] != max_len):

                data[i] = np.pad(
                    data[i], ((0, max_len-length[i]), (0, 0)), 'constant', constant_values=(0, 0))
                # length[i]=data[i].shape[0]

        data = np.asarray(data)
        train_inputs = (data - np.mean(data))/np.std(data)
        train_inputs = np.expand_dims(train_inputs, axis=3)
        train_targets = self.sparse_tuple_from(target)
        train_seq_len = length

        return train_inputs, train_seq_len, train_targets, original_text

        # return list_files,list_data
    def __iter__(self):
        return self

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(
            indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    def compute_linear_specgram(self, samples,
                                sample_rate,
                                stride_ms=10.0,
                                window_ms=20.0,
                                max_freq=None,
                                eps=1e-14):
        """Compute the linear spectrogram from FFT energy."""
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             "sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than "
                             "window size.")
        stride_size = int(0.001 * sample_rate * stride_ms)
        window_size = int(0.001 * sample_rate * window_ms)

        

        specgram, freqs = self._specgram_real(samples,
                                              window_size=window_size,
                                              stride_size=stride_size,
                                              sample_rate=sample_rate)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        spectrogram = np.log(specgram[:ind, :] + eps)

        spectrogram = spectrogram.transpose()

        # z-score normalizer
        spectrogram = spectrogram - np.mean(spectrogram)
        if(np.std(spectrogram)!=0):
            spectrogram = spectrogram / np.std(spectrogram)
        else:
            raise ArithmeticError
        
        return spectrogram

    def _specgram_real(self, samples, window_size, stride_size, sample_rate):
        """Compute the spectrogram for samples from a real signal."""
       
        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[:len(samples) - truncate_size]
        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
        nstrides = (samples.strides[0], samples.strides[0] * stride_size)
        windows = np.lib.stride_tricks.as_strided(
            samples, shape=nshape, strides=nstrides)
        assert np.all(windows[:, 1] ==
                      samples[stride_size:(stride_size + window_size)])
        
        weighting = np.hanning(window_size)[:, None]
        fft = np.fft.rfft(windows * weighting, axis=0)
        fft = np.absolute(fft)
        fft = fft**2
        scale = np.sum(weighting**2) * sample_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[(0, -1), :] /= scale
        
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
        return fft, freqs


def get_rnn_seqlen(seq_lens):
    seq_lens = tf.cast(seq_lens, tf.float64)
    rnn_seq_lens = tf.div(tf.subtract(seq_lens, 10), 2.0)
    rnn_seq_lens = tf.ceil(rnn_seq_lens)
    rnn_seq_lens = tf.div(tf.subtract(rnn_seq_lens, 10), 1.0)
    rnn_seq_lens = tf.ceil(rnn_seq_lens)
    rnn_seq_lens = tf.cast(rnn_seq_lens, tf.int32)

    
    return rnn_seq_lens


def dynamic_lr(initial_lr):
    return initial_lr*0.999


def uni_rnn(inputs, seqLengths, time_major=True):
    hidden = inputs
    for i in range(num_layers):
        scope = 'DRNN_' + str(i + 1)
        cell = tf.contrib.rnn.GRUCell(num_hidden, activation=tf.tanh)
        outputs, output_states = tf.nn.dynamic_rnn(
            cell=cell, inputs=hidden, sequence_length=seqLengths, dtype=tf.float32, time_major=time_major, scope=scope)
        hidden = tf.contrib.layers.dropout(
            outputs, keep_prob=1-dropout, is_training=True)
        if i != num_layers-1:
            pass
        else:
            return tf.reshape(hidden, [-1, num_hidden])


def model(epochs):
    inputs = tf.placeholder(tf.float32, [None, None, num_features, 1])
    seq_len = tf.placeholder(tf.int32, [None])
    learning_rate = tf.placeholder(tf.float32, shape=[])
    targets = tf.sparse_placeholder(tf.int32)
    conved_seq_lens = get_rnn_seqlen(seq_len)
    kernel = tf.get_variable(
        "conv1", [11, num_features, 1, num_filters], initializer=None, dtype=tf.float32)
    conv1 = tf.nn.conv2d(inputs, kernel, [1, 2, 1, 1], padding='VALID')
    kernel1 = tf.get_variable(
        "conv2", [11, 1, num_filters, 2*num_filters], initializer=None, dtype=tf.float32)
    conv2 = tf.nn.conv2d(conv1, kernel1, [1, 1, 1, 1], padding='VALID')
    fdim = conv2.get_shape().dims
    feat_dim = fdim[2].value * fdim[3].value
    rnn_input = tf.reshape(conv2, [batch_size, -1, feat_dim])
    rnn_input = tf.transpose(rnn_input, (1, 0, 2))
    rnn_outputs = uni_rnn(rnn_input, conved_seq_lens)
    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0.0, shape=[num_classes]))
    logits = tf.matmul(rnn_outputs, W)+b
    logits = tf.reshape(logits, [-1, batch_size, num_classes])
    loss = tf.nn.ctc_loss(targets, logits, conved_seq_lens,
                          time_major=True, ignore_longer_outputs_than_inputs=True)
    cost = tf.reduce_mean(loss)
    first_summary = tf.summary.scalar(name='cost_summary', tensor=cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_value(grad, -400., 400.), var)
                  for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, conved_seq_lens)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    lr = initial_lr
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if(os.path.isfile(os.path.join(model_path,"model.ckpt.index"))):
            saver.restore(
                sess, os.path.join(model_path,"model.ckpt"))
        else:
            sess.run(tf.global_variables_initializer())
            print("in am here in global")
        writer = tf.summary.FileWriter(graph_path, sess.graph)
        kl_ep = 0
        try:
            for epoch in range(epochs):
                ct = 0.0
                for folder_name in data_folder:
                    print(os.path.join(data_root,folder_name))
                    a = dataloader(
                        batch_size,os.path.join(data_root,folder_name))

                    counter=0
                    for train_inputs, train_seq_len, train_targets, original_text in a:
                        if(kl_ep != 0 and kl_ep % 2000 == 0):
                            lr = dynamic_lr(lr)
                        c, _, f_c = sess.run([cost, train_op, first_summary], feed_dict={
                            inputs: train_inputs, seq_len: train_seq_len, targets: train_targets, learning_rate: lr})
                        
                        kl_ep += 1
                        print(counter,c)
                        if(not (math.isinf(c) and c > 0)):
                            writer.add_summary(f_c, kl_ep)
                            ct += c
                            counter+=1
                    else:
                        print("{0}/{1}epoch loss after completing {3} {2}".format(epoch,epochs,str(ct/(counter+1)),folder_name))
                
                saver.save(sess,os.path.join( model_path,"model.ckpt"))
        except KeyboardInterrupt:
                print("need to save model file details")
                saver.save(sess,os.path.join( model_path,"model.ckpt"))
                config["lr"]=lr
                print(lr)
                with open(config_path,"w") as file:
                    json.dump(config,file)

model(10)

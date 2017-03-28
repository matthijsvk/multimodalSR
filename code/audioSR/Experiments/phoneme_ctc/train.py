import os
import time
import logging
logging.basicConfig(level=logging.INFO)
import signal
import subprocess

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops

from phoneme_set import phoneme_set_39
from utils import process_wav


num_features = 39 # 13 mfcc + 26 logfbank
num_classes = 40 # 39 phonemes + blank

num_layers = 3 # lstm cells stack together
num_hidden = 128 # lstm state

num_epochs = 120 
batch_size = 16

learning_rate = 0.001
momentum = 0.9

def train_model(ENV, train_data=None, test_data=None, decode=False, file_decode=False):
    graph = tf.Graph()
    with graph.as_default():
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features])

        targets_idx = tf.placeholder(tf.int64)
        targets_val = tf.placeholder(tf.int32)
        targets_shape = tf.placeholder(tf.int64)
        targets = tf.SparseTensor(targets_idx, targets_val, targets_shape)
        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None])

        # Weights & biases
        weight_classes = tf.Variable(tf.truncated_normal([num_hidden, num_classes],
                                                         mean=0, stddev=0.1,
                                                         dtype=tf.float32))
        bias_classes = tf.Variable(tf.zeros([num_classes]), dtype=tf.float32)

        # Network
        forward_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True)
        backward_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True)

        stack_forward_cell = tf.nn.rnn_cell.MultiRNNCell([forward_cell] * num_layers,
                                                         state_is_tuple=True)
        stack_backward_cell = tf.nn.rnn_cell.MultiRNNCell([backward_cell] * num_layers,
                                                          state_is_tuple=True)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(stack_forward_cell, 
                                                     stack_backward_cell,
                                                     inputs,
                                                     sequence_length=seq_len,
                                                     time_major=False, # [batch_size, max_time, num_hidden]
                                                     dtype=tf.float32)
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]

        """
        outputs_concate = tf.concat_v2(outputs, 2)
        outputs_concate = tf.reshape(outputs_concate, [-1, 2*num_hidden])
        # logits = tf.matmul(outputs_concate, weight_classes) + bias_classes
        """
        fw_output = tf.reshape(outputs[0], [-1, num_hidden])
        bw_output = tf.reshape(outputs[1], [-1, num_hidden])
        logits = tf.add(tf.add(tf.matmul(fw_output, weight_classes), tf.matmul(bw_output, weight_classes)), bias_classes)

        logits = tf.reshape(logits, [batch_size, -1, num_classes])
        loss = tf.reduce_mean(ctc_ops.ctc_loss(logits, targets, seq_len, time_major=False))
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

        # Evaluating
        # decoded, log_prob = ctc_ops.ctc_greedy_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_len)
        decoded, log_prob = ctc_ops.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_len)
        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
        if not decode:
            ckpt = tf.train.get_checkpoint_state(ENV.output)
            if ckpt:
                print('load', ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)

            total_train_data = len(train_data)
            total_test_data = len(test_data)
            num_batch = total_train_data
            for curr_epoch in range(num_epochs):
                start = time.time()
                train_cost = 0
                train_ler = 0
                for i in range(num_batch-1):
                    feed = {
                        inputs: train_data[i][0],
                        targets_idx: train_data[i][1][0],
                        targets_val: train_data[i][1][1],
                        targets_shape: train_data[i][1][2],
                        seq_len: train_data[i][2]
                    }
                    batch_cost, _ = session.run([loss, optimizer], feed)
                    train_cost += batch_cost*batch_size
                    train_ler += session.run(label_error_rate, feed_dict=feed)*batch_size
                    log = "Epoch {}/{}, iter {}, batch_cost {}"
                    logging.info(log.format(curr_epoch+1, num_epochs, i, batch_cost))

                train_cost /= num_batch
                train_ler /= num_batch
                saver.save(session, os.path.join(ENV.output, 'best.ckpt'), global_step=curr_epoch)

                feed_test = {
                    inputs: test_data[0][0],
                    targets_idx: test_data[0][1][0],
                    targets_val: test_data[0][1][1],
                    targets_shape: train_data[0][1][2],
                    seq_len: test_data[0][2]
                }
                test_cost, test_ler = session.run([loss, label_error_rate], feed_dict=feed_test)
                log = "Epoch {}/{}, test_cost {}, test_ler {}"
                logging.info(log.format(curr_epoch+1, num_epochs, test_cost, test_ler))
        else:
            ckpt = tf.train.get_checkpoint_state(ENV.model_path)
            print('load', ckpt.model_checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(session, ckpt.model_checkpoint_path)

            while True:
                if file_decode:
                    wav_file = raw_input('Enter the wav file path:')
                else:
                    wav_file = 'temp.wav'
                    raw_input('Press Enter to start...')
                    try:
                        sox = subprocess.Popen(['sox', '-d', '-b', '16', '-c', '1', '-r', '16000', 'temp.wav'])
                        sox.communicate()
                    except KeyboardInterrupt:
                        os.kill(sox.pid, signal.SIGTERM)
                        if sox.poll() is None:
                            time.sleep(2)
                    print('Done recording')
                features = process_wav(wav_file)
                batch_features = np.array([features for i in range(16)])
                batch_seq_len = np.array([features.shape[0] for i in range(16)])
                print(batch_features.shape)
                feed = {
                    inputs: batch_features,
                    seq_len: batch_seq_len
                }
                d, oc = session.run([decoded[0], outputs], feed_dict=feed)
                dsp = d.shape
                res = []
                for label in d.values[:dsp[1]]:
                    for k, v in phoneme_set_39.items():
                        if v == label + 1:
                            res.append(k)           
                print(res)

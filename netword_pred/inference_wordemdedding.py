import tensorflow as tf
from tensorflow.contrib import rnn,seq2seq
import numpy as np
from assemble_config import *
# from assemble_utils import *

def get_lstm_layer(rnn_size,layer_sise):
    lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layers_size)]
    return tf.contrib.rnn.MultiRNNCell(lstm_layers)

def network(inputs,seq_lens,max_seq_len,rnn_size,layers_size,keep_prob_ph):
    with tf.variable_scope('embedding'):
        encoder_embedding = tf.Variable(tf.truncated_normal(shape=[encoder_vocab_size,embedding_dim],stddev=1),name='encoder_embedding')
        input_x_embedded = tf.nn.embedding_lookup(encoder_embedding,inputs)

    with tf.variable_scope('encoder_embedding'):
        fw = get_lstm_layer(rnn_size,layers_size)
        bw = get_lstm_layer(rnn_size,layers_size)
        fw_cell_zero = fw.zero_state(batch_size, tf.float32)
        bw_cell_zero = bw.zero_state(batch_size, tf.float32)
        enc_out,en_state = tf.nn.bidirectional_dynamic_rnn(fw,bw,input_x_embedded,sequence_length=seq_lens,
                                                           initial_state_fw=fw_cell_zero,initial_state_bw=bw_cell_zero)
        enc_outs = tf.concat(enc_out,2)


    with tf.variable_scope('fc_embedding'):
        flatten = tf.reshape(enc_outs,[-1, rnn_size*2])
        drop_flatten = tf.nn.dropout(flatten, keep_prob_ph)
        W = tf.Variable(tf.truncated_normal([rnn_size*2, 128], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
        b = tf.Variable(tf.constant(0., shape=[128]))
        out_fc = tf.matmul(drop_flatten, W)+b
        out_fc = tf.nn.relu(out_fc)

    with tf.variable_scope('fc1_embedding'):
        drop_flatten = tf.nn.dropout(out_fc, keep_prob_ph)
        W = tf.Variable(tf.truncated_normal([128, num_class], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
        b = tf.Variable(tf.constant(0., shape=[num_class]))
        out_fc1 = tf.matmul(drop_flatten, W)+b
        out = tf.reshape(out_fc1,[batch_size,max_seq_len,num_class])

    return out

def inference_wordemdedding(input_data_wordemdedding,seq_lens,max_seq_len,
                            rnn_size,layers_size,keep_prob_ph):

    with tf.name_scope('inference_embedding'):
        out = network(input_data_wordemdedding,seq_lens,max_seq_len,rnn_size,layers_size,keep_prob_ph)

    return out












# -*- coding: UTF-8 -*-
import os
# os.chdir('G:\\SS_server\\netword_pred')
os.chdir('/home/3s540/SSsoftware/SS_server/netword_pred')
import sys

from generate_features import *
import tensorflow as tf
import numpy as np
from inference_hhm import *
from inference_pp import *
from inference_pssm import *
from inference_pssm_count import *
from inference_wordemdedding import *
from assemble_utils import *
from assemble_config import *


def test_assemble():
    gen_all_features()
    test_data_hhm,test_data_pp,test_data_pssm,test_data_pssm_count,test_data_wordembedding = get_all_data()

    with tf.name_scope('assemble_place_holder'):
        keep_prob_ph = tf.placeholder(tf.float32)
        seq_lens = tf.placeholder(tf.int32,shape=[None])
        max_seq_len = tf.reduce_max(seq_lens, name='max_len')

    with tf.name_scope('hhm_place_holder'):
        input_data_hhm = tf.placeholder(tf.float32,shape=[None,None,30])

    out_hhm = inference_hhm(input_data_hhm,seq_lens,max_seq_len,128,layers_size,keep_prob_ph)

    with tf.name_scope('pssm_place_holder'):
        input_data_pssm = tf.placeholder(tf.float32,shape=[None,None,20])

    out_pssm = inference_pssm(input_data_pssm,seq_lens,max_seq_len,100,layers_size,keep_prob_ph)

    with tf.name_scope('pssm_count_place_holder'):
        input_data_pssm_count = tf.placeholder(tf.float32,shape=[None,None,20])

    out_pssm_count = inference_pssm_count(input_data_pssm_count,seq_lens,max_seq_len,100,layers_size,keep_prob_ph)
    with tf.name_scope('pp_place_holder'):
        input_data_pp = tf.placeholder(tf.float32,shape=[None,None,7])

    out_pp = inference_pp(input_data_pp,seq_lens,max_seq_len,64,layers_size,keep_prob_ph)

    with tf.name_scope('wordemdedding_place_holder'):
        input_data_wordemdedding = tf.placeholder(tf.int32,shape=[None,None])

    out_wordemdedding = inference_wordemdedding(
        input_data_wordemdedding,seq_lens,max_seq_len,128,layers_size,keep_prob_ph)

    with tf.name_scope('secondary_level'):
        second_inputs = tf.concat([out_hhm,out_pssm,out_pssm_count,
                                   out_pp,out_wordemdedding],axis=-1)#[B,T,15]
        # ss = second_inputs.get_shape().as_list()[-1]# ss = 15
        # second_inputs_flatten = tf.reshape(second_inputs,[-1,ss])# shape=(?,5)

    with tf.name_scope('second_lstm'):
        fw = tf.contrib.rnn.LSTMCell(second_rnn_size)
        bw = tf.contrib.rnn.LSTMCell(second_rnn_size)
        fw_cell_zero = fw.zero_state(batch_size, tf.float32)
        bw_cell_zero = bw.zero_state(batch_size, tf.float32)
        second_enc_out,_ = tf.nn.bidirectional_dynamic_rnn(fw,bw,second_inputs,sequence_length=seq_lens,
                                                           initial_state_fw=fw_cell_zero,initial_state_bw=bw_cell_zero)
        second_enc_outs = tf.concat(second_enc_out,2)# [B,T,2*second_rnn_size]
        second_inputs_flatten = tf.reshape(second_enc_outs,[-1, second_rnn_size*2])
        W = tf.Variable(tf.truncated_normal([second_rnn_size*2, num_class], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_class]))
        out_fc1 = tf.matmul(second_inputs_flatten, W)+b
        # out_fc1 = tf.squeeze(out_fc1,axis=-1)
        second_out = tf.reshape(out_fc1,[batch_size,max_seq_len,num_class])
        second_out = tf.nn.softmax(second_out)

    masks = tf.sequence_mask(seq_lens, max_seq_len)
    second_out_masked = tf.boolean_mask(second_out,masks)
    second_preds = tf.argmax(second_out_masked,axis=1)


    saver = tf.train.Saver()
    # real_number = 0
    with tf.Session() as sess:
        saver.restore(sess,"model/model_whole.ckpt")
        l_test = len(test_data_hhm)
        batch_lens,batch_hhm,batch_pssm,batch_pssm_count,batch_pp,batch_wordembedding = get_assemble_test_batch(test_data_hhm,test_data_pssm,test_data_pssm_count,test_data_pp,test_data_wordembedding,0,l_test)

        # real_number = real_number + sum(batch_lens[0:l_test])
        feed_dict = {input_data_hhm:batch_hhm,input_data_pssm:batch_pssm,input_data_pssm_count:batch_pssm_count,input_data_pp:batch_pp,
                             input_data_wordemdedding:batch_wordembedding,seq_lens:batch_lens,keep_prob_ph:1.0}
        logits,predictions = sess.run([second_out_masked,second_preds],feed_dict=feed_dict)

    # logits=logits[:real_number]
    # predictions = predictions[:real_number]

    return logits,predictions

def result_written(logits,predictions):
    f_in = open(rawseq_dir,'r')
    seqs = []
    seq_names = []
    test_lens = []
    while True:
        line = f_in.readline().strip()
        if not line:
            break
        if line.startswith('>'):
            seq_names.append(line[1:])

        if not line.startswith('>'):
            seqs.append(list(line))
            test_lens.append(len(line))

    f_in.close()
    num_seq = len(seq_names)

    f_logits = []
    f_ss = []
    n  = -1
    for i in range(num_seq):
        cur_ss = []
        cur_logits = np.zeros((test_lens[i],3),np.float32)
        for j in range(test_lens[i]):
            n += 1
            if predictions[n] == 0:
                cur_ss.append('H')
            elif predictions[n] == 1:
                cur_ss.append('E')
            else:
                cur_ss.append('C')

            cur_logits[j] = logits[n]

        f_ss.append(cur_ss)
        f_logits.append(cur_logits)


    f = open(save_dir,'w')

    f.write('#Bi-LSTM based ensemble algorithm SS3: three-class secondary structure prediction results'+'\n')
    f.write('#probabilities are in the order of H E C, the 3 secondary structure types used in DSSP '+'\n')

    for i in range(num_seq):
        f.write('\n')
        f.write('\n')
        f.write('>'+seq_names[i])
        f.write('\n')
        temp_len = test_lens[i]
        for j in range(temp_len):
            f.write(seqs[i][j]+'\t')
            f.write(f_ss[i][j]+'\t')
            f.write('%.5f' % f_logits[i][j][0]+'\t'+'\t')
            f.write('%.5f' % f_logits[i][j][1]+'\t'+'\t')
            if i == num_seq-1 and j == temp_len -1:
                f.write('%.5f' % f_logits[i][j][2])
            else:
                f.write('%.5f' % f_logits[i][j][2]+'\n')

    f.close()


def main():
    rmfiles('../pssm_data/')
    rmfiles('../hhm_data/')
    print('old datas have been removed!')
    logits,predictions = test_assemble()
    result_written(logits,predictions)
    print('result has restored!')
    print('finish!')

if __name__ == '__main__':
    main()

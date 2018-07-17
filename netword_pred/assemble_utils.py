# -*- coding: UTF-8 -*-
import numpy as np
import os
from assemble_config import *

def rmfiles(top_path):
    files = os.listdir(top_path)
    for f in files:
        # print(f)
        f_path = top_path+f
        os.remove(f_path)


def getdata_hhm():
    train_data_letter = np.load('../test_features/test_hhm.npy')
    # train_data_letter = np.load('../data/nor_hhm_5772_exclude.npy')
    train_data = train_data_letter

    return train_data

def getdata_pp():
    train_data_letter = np.load('../test_features/test_pps.npy')
    # train_data_letter = np.load('../data/nor_pps_5772_exclude.npy')
    train_data = train_data_letter

    return train_data

def getdata_pssm():
    # train_data_letter = np.load('../data/nor_pssm_5772_exclue.npy')
    train_data_letter = np.load('../test_features/test_pssm.npy')
    train_data = train_data_letter

    return train_data

def getdata_pssm_count():
    # train_data_letter = np.load('../data/nor_count_5772_exclude.npy')
    train_data_letter = np.load('../test_features/test_pssm_count.npy')
    train_data = train_data_letter

    return train_data

def getdata_wordembedding():
    # train_data_letter = np.load('../data/nor_wordids_5772_exclude.npy')
    train_data_letter = np.load('../test_features/test_wordids.npy')
    train_data = train_data_letter

    return train_data


def get_batch_hhm(data,start,end):
    cur_data = data[start:end]
    batch_lengths = []
    length = len(cur_data)
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        batch_lengths.append(cur_len)
    if length!=batch_size:
        for j in range(length,batch_size):
            batch_lengths.append(0)

    batch_max_len = max(batch_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, batch_max_len,30],dtype=np.float32) # == PAD
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        for j in range(cur_len):
            pp = cur_p[j]
            inputs_batch_major[i,j,:] = pp

    return inputs_batch_major,batch_lengths



def get_batch_pssm(data,start,end):
    cur_data = data[start:end]
    batch_lengths = []
    length = len(cur_data)
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        batch_lengths.append(cur_len)
    if length!=batch_size:
        for j in range(length,batch_size):
            batch_lengths.append(0)

    batch_max_len = max(batch_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, batch_max_len,20],dtype=np.float32) # == PAD
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        for j in range(cur_len):
            pp = cur_p[j]
            inputs_batch_major[i,j,:] = pp

    return inputs_batch_major,batch_lengths


def get_batch_pssm_count(data,start,end):
    cur_data = data[start:end]
    batch_lengths = []
    length = len(cur_data)
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        batch_lengths.append(cur_len)

    if length!=batch_size:
        for j in range(length,batch_size):
            batch_lengths.append(0)

    batch_max_len = max(batch_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, batch_max_len,20],dtype=np.float32) # == PAD
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        for j in range(cur_len):
            pp = cur_p[j]
            inputs_batch_major[i,j,:] = pp

    return inputs_batch_major,batch_lengths

def get_batch_pp(data,start,end):
    cur_data = data[start:end]
    batch_lengths = []
    length = len(cur_data)
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        batch_lengths.append(cur_len)
    if length!=batch_size:
        for j in range(length,batch_size):
            batch_lengths.append(0)

    batch_max_len = max(batch_lengths)
    inputs_batch_major = np.zeros(shape=[batch_size, batch_max_len,7],dtype=np.float32) # == PAD
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        for j in range(cur_len):
            pp = cur_p[j]
            inputs_batch_major[i,j,:] = pp

    return inputs_batch_major,batch_lengths


def get_batch_wordembedding(data,start,end):
    cur_data = data[start:end]
    batch_lengths = []
    length = len(cur_data)
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        batch_lengths.append(cur_len)

    if length!=batch_size:
        for j in range(length,batch_size):
            batch_lengths.append(0)

    batch_max_len = max(batch_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, batch_max_len],dtype=np.int32)
    for i in range(length):
        cur_p = cur_data[i]
        cur_len = len(cur_p)
        for j in range(cur_len):
            pp = cur_p[j]
            inputs_batch_major[i,j] = pp

    return inputs_batch_major,batch_lengths

def get_all_data():
    train_data_hhm = getdata_hhm()
    train_data_pp = getdata_pp()
    train_data_pssm = getdata_pssm()
    train_data_pssm_count = getdata_pssm_count()
    train_data_wordembedding = getdata_wordembedding()

    return train_data_hhm,train_data_pp,train_data_pssm,train_data_pssm_count,train_data_wordembedding

# def get_assemble_train_batch(train_data_hhm,train_data_pssm,train_data_pssm_count,train_data_wordembedding,train_data_pp,train_label,start,end):
def get_assemble_test_batch(train_data_hhm,train_data_pssm,
                             train_data_pssm_count,train_data_pp,train_data_wordembedding,start,end):
    batch_hhm,batch_lens = get_batch_hhm(train_data_hhm,start,end)
    batch_pssm,_ = get_batch_pssm(train_data_pssm,start,end)
    batch_pssm_count,_ = get_batch_pssm_count(train_data_pssm_count,start,end)
    batch_pp,_ = get_batch_pp(train_data_pp,start,end)
    batch_wordembedding,_ = get_batch_wordembedding(train_data_wordembedding,start,end)

    return batch_lens,batch_hhm,batch_pssm,batch_pssm_count,batch_pp,batch_wordembedding


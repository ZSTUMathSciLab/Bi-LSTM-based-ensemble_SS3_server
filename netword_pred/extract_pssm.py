import numpy as np
import os
import glob as glob

def extrat_one(file):

    idx_res = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18)
    f = open(file)
    line = f.readline().strip()
    line = f.readline().strip()
    line = f.readline().strip()#
    one_pssm = []
    one_count = []
    while True:
        line = f.readline().strip()
        if not line:
            break
        split_line = line.replace('-', ' -').split()
        if not len(split_line) == 44:
            print('length not match ')
            print(line)
            break

        pssm_temp = [-float(i) for i in split_line[2:22]]
        pssm_temp = [pssm_temp[k] for k in idx_res]#
        one_pssm.append(pssm_temp)

        count_temp = [float(i)/100.0 for i in split_line[22:42]]
        count_temp = [count_temp[k] for k in idx_res]#
        one_count.append(count_temp)

    f.close()
    return one_pssm,one_count

def process_pssm():

    basepath = '../pssm_data/'
    filelist = glob.glob('../pssm_data/*.pssm')
    file_num = len(filelist)
    pssm_all = []
    pssm_count_all = []
    for i in range(0,file_num):
        temp_f = open( basepath+'seq'+str(i+1))
        l = temp_f.readline()
        l = temp_f.readline().strip()
        len_seq = len(l)
        temp_f.close()
        name = 'seq'+str(i+1) + '.pssm'
        file = basepath+name
        one_pssm,one_count = extrat_one(file)
        if not len_seq == len(one_pssm):
            print('length error!  ',str(i+1))
        pssm_all.append(one_pssm)
        pssm_count_all.append(one_count)

    np.save('../test_features/test_pssm.npy',pssm_all)
    np.save('../test_features/test_pssm_count.npy',pssm_count_all)


if __name__ == '__main__':
    process_pssm()
    print('ok')



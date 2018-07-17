import numpy as np
import os
import glob as glob

def read_hhm(file):
    P_hhm = []
    f = open(file)
    for i in range(11):
        line = f.readline().strip()
    while line[0]!='#':
        line = f.readline().strip()
    line = f.readline().strip()
    line = f.readline().strip()
    line = f.readline().strip()
    line = f.readline().strip()

    while True:
        line1 = f.readline().strip()
        line2 = f.readline().strip()
        _ = f.readline().strip()#
        if line1[0:2]=='//':
            break
        temp_hhm = np.zeros((30,),np.float32)
        lineinfo1 = line1.split()
        probs_ = [2**(-float(lineinfo1[i])/1000) if lineinfo1[i]!='*' else 0. for i in range(2,22)]
        temp_hhm[0:20] = probs_
        lineinfo2 = line2.split()
        extras_ = [2**(-float(lineinfo2[i])/1000) if lineinfo2[i]!='*' else 0. for i in range(0,10)]
        temp_hhm[20:] = extras_

        P_hhm .append(temp_hhm)

    f.close()
    return P_hhm

def process_hhm():
    test_hhm = []
    filelist = glob.glob('../hhm_data/*.hhm')
    file_num = len(filelist)
    for i in range(file_num):
        # print(str(i+1)+' is process...')
        file = '../hhm_data/seq'+str(i+1)+'.hhm'
        P_hhm = read_hhm(file)
        test_hhm.append(P_hhm)

    np.save('../test_features/test_hhm.npy',test_hhm)

if __name__ == '__main__':
    process_hhm()
    print('ok')
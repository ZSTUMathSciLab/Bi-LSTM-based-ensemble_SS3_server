# coding:utf-8
import os

def get_one_hhm(name):
    inp = '/home/3s540/SSsoftware/HHsuite/hhsuite3/bin/hhblits -cpu 4 -i '+name
    out = '-ohhm ' + name+'.hhm'
    db = '-d /home/3s540/SSsoftware/HHsuite/uniprot20/uniprot20_2016_02 -n 2'
    ll = [inp,out,db]
    com = ' '.join(ll)
    os.system(com)

def gen_mutil_hhm(mutil_seqs,cc):
    f_multi = open(mutil_seqs)
    while True:
        line = f_multi.readline().strip()
        if not line:
            break
        if line.startswith('>'):
            # name = line[1:]
            cc = cc +1
            name = '../hhm_data/seq'+str(cc)
            f = open(name,'w')
            f.write(line+'\n')
            line = f_multi.readline().strip()
            f.write(line+'\n')
            f.close()

        # print(com)
        get_one_hhm(name)


    f_multi.close()

# if __name__ == '__main__':
#     cc =0
#     mutil_seqs = '../raw_data/mutil_seqs'
#     gen_mutil_hhm(mutil_seqs,cc)

    # print(com)

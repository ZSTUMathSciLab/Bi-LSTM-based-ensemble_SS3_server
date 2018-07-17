import os

def get_one_pssm(name):
    inp = '/home/3s540/SSsoftware/blast/bin/psiblast  -num_iterations 3 -num_alignments 1 -db /home/3s540/SSsoftware/swissprot_db/swissprot  -num_threads 16 -query '+name
    out = '-out_ascii_pssm ' + name+'.pssm'
    ll = [inp,out]
    com = ' '.join(ll)
    # print(com)

    os.system(com)

def gen_mutil_pssm(mutil_seqs,cc):
    f_multi = open(mutil_seqs)
    while True:
        line = f_multi.readline().strip()
        if not line:
            break
        if line.startswith('>'):
            # name = line[1:]
            cc = cc +1
            name = '/home/3s540/SSsoftware/SS_server/pssm_data/seq'+str(cc)
            f = open(name,'w')
            f.write(line+'\n')
            line = f_multi.readline().strip()
            f.write(line+'\n')
            f.close()

        # print(com)
        get_one_pssm(name)

    f_multi.close()


if __name__ == '__main__':
    mutil_seqs = '../raw_data/mutil_seqs'
    cc = 0
    gen_mutil_pssm(mutil_seqs,cc)
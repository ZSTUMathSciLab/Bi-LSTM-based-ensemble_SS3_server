import numpy as np

def word_id(file):
    TEMPLE_ACC = 'ACDEFGHIKLMNPQRSTVWY'
    f = open(file)
    all_wordids = []
    while True:
        line = f.readline().strip()
        if not line:
            break
        if not line.startswith('>'):
            cur_len = len(line)
            # one_line = np.zeros((cur_len,),np.int32)
            one_line = []
            for j in range(cur_len):
                aac = line[j]
                ind = TEMPLE_ACC.find(aac)
                if ind<0:
                    ind = 20
                one_line.append(ind+1)#
                # one_line[j] = ind
            all_wordids.append(one_line)

    f.close()
    np.save('../test_features/test_wordids.npy',all_wordids)
    # return all_wordids

if __name__ == '__main__':
    mutil_seqs = '../raw_data/mutil_seqs'
    word_id(mutil_seqs)
    print('word id OK')
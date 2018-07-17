# -*- coding: UTF-8 -*-
import numpy as np
from extract_hhm import *
from extract_pssm import *
from generate_pp import *
from word_ids import *
from generate_pssm import *
from generate_hhm import *

def gen_all_features():
    mutil_seqs = '../raw_data/mutil_seqs'
    cc = 0
    generate_pp(mutil_seqs)
    print('pp ok')
    word_id(mutil_seqs)
    print('word_id ok')

    gen_mutil_hhm(mutil_seqs,cc)
    process_hhm()
    print('hhm ok')
    gen_mutil_pssm(mutil_seqs,cc)
    process_pssm()
    print('pssm ok')
    print('pssm count ok')

    print('features generated!')

if __name__ == '__main__':
    gen_all_features()


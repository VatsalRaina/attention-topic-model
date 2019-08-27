#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from core.utilities.utilities import text_to_array

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
#commandLineParser.add_argument('prompt_path', type=str,
 #                              help='which should be loaded')
#commandLineParser.add_argument('save_path', type=str,
 #                              help='which should be loaded')
commandLineParser.add_argument('--strip_start_end', action='store_true', help='whether to strip the <s> </s> marks at the beginning and end of prompts in sorted_topics.txt file (used for legacy sorted_topics.txt formatting')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_compute_response_vectors.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    if args.strip_start_end:
        print("Stripping the first and last word (should correspond to <s> and </s> marks) from the input prompts. Should only be used with legacy dataset formatting")

    # Get indices for each word
    vocab_file = '/home/alta/relevance/vr311/data/input.wlist.index'
    resp_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/shuffled/responses.txt' 
    resps, resp_lens = text_to_array(resp_path, vocab_file)

    tf_resps = tf.convert_to_tensor(resps)
    tf_resp_lens = tf.convert_to_tensor(resp_lens) 
    
    save_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/shuffled_as_embedding'

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        resps, resp_lens = sess.run([tf_resps, tf_resp_lens])

    path = os.path.join(save_path, 'response_ids.txt')
    np.savetxt(path, tf_resps)
    path2 = os.path.join(save_path, 'response_lens.txt')
    np.savetxt(path2, tf_resp_lens)


if __name__ == '__main__':
    main()

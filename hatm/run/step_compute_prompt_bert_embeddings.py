#! /usr/bin/env python

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

from core.utilities.bertEmbeddings import BertEmbeddings
from core.utilities.utilities import text_to_array_bert

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
    with open('CMDs/step_compute_prompt_bert_embeddings.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    vocab_file = '/home/alta/relevance/vr311/uncased_L-12_H-768_A-12/vocab.txt'
    prompt_path = '/home/alta/relevance/vr311/data_vatsal/LINSK/tfrecords_train/sorted_topics.txt'
    prompts, _ = text_to_array_bert(prompt_path, vocab_file)

    # Create instance of Bert sentence embedding (BSE)
    bse = BertEmbeddings()
    bse.create_model()
    embeddings = bse.get_embeddings(prompts)

    save_path = '/home/alta/relevance/vr311/models_new/correct_bert/ATM'
    path = os.path.join(save_path, 'sorted_prompt_embeddings_eval_unseen.txt')
    np.savetxt(path, embeddings)


if __name__ == '__main__':
    main()

#! /usr/bin/env python

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

from core.utilities.utilities import text_to_array_bert
import core.utilities.bert.modeling as modeling
import core.utilities.bert.tokenization as tokenization

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

    if args.strip_start_end:
        print("Stripping the first and last word (should correspond to <s> and </s> marks) from the input prompts. Should only be used with legacy dataset formatting")

    # Tokenise prompts
    vocab_file = '/home/alta/relevance/vr311/uncased_L-12_H-768_A-12/vocab.txt'
    prompt_path = '/home/alta/relevance/vr311/data_vatsal/LINSK/tfrecords_train/sorted_topics.txt' 
    prompts, prompt_lens = text_to_array_bert(prompt_path, vocab_file)
    
    # Get BERT embeddings

    tf_prompts = tf.convert_to_tensor(prompts)
    tf_prompt_lens = tf.convert_to_tensor(prompt_lens)

    tf_mask = tf.sequence_mask(tf_prompt_lens, tf.shape(tf_prompts)[1])

    config = modeling.BertConfig(vocab_size=32000)

    model = modeling.BertModel(config=config, is_training=False, input_ids=tf_prompts, input_mask=tf_mask)

    pooled_output = model.get_pooled_output()

    print(pooled_output.shape)

    save_path = '/home/alta/relevance/vr311/models_new/ATM'

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        embeddings = sess.run(pooled_output)
    path = os.path.join(save_path, 'sorted_prompt_embeddings_eval_unseen_improved.txt')
    np.savetxt(path, embeddings)


if __name__ == '__main__':
    main()

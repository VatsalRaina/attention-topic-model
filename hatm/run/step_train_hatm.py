#! /usr/bin/env python

import argparse
import os
import sys

import tensorflow as tf

from core.utilities.utilities import text_to_array
from hatm.hatm import HierarchicialAttentionTopicModel

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--valid_size', type=int, default=14188,  # 1034,##28375,
                               help='Specify the validation set size')
commandLineParser.add_argument('--batch_size', type=int, default=100,
                               help='Specify the training batch size')
commandLineParser.add_argument('--learning_rate', type=float, default=1e-3,
                               help='Specify the intial learning rate')
commandLineParser.add_argument('--lr_decay', type=float, default=0.85,
                               help='Specify the learning rate decay rate')
commandLineParser.add_argument('--dropout', type=float, default=1.0,
                               help='Specify the dropout keep probability')
commandLineParser.add_argument('--attention_dropout', type=float, default=1.0,
			       help='Specify the dropout keep probability for attention weights')
commandLineParser.add_argument('--n_epochs', type=int, default=1,
                               help='Specify the number of epoch to run training for')
commandLineParser.add_argument('--n_samples', type=int, default=1,
                               help='Specify the number of negative samples to take')
commandLineParser.add_argument('--seed', type=int, default=100,
                               help='Specify the global random seed')
commandLineParser.add_argument('--name', type=str, default='model',
                               help='Specify the name of the model')
commandLineParser.add_argument('--debug', type=int, choices=[0, 1, 2], default=0,
                               help='Specify the debug output level')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--init', type=str, default=None,
                               help='Specify path to from which to initialize model')
commandLineParser.add_argument('--distortion', type=float, default=1.0,
                               help='Specify whether to use uniform negative sampliong')
commandLineParser.add_argument('--epoch', type=str, default=None,
                               help='which should be loaded')
commandLineParser.add_argument('train_data', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('train_size', type=int,
                               help='which should be loaded')
commandLineParser.add_argument('valid_data', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('topic_path', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('topic_count_path', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('wlist_path', type=str,
                               help='which should be loaded')



def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_attention_grader.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    topics, topic_lens = text_to_array(args.topic_path, input_index=args.wlist_path)

    atm = HierarchicialAttentionTopicModel(network_architecture=None,
                             seed=args.seed,
                             name=args.name,
                             save_path='./',
                             load_path=args.load_path,
                             debug_mode=args.debug,
                             epoch=args.epoch)

    atm.fit(train_data=args.train_data,
            valid_data=args.valid_data,
            load_path=args.init,
            topics=topics,
            topic_lens=topic_lens,
            unigram_path=args.topic_count_path,
            train_size=args.train_size,
            learning_rate=args.learning_rate,
            lr_decay=args.lr_decay,
            dropout=args.dropout,
	    attention_dropout=args.attention_dropout,
            distortion=args.distortion,
            batch_size=args.batch_size,
            optimizer=tf.train.AdamOptimizer,
            optimizer_params={},
            n_epochs=args.n_epochs,
            n_samples=args.n_samples,
            epoch=0)

    atm.save()


if __name__ == '__main__':
    main()

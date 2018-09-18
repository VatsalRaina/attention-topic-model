"""
Distill knowledge from an ensemble
"""
#! /usr/bin/env python

import argparse
import os
import sys

import tensorflow as tf

import context
from core.utilities.utilities import text_to_array
from atm.atm import AttentionTopicModel
from atm.ensemble import Ensemble

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--valid_size', type=int, default=14188,
                               help='Specify the validation set size')
commandLineParser.add_argument('--batch_size', type=int, default=100,
                               help='Specify the training batch size')
commandLineParser.add_argument('--learning_rate', type=float, default=1e-3,
                               help='Specify the intial learning rate')
commandLineParser.add_argument('--lr_decay', type=float, default=0.85,
                               help='Specify the learning rate decay rate')
commandLineParser.add_argument('--dropout', type=float, default=1.0,
                               help='Specify the dropout keep probability')
commandLineParser.add_argument('--n_epochs', type=int, default=6,
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
                               help='which epoch should be loaded')
commandLineParser.add_argument('--strip_start_end', action='store_true', help='whether to strip the <s> </s> marks at the beginning and end of prompts in sorted_topics.txt file (used for legacy sorted_topics.txt formatting')
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
commandLineParser.add_argument('ensemble_dir', type=str,
                               help='which should be loaded')
commandLineParser.add_argument('--ensemble_size', type=int, default=10, help='number of base learners in the ensemble '
                                                                             'directory to use')
commandLineParser.add_argument('--ensemble_epoch', type=int, default=6, help='epoch of the base models in the ensemble '
                                                                             'to load for the teacher')



def main(args):
    # Cache the command
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_distill.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    topics, topic_lens = text_to_array(args.topic_path, args.wlist_path, strip_start_end=args.strip_start_end)
    if args.strip_start_end:
        print("Stripping the first and last word (should correspond to <s> and </s> marks) from the input prompts. "
              "Should only be used with legacy dataset formatting")

    # Load the trained base learners from the ensemble

    ensemble_models = []
    for i in range(1, args.ensemble_size + 1):
        base_model_path = os.path.join(args.ensemble_dir, 'atm_seed_' + str(i))
        print("Loading ensemble model at:\n", base_model_path)
        base_model = AttentionTopicModel(network_architecture=None,
                                         load_path=base_model_path,
                                         epoch=args.ensemble_epoch)
        ensemble_models.append(base_model)

    ensemble = Ensemble(ensemble_models, tf.reduce_mean)

    atm = AttentionTopicModel(network_architecture=None,
                              seed=args.seed,
                              name=args.name,
                              save_path='./',
                              load_path=args.load_path,
                              debug_mode=args.debug,
                              epoch=args.epoch)

    atm.fit_teacher(train_data=args.train_data,
                    valid_data=args.valid_data,
                    load_path=args.init,
                    topics=topics,
                    topic_lens=topic_lens,
                    unigram_path=args.topic_count_path,
                    teacher=ensemble,
                    train_size=args.train_size,
                    learning_rate=args.learning_rate,
                    lr_decay=args.lr_decay,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    distortion=args.distortion,
                    optimizer=tf.train.AdamOptimizer,
                    optimizer_params={},
                    n_epochs=args.n_epochs,
                    n_samples=args.n_samples,
                    epoch=0)

    atm.save()


if __name__ == '__main__':
    args = commandLineParser.parse_args()
    main(args)
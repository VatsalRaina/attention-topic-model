#! /usr/bin/env python
from __future__ import print_function, division
from builtins import range

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score as roc

from atm.atm import AttentionTopicModel
from core.utilities.utilities import text_to_array

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--debug', type=int, default=0,
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--epoch', type=str, default=None,
                               help='which should be loaded')
commandLineParser.add_argument('data_pattern', type=str,
                               help='absolute path to response data')
commandLineParser.add_argument('name', type=str, default=None,
                               help='which should be loaded')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_test_attention_grader.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')


    # Initialize and Run the Model
    atm = AttentionTopicModel(network_architecture=None,
                              load_path=args.load_path,
                              debug_mode=args.debug,
                              epoch=args.epoch)

    test_labels, test_probs, test_loss = atm.predict(args.data_pattern)
    # Do evaluations, calculate metrics, etc...

    format_str = ('Test Loss, %.4f, Test ROC AUC = %.3f')
    roc_score = roc(np.squeeze(test_labels), np.squeeze(test_probs))
    print(format_str % (test_loss, roc_score))


if __name__ == '__main__':
    main()

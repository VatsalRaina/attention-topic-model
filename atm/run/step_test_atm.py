#! /usr/bin/env python

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
sns.set()

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
commandLineParser.add_argument('output_dir', type=str, default=None,
                               help='which should be loaded')
commandLineParser.add_argument('name', type=str, default=None,
                               help='which should be loaded')



def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_test_attention_grader.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)



    # Initialize and Run the Model
    atm = AttentionTopicModel(network_architecture=None,
                              load_path=args.load_path,
                              debug_mode=args.debug,
                              epoch=args.epoch)

    test_labels, test_probs, test_loss = atm.predict(args.data_pattern)

    data=np.concatenate((test_labels[:,np.newaxis], test_probs[:,np.newaxis]), axis=1)
    np.savetxt(os.path.join(args.output_dir, 'labels-probs.txt'), data)
    # Do evaluations, calculate metrics, etc...
    roc_score = roc(np.squeeze(test_labels), np.squeeze(test_probs))


    fpr, tpr, thresholds = roc_curve(np.asarray(np.squeeze(test_labels), dtype=np.int32), test_probs)
    plt.plot(fpr, tpr, c='r')
    plt.plot([0, 1], [0, 1], 'k--', lw=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(os.path.join(args.output_dir, 'test_roc_curve.png'))
    plt.close()


    precision, recall, thresholds = precision_recall_curve(test_labels, test_probs)
    aupr_rel = auc(recall, precision)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0,1.0)
    plt.xlim(0.0,1.0)
    plt.savefig(os.path.join(args.output_dir, 'test_pr_curve_relevant.png'))
    plt.close()

    precision, recall, thresholds = precision_recall_curve(1-test_labels, 1.0-test_probs)
    aupr_nonrel = auc(recall, precision)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0,1.0)
    plt.xlim(0.0,1.0)
    plt.savefig(os.path.join(args.output_dir, 'test_pr_curve_off-topic.png'))
    plt.close()

    with open(os.path.join(args.output_dir, 'results.txt'), 'a') as f:
        f.write('ROC AUC:' + str(np.round(roc_score,3)) + '\n')
        f.write('ROC PR Detect Relevant:' + str(np.round(aupr_rel, 3)) + '\n')
        f.write('ROC PR Detect Non-Relevant:' + str(np.round(aupr_nonrel, 3)) + '\n')
        f.write('Cross Entropy:' + str(np.round(test_loss, 3)) + '\n')

if __name__ == '__main__':
    main()

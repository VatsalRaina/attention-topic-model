from __future__ import print_function, division

import matplotlib

matplotlib.use('Agg')
import os
import numpy as np
import scipy.stats
import math
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches as mpatches

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

import seaborn as sns

sns.set()

matplotlib.rcParams['savefig.dpi'] = 200

parser = argparse.ArgumentParser(description='Calculate and retrieve numerical results for the ensemble.')
parser.add_argument('models_parent_dir', type=str, help='Path to ensemble directory')
parser.add_argument('--rel_labels_path', type=str, default='labels-probs.txt')


# parser.add_argument('--savedir', type=str, default='./',
#                     help='Path to directory where to save the plots.')
# parser.add_argument('--rel_attention_path', type=str, default='eval4_naive/prompt_attention.txt')
# parser.add_argument('--n_levels', type=int, default=20)
# parser.add_argument('--hatm', action='store_true', help='Whether to analyse ATM or HATM ensemble')


def run_misclassification_detection(misclassification_labels, uncertainty):
    precision, recall, thresholds = precision_recall_curve(misclassification_labels, uncertainty, pos_label=1)
    aupr_pos = auc(recall, precision)

    precision, recall, thresholds = precision_recall_curve(misclassification_labels, -uncertainty, pos_label=0)
    aupr_neg = auc(recall, precision)

    roc_auc = roc_auc_score(misclassification_labels, uncertainty)

    return [roc_auc, aupr_pos, aupr_neg]


def run_misclassification_detection_over_ensemble(labels, predictions, uncertainty, uncertainty_name, savedir=None):
    """
    This functions runs a uncertainty based miclassification detection experiment

    :param labels: Labels on-topic / off-topic
    :param predictions: array of predictions (probabilities) of [batch_size, size_ensemble]
    :param prompt_entopies: array of entropies of prompt attention mechanism [batch_size, size_ensemble]
    :return: None. Saves stuff
    """

    off_topic_probabilities = 1.0 - predictions

    on_topic_probabilities = np.expand_dims(predictions, axis=1)
    off_topic_probabilities = np.expand_dims(off_topic_probabilities)

    probabilities = np.concatenate([off_topic_probabilities, on_topic_probabilities], axis=1)

    entropies = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

    # Threshold the predictions with threshold 0.5 (take prediction to be the class with max prob.)
    predictions = np.argmax(probabilities, axis=1)  # on-topic is one

    misclassification = np.asarray(labels[:, np.newaxis] != predictions, dtype=np.int32)
    correct = np.asarray(labels[:, np.newaxis] == predictions, dtype=np.float32)

    accuracies = np.mean(correct, axis=0)
    m_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    auc_array_entropy = []
    for i in range(predictions.shape[-1]):
        auc = run_misclassification_detection(misclassification[:, i], entropies[:, i])
        auc_array_entropy.append(auc)

    auc_entropy = np.stack(auc_array_entropy, axis=0)
    auc_entropy_mean, auc_entropy_std = np.mean(auc_entropy, axis=0), np.std(auc_entropy, axis=0)

    # if prompt_entropies is not None:
    #     auc_array_pentropy = []
    #     for i in range(predictions.shape[-1]):
    #         auc = run_misclassification_detection(misclassification[:, i], prompt_entropies[:, i])
    #         auc_array_pentropy.append(auc)
    #         auc_pentropy = np.stack(auc_array_pentropy, axis=0)
    #         auc_pentropy_mean, auc_pentropy_std = np.mean(auc_pentropy, axis=0), np.std(auc_pentropy, axis=0)

    if savedir:
        with open(os.path.join(savedir, 'misclassification_detect_individual.txt'), 'w') as f:
            f.write('Mean Accuracy = ' + str(m_accuracy) + '+/-' + str(std_accuracy) + '\n')
            f.write('entropy ROC AUC: ' + str(auc_entropy_mean[0]) + ' +/- ' + str(auc_entropy_std[0]) + '\n')
            f.write('entropy AUPR POS: ' + str(auc_entropy_mean[1]) + ' +/- ' + str(auc_entropy_std[1]) + '\n')
            f.write('entropy AUPR NEG: ' + str(auc_entropy_mean[2]) + ' +/- ' + str(auc_entropy_std[2]) + '\n')

            # if prompt_entropies is not None:
            #     f.write(
            #         'prompt entropy ROC AUC: ' + str(auc_pentropy_mean[0]) + ' +/- ' + str(auc_pentropy_std[0]) + '\n')
            #     f.write(
            #         'prompt entropy AUPR POS: ' + str(auc_pentropy_mean[1]) + ' +/ -' + str(auc_pentropy_std[1]) + '\n')
            #     f.write(
            #         'prompt entropy AUPR NEG: ' + str(auc_pentropy_mean[2]) + ' +/ -' + str(auc_pentropy_std[2]) + '\n')

    return entropies

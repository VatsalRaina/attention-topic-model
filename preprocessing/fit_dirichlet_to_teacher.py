#! /usr/bin/env python

"""
OBSOLETE


Fits parameters alpha to teacher predictions for each example as to maximise log-likelihood.

Creates a file:
alphas.txt

"""
from __future__ import print_function

import os
import sys
import argparse

import numpy as np
import tensorflow as tf

import sys
import scipy as sp
import scipy.stats as stats


parser = argparse.ArgumentParser(description='Create k new k-fold split datasets that can be used for generating '
                                             'teacher data on unseen prompts from a single dataset.')
parser.add_argument('data_dir', type=str,
                    help='absolute path to the directory with the processed responses,'
                         'prompts, speakers, grades etc. .txt data')
parser.add_argument('--teacher_predictions_filename', type=str, default='teacher_predictions.txt')
parser.add_argument('--output_filename', type=str, default='alphas.txt')
# parser.add_argument('--batch_size', type=int, default=1000, help='batch size for fitting the 2 alpha parameters')
parser.add_argument('--learning_rate', type=float, default=0.04, help='learning rate for fitting the dirchlets')
parser.add_argument('--num_steps', type=int, default=10000, help='number of steps to fit the dirichlets')


def nll_loss(alphas, teacher_predictions):
    alpha1, alpha2 = tf.split(alphas, num_or_size_splits=2, axis=1)

    log_likelihood_const_part = tf.lgamma(alpha1 + alpha2) - tf.lgamma(alpha1) - tf.lgamma(alpha2)
    log_likelihood_var_part = tf.log(teacher_predictions) * (alpha1 - 1.0) + tf.log(1.0 - teacher_predictions) * (
        alpha2 - 1.0)
    log_likelihood = log_likelihood_const_part + log_likelihood_var_part

    nll_loss = -1.0 * tf.reduce_mean(log_likelihood, axis=1)  # Take the mean over individual ensemble predictions
    nll_cost = tf.reduce_mean(nll_loss)  # Take mean batch-wise (over the number of examples)
    return nll_cost


def main(args):
    # Get the input data

    # Get the paths to the relevant files
    teacher_pred_path = os.path.join(args.data_dir, args.teacher_predictions_filename)

    # Assert the required files exist
    if not os.path.isfile(teacher_pred_path):
        print('File: {} doesn`t exist. Exiting...'.format(teacher_pred_path))
        exit()

    # Cache the command:
    if not os.path.isdir(os.path.join(args.destination_dir, 'CMDs')):
        os.makedirs(os.path.join(args.destination_dir, 'CMDs'))
    with open(os.path.join(args.destination_dir, 'CMDs/preprocessing.cmd'), 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('-----------------------------------------------------------\n')

    # Open the files
    teacher_preds_array = np.loadtxt(teacher_pred_path, dtype=np.float32)


    # Create the graph for estimating the Dirichlet
    sess = tf.InteractiveSession()

    alphas = tf.Variable([[1., 1.]], dtype=tf.float32, trainable=True)
    teacher_predictions = tf.placeholder(dtype=tf.float32, shape=[1, 10])

    init = tf.global_variables_initializer()
    sess.run(init)

    # The reset operation
    reset_alphas = alphas.assign([[1., 1.]])

    # The alpha rounding operation (to keep alphas always non-negative)
    clip_alphas = alphas.assign(tf.clip_by_value(alphas, 1e-7, 10e7))
    log_likelihood_loss = nll_loss(alphas, teacher_predictions)

    gdo = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
    fit_op = gdo.minimize(log_likelihood_loss)


    # array to store the alpha values
    alphas_fitted = np.zeros([teacher_preds_array.shape[0], 2], dtype=np.float32)

    # Fit a Dirichlet to each example
    num_examples = teacher_preds_array.shape[0]
    for i in range(num_examples):
        sess.run(reset_alphas)
        for j in range(args.num_iter):
            sess.run(fit_op, feed_dict={teacher_predictions: teacher_preds_array[i, :]})
            sess.run(clip_alphas)
        alphas_fitted[i, :] = alphas.eval()

        # Print every n steps
        if i % 500 == 0:
            print('Step {} out of {}'.format(i, num_examples))

    # Save the results
    save_path = os.path.join(args.data_dir, args.output_filename)
    np.savetxt(save_path, alphas_fitted)

    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


#! /usr/bin/env python

from __future__ import print_function, division
import context
import argparse
import os
import sys
import shutil

import numpy as np
import tensorflow as tf

import core.utilities.tfrecord_utils as tfrecord_utils
from core.utilities.utilities import load_text

parser = argparse.ArgumentParser(description='Convert the processed data into a tfrecords dataset format')
parser.add_argument('data_dir', type=str,
                               help='absolute path to the directory with the processed responses, prompts, speakers, grades etc. .txt data')
parser.add_argument('input_wlist_path', type=str,
                               help='absolute path to input word list')
parser.add_argument('destination_dir', type=str,
                               help='absolute path to directory location where to setup and save the tfrecords data')
parser.add_argument('--valid_fraction', type=float, default=0.1,
                               help='fraction of full data to reserve for validation')
parser.add_argument('--rand_seed', type=float, default=1000,
                               help='random seed to use when shuffling the data into validation and training sets')
parser.add_argument('--preprocessing_type', type=str, choices=['train', 'test'], default='train')
parser.add_argument('--sorted_topics_path', type=str, default='',
                               help='Absolute path to file with sorted topics as used for model training. Leave unspecified if new mapping is to be generated.')


def write_to_tfrecords(filename, destination_dir, responses, prompts, q_ids, grades, speakers, targets=1.0):
    if type(targets) is float or type(targets) is int:
        # If targets is an integer make each target this value
        targets = [float(targets)] * len(responses)
    else: 
        assert type(targets) is list
        assert len(targets) == len(responses)

    # Create the training TF Record file
    print('Writing: ', filename)

    writer = tf.python_io.TFRecordWriter(os.path.join(args.destination_dir, filename))
    for response, prompt, q_id, grd, spkr, tgt in zip(responses, prompts, q_ids, grades, speakers, targets):
        if grd == '' or grd == '.':
            grd = -1
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'targets': tfrecord_utils.float_feature([tgt]),
                'grade': tfrecord_utils.float_feature([float(grd)]),
                'spkr': tfrecord_utils.bytes_feature([spkr]),
                'q_id': tfrecord_utils.int64_feature([q_id])
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'response': tfrecord_utils.int64_feature_list(response),
                'prompt': tfrecord_utils.int64_feature_list(prompt)}))
        writer.write(example.SerializeToString())
    writer.close()
    return


def generate_topic_dict(prompts_path, destination_dir):
    """Generate a topic dict mapping sentence to topic id and save files with the corresponding mappings in the destination directory"""
    # Load up the prompts as sequences of words
    with open(prompts_path, 'r') as file:
        topics = [line.replace('\n', '') for line in file.readlines()]

    # Get unique set of topics and topic counts
    unique_topics, topic_counts = np.unique(topics, return_counts=True)
    # Sort them in decreasing order
    topics = unique_topics[np.flip(np.argsort(topic_counts), 0)]
    topic_counts = np.flip(np.sort(topic_counts), 0)

    # Create dictionary for topics mapping sentence to topic id
    # Also create file of sorted topics and a unigrams file
    # Unigram file later used for training
    topic_dict = {}
    with open(os.path.join(destination_dir, 'unigrams.txt'), 'w') as ufile, open(
            os.path.join(destination_dir, 'sorted_topics.txt'), 'w') as tfile:
        for i, topic, count in zip(xrange(topics.shape[0]), topics, topic_counts):
            topic_dict[topic] = i
            ufile.write(str(i) + ',' + str(int(count)) + '\n')
            tfile.write(topic + '\n')
    return topic_dict


def load_topic_dict(sorted_topics_path):
    """Load a topic dict mapping sentence to topic id from a file generated previously for creating a training dataset."""
    topic_dict = {}
    i = 0
    # Load up sorted topics and (re)construct the topic dict so that I map each prompt word sequence to its q_id
    with open(os.path.join(args.sorted_topics_path), 'r') as tfile:
        for topic in tfile.readlines():
            topic_dict[topic.replace('\n', '')] = i
            i += 1
    return topic_dict

    


def main(args):
    """Converts a dataset to tfrecords."""

    if os.path.isdir(args.destination_dir):
        usr_response = raw_input("Destination directory exists. Type 'y' if you want to overwrite it: ")
        if usr_response != 'y':
            print('destination directory {} exists.\nExiting...'.format(args.destination_dir))
            exit()
    else:
        os.makedirs(args.destination_dir)

    shutil.copyfile(args.input_wlist_path, os.path.join(args.destination_dir, 'input.wlist.index'))

    # Get the paths to the relevant files
    responses_path = os.path.join(args.data_dir, 'responses.txt')
    prompts_path = os.path.join(args.data_dir, 'prompts.txt')
    grades_path = os.path.join(args.data_dir, 'grades.txt')
    speakers_path = os.path.join(args.data_dir, 'speakers.txt')
    required_files = [responses_path, prompts_path, grades_path, speakers_path]
    
    # If generating a test dataset, load the targets
    if args.preprocessing_type == 'test':
        targets_path = os.path.join(args.data_dir, 'targets.txt')
        required_files.append(targets_path)

    # Assert the required files exist
    for path in required_files:
        if not os.path.isfile(path):
            print('File: {} doesn`t exist. Exiting...'.format(path))
            exit()

    # Cache the command:
    if not os.path.isdir(os.path.join(args.destination_dir, 'CMDs')):
        os.makedirs(os.path.join(args.destination_dir, 'CMDs'))
    with open(os.path.join(args.destination_dir, 'CMDs/preprocessing.cmd'), 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    # Load responses and prompts as sequences of word ids
    responses, _ = load_text(responses_path, args.input_wlist_path, strip_start_end=False)
    prompts, _ = load_text(prompts_path, args.input_wlist_path, strip_start_end=False)


    # Load up the speakers and grades
    with open(grades_path, 'r') as file:
        grades = np.asarray([line.replace('\n', '') for line in file.readlines()])
    with open(speakers_path, 'r') as file:
        speakers = np.asarray([line.replace('\n', '') for line in file.readlines()])

    # If creating test data, load targets
    if args.preprocessing_type == 'test':
        targets = np.loadtxt(args.input_tgt_path, dtype=np.float32)
    
    # Create or load the topic ID dictionary:
    if args.sorted_topics_path == '':
        # Generate and save new topic ID mapping
        topic_dict = generate_topic_dict(prompts_path, args.destination_dir)
    else:
        # Load a pre-existing topic ID mapping
        topic_dict = load_topic_dict(args.sorted_topics_path)


    # Create a list of topic IDs for every response
    with open(prompts_path, 'r') as file:
        # Use id -1 if topic not in topic_dict (not present in training set)
        q_ids = np.asarray([topic_dict.get(line.replace('\n', ''), -1) for line in file.readlines()])

    if args.preprocessing_type == 'train':
        ### Split data into train and validation data sets
        n = len(responses)
        train_size = int(n * (1.0 - args.valid_fraction))
        valid_size = n - train_size

        print('Total dataset size', n, 'Train dataset size', train_size, 'Valid dataset size', valid_size)

        np.random.seed(args.rand_seed)

        permutation = np.random.choice(np.arange(n), n, replace=False)
        index_train = permutation[:train_size]
        index_valid = permutation[train_size:]

        trn_responses = responses[index_train]
        trn_prompts = prompts[index_train]
        trn_q_ids = q_ids[index_train]
        trn_speakers = speakers[index_train]
        trn_grades = grades[index_train]

        valid_responses = responses[index_valid]
        valid_prompts = prompts[index_valid]
        valid_q_ids = q_ids[index_valid]
        valid_speakers = speakers[index_valid]
        valid_grades = grades[index_valid]

        # Create the training TF Record file
        write_to_tfrecords('relevance.train.tfrecords', args.destination_dir, trn_responses, trn_prompts, trn_q_ids, trn_grades, trn_speakers, targets=1.0)

        # Create the validation TF Record file
        write_to_tfrecords('relevance.valid.tfrecords', args.destination_dir, valid_responses, valid_prompts, valid_q_ids, valid_grades, valid_speakers, targets=1.0)

    elif args.preprocessing_type == 'test':
        write_to_tfrecords('relevance.test.tfrecords', args.destination_dir, responses, prompts, q_ids, grades, speakers, targets=targets)

    print('Finished')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

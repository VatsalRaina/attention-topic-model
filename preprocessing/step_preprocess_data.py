#! /usr/bin/env python


import argparse
import os
import sys

import numpy as np
import tensorflow as tf

import core.utilities.tfrecord_utils as tfrecord_utils
from core.utilities.utilities import load_text

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('input_data_path', type=str,
                               help='absolute path to response data')
commandLineParser.add_argument('input_prompt_path', type=str,
                               help='absolute path to prompt data')
commandLineParser.add_argument('input_grade_path', type=str,
                               help='absolute path to ordered list of unique prompts')
commandLineParser.add_argument('input_spkr_path', type=str,
                               help='absolute path to ordered list of unique prompts')
commandLineParser.add_argument('input_wlist_path', type=str,
                               help='absolute path to input word list')
commandLineParser.add_argument('valid_fraction', type=float,
                               help='fraction of full data to reserve for validation')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path location wheree to setup ')

#Need to add stuff about splitting train and eval data!!!
def main(argv=None):
    """Converts a dataset to tfrecords."""
    args = commandLineParser.parse_args()

    if os.path.isdir(args.destination_dir):
        print 'destination directory exists. Exiting...'
    else:
        os.mkdir(args.destination_dir)

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/step_process_relevance_data.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    os.chdir(args.destination_dir)

    #Load responses - check out load_text files...
    responses, _ = load_text(args.input_data_path, args.input_wlist_path)
    prompts, _ = load_text(args.input_prompt_path, args.input_wlist_path)


    #Load up the prompts file
    with open(args.input_prompt_path, 'r') as file:
        topics = [line.replace('\n', '') for line in file.readlines()]

    # Get unique set of topics and topic counts (and sort tem)
    unique_topics, topic_counts = np.unique(topics, return_counts=True)
    topics = unique_topics[np.flip(np.argsort(topic_counts), 0)]
    topic_counts = np.flip(np.sort(topic_counts), 0)

    #Create dictionary for topics mapping sentence to topic id
    # Also create file of sorted topics and unigrams file
    # Unigram file later used for training
    topic_dict = {}
    with open('unigrams.txt', 'w') as ufile:
        with open('sorted_topics.txt', 'w') as tfile:
            for i, topic, count in zip(xrange(topics.shape[0]), topics, topic_counts):
                topic_dict[topic] = i
                ufile.write(str(i) + ',' + str(int(count)) + '\n')
                tfile.write(topic + '\n')

    # Load up the speakers and speakers
    grades = np.loadtxt(args.input_grade_path)
    with open(args.input_spkr_path, 'r') as file:
        speakers = [line.replace('\n', '') for line in file.readlines()]

    # Create a list of topic IDs for every response
    with open(args.input_prompt_path, 'r') as file:
        q_ids = [topic_dict[line.replace('\n', '')] for line in file.readlines()]

    ### Split data into train and validation  data sets
    size = len(responses)
    valid_size = int(size * args.valid_fraction)
    train_size = size - valid_size

    print 'Total dataset size', size, 'Train dataset size', train_size, 'Valid dataset size', valid_size

    trn_responses = responses[valid_size:]
    trn_prompts = prompts[valid_size:]
    trn_q_ids = q_ids[valid_size:]
    trn_speakers = speakers[valid_size:]
    trn_grades = grades[valid_size:]

    valid_responses = responses[:valid_size]
    valid_prompts = prompts[:valid_size]
    valid_q_ids = q_ids[:valid_size]
    valid_speakers = speakers[:valid_size]
    valid_grades = grades[:valid_size]

    # Create the training TF Record file
    ### Maybe want to split into multiple files???
    filename = 'relevance.train.tfrecords'
    print 'Writing', filename
    writer = tf.python_io.TFRecordWriter(filename)
    for response, prompt, q_id, grd, spkr in zip(trn_responses, trn_prompts, trn_q_ids, trn_grades, trn_speakers):
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'targets': tfrecord_utils.float_feature([1.0]),
                'grade': tfrecord_utils.float_feature([grd]),
                'spkr': tfrecord_utils.bytes_feature([spkr]),
                'q_id': tfrecord_utils.int64_feature([q_id])  # ,
                # 'section': tfrecord_utils.int64_feature(section),
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'response': tfrecord_utils.int64_feature_list(response),
                'prompt': tfrecord_utils.int64_feature_list(prompt)}))
        writer.write(example.SerializeToString())
    writer.close()

    # Create the validation TF Record file
    filename = 'relevance.valid.tfrecords'
    print 'Writing', filename
    writer = tf.python_io.TFRecordWriter(filename)
    for response, prompt, q_id, grd, spkr in zip(valid_responses, valid_prompts, valid_q_ids, valid_grades,
                                                 valid_speakers):
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'targets': tfrecord_utils.float_feature([1.0]),
                'grade': tfrecord_utils.float_feature([grd]),
                'spkr': tfrecord_utils.bytes_feature([spkr]),
                'q_id': tfrecord_utils.int64_feature([q_id])  # ,
                # 'section': tfrecord_utils.int64_feature(section),
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'response': tfrecord_utils.int64_feature_list(response),
                'prompt': tfrecord_utils.int64_feature_list(prompt)}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()

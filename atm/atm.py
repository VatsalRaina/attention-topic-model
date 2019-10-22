from __future__ import print_function
import os
import time

import matplotlib
import numpy as np
import math

matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score as roc
import scipy
from scipy.special import loggamma

import tensorflow as tf
import tensorflow.contrib.slim as slim

import context
from core.basemodel import BaseModel
import core.utilities.utilities as util
import core.utilities.bert.tokenization as tokenization
import core.utilities.bert.modeling as modeling
from core.utilities.utilities import IdToWordConverter
from core.utilities.utilities import text_to_array_bert


class AttentionTopicModel(BaseModel):
    def __init__(self, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0, seed=100,
                 epoch=None):

        BaseModel.__init__(self, network_architecture=network_architecture, seed=seed, name=name, save_path=save_path,
                           load_path=load_path, debug_mode=debug_mode)

        with self._graph.as_default():
            with tf.variable_scope('input') as scope:
                self._input_scope = scope
                self.x_a = tf.placeholder(tf.int32, [None, None])
                self.x_q = tf.placeholder(tf.float32, [None, None])
                self.qlens = tf.placeholder(tf.int32, [None])
                self.alens = tf.placeholder(tf.int32, [None])
                self.y = tf.placeholder(tf.float32, [None, 1])
                self.maxlen = tf.placeholder(dtype=tf.int32, shape=[])

                self.dropout = tf.placeholder(tf.float32, [])
                self.batch_size = tf.placeholder(tf.int32, [])

            with tf.variable_scope('atm') as scope:
                self._model_scope = scope
                self._predictions, \
                self._probabilities, \
                self._logits, \
                self.attention, = self._construct_network(a_input=self.x_a,
                                                          a_seqlens=self.alens,
                                                          q_input=self.x_q,
                                                          q_seqlens=self.qlens,
                                                          n_samples=0,
                                                          maxlen=self.maxlen,
                                                          batch_size=self.batch_size,
                                                          keep_prob=self.dropout)

            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if load_path == None:
            with self._graph.as_default():
                init = tf.global_variables_initializer()
                self.sess.run(init)

                # If necessary, restore model from previous
        elif load_path != None:
            self.load(load_path=load_path, step=epoch)


    def _construct_network(self, a_input, a_seqlens, n_samples, q_input, q_seqlens, maxlen, batch_size, keep_prob=1.0):
        """

        :param a_input:
        :param a_seqlens:
        :param n_samples: Number of samples - used to repeat the response encoder output for the resampled prompt
        examples
        :param q_input:
        :param q_seqlens:
        :param maxlen:
        :param batch_size: The batch size before sampling!
        :param keep_prob:
        :return: predictions, probabilities, logits, attention
        """

        L2 = self.network_architecture['L2']
        initializer = self.network_architecture['initializer']

        # Question Encoder RNN
        with tf.variable_scope('Embeddings', initializer=initializer(self._seed)) as scope:
            embedding = slim.model_variable('word_embedding',
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')
            a_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, a_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 1)

            a_inputs_fw = tf.transpose(a_inputs, [1, 0, 2])
            a_inputs_bw = tf.transpose(tf.reverse_sequence(a_inputs, seq_lengths=a_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

            question_embeddings = tf.convert_to_tensor(q_input, tf.float32)  
        """
        with tf.variable_scope('Activation', initializer=initializer(self._seed)) as scope:
            # Apply cascading activation functions
            m1 = tf.get_variable('qout1', shape=[768,768], regularizer=slim.l2_regularizer(L2), trainable=True)
            qout1 = tf.nn.relu6(tf.einsum('ik,jk->ij', question_embeddings, m1))
            m2 = tf.get_variable('qout2', shape=[768,768], regularizer=slim.l2_regularizer(L2), trainable=True)
            qout2 = tf.nn.relu6(tf.einsum('ik,jk->ij', qout1, m2))
            m3 = tf.get_variable('qout3', shape=[768,768], regularizer=slim.l2_regularizer(L2), trainable=True)
            qout3 = tf.nn.relu6(tf.einsum('ik,jk->ij', qout2, m3))
        """

        #    qout1 = slim.fully_connected(question_embeddings,
         #                                num_outputs=768,
         #                                activation_fn=tf.nn.relu6,
         #                                weights_regularizer=slim.l2_regularizer(L2),
         #                                scope="qout1")
         #   qout2 = slim.fully_connected(qout1,
         #                                num_outputs=768,
         #                                activation_fn=tf.nn.relu6,
         #                                weights_regularizer=slim.l2_regularizer(L2))
         #   qout3 = slim.fully_connected(qout2,
         #                                num_outputs=768,
         #                                activation_fn=tf.nn.relu6,
         #                                weights_regularizer=slim.l2_regularizer(L2))
            
        # Response Encoder RNN
        with tf.variable_scope('RNN_A_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_fw, _ = rnn_fw(a_inputs_fw, sequence_length=a_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_A_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_bw, _ = rnn_bw(a_inputs_bw, sequence_length=a_seqlens, dtype=tf.float32)

        outputs = tf.concat([outputs_fw, outputs_bw], axis=2)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob, seed=self._seed)

        a_seqlens = tf.tile(a_seqlens, [n_samples + 1])
        outputs = tf.tile(outputs, [1 + n_samples, 1, 1])

        hidden, attention = self._bahdanau_attention(memory=outputs, seq_lens=a_seqlens, maxlen=maxlen,
                                                     query=question_embeddings,
                                                     size=2 * self.network_architecture['n_rhid'],
                                                     batch_size=batch_size * (n_samples + 1),
                                                     bert_size=768)

        with tf.variable_scope('Grader') as scope:
            for layer in xrange(self.network_architecture['n_flayers']):
                hidden = slim.fully_connected(hidden,
                                              self.network_architecture['n_fhid'],
                                              activation_fn=self.network_architecture['f_activation_fn'],
                                              weights_regularizer=slim.l2_regularizer(L2),
                                              scope="hidden_layer_" + str(layer))
                hidden = tf.nn.dropout(hidden, keep_prob=keep_prob, seed=self._seed + layer)

            logits = slim.fully_connected(hidden,
                                          self.network_architecture['n_out'],
                                          activation_fn=None,
                                          scope="output_layer")
            probabilities = self.network_architecture['output_fn'](logits)
            predictions = tf.cast(tf.round(probabilities), dtype=tf.float32)

        return predictions, probabilities, logits, attention

    def _construct_prompt_encoder(self, p_input, p_seqlens):
        """ Construct RNNLM network
        Args:
          ?
        Returns:
          predictions, probabilities, logits, attention
        """

        L2 = self.network_architecture['L2']
        initializer = self.network_architecture['initializer']

        # Question Encoder RNN
        with tf.variable_scope('Embeddings', initializer=initializer(self._seed)) as scope:
            embedding = slim.model_variable('word_embedding',
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')

            p_inputs = tf.nn.embedding_lookup(embedding, p_input, name='embedded_data')

            p_inputs_fw = tf.transpose(p_inputs, [1, 0, 2])
            p_inputs_bw = tf.transpose(tf.reverse_sequence(p_inputs, seq_lengths=p_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

        # Prompt Encoder RNN
        with tf.variable_scope('RNN_Q_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_fw = rnn_fw(p_inputs_fw, sequence_length=p_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_Q_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_bw = rnn_bw(p_inputs_bw, sequence_length=p_seqlens, dtype=tf.float32)

            prompt_embeddings = tf.concat([state_fw[1], state_bw[1]], axis=1)

        return prompt_embeddings

    def fit(self,
            train_data,
            valid_data,
            load_path,
            topics,
            topic_lens,
            unigram_path,
            train_size=100,
            valid_size=100,
            learning_rate=1e-2,
            lr_decay=0.8,
            dropout=1.0,
            batch_size=50,
            distortion=1.0,
            optimizer=tf.train.AdamOptimizer,
            optimizer_params={},
            n_epochs=30,
            n_samples=1,  # Number of negative samples to generate per positive sample
            epoch=1):
        with self._graph.as_default():
            # Compute number of training examples and batch size
            n_examples = train_size * (1 + n_samples)
            n_batches = n_examples / (batch_size * (1 + n_samples))

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                # Construct training data queues
                targets, \
                q_ids, \
                responses, \
                response_lengths, _, _ = self._construct_dataset_from_tfrecord([train_data],
                                                                               self._parse_func,
                                                                               self._map_func,
                                                                               self._batch_func,
                                                                               batch_size,
                                                                               train=True,
                                                                               capacity_mul=1000,
                                                                               num_threads=8)

                valid_iterator = self._construct_dataset_from_tfrecord([valid_data],
                                                                       self._parse_func,
                                                                       self._map_func,
                                                                       self._batch_func,
                                                                       batch_size,
                                                                       train=False,
                                                                       capacity_mul=100,
                                                                       num_threads=1)
                valid_targets, \
                valid_q_ids, \
                valid_responses, \
                valid_response_lengths, _, _ = valid_iterator.get_next(name='valid_data')

                targets, q_ids = self._sampling_function(targets=targets,
                                                         q_ids=q_ids,
                                                         unigram_path=unigram_path,
                                                         batch_size=batch_size,
                                                         n_samples=n_samples,
                                                         name='train',
                                                         distortion=distortion)
                valid_targets, valid_q_ids = self._sampling_function(targets=valid_targets,
                                                                     q_ids=valid_q_ids,
                                                                     unigram_path=unigram_path,
                                                                     batch_size=batch_size,
                                                                     n_samples=n_samples,
                                                                     name='valid',
                                                                     distortion=1.0)

            topics = tf.convert_to_tensor(topics, dtype=tf.float32)
           # topic_lens = tf.convert_to_tensor(topic_lens, dtype=tf.int32)

            prompts = tf.nn.embedding_lookup(topics, q_ids, name='train_prompot_loopkup')
           # prompt_lens = tf.gather(topic_lens, q_ids)

            valid_prompts = tf.nn.embedding_lookup(topics, valid_q_ids, name='valid_prompot_loopkup')
           # valid_prompt_lens = tf.gather(topic_lens, valid_q_ids)

            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits, _, = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lengths,
                                                         n_samples=n_samples,
                                                         q_input=prompts,
                                                         q_seqlens=None,
                                                         maxlen=tf.reduce_max(response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=self.dropout)

                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lengths,
                                                          n_samples=n_samples,
                                                          q_input=valid_prompts,
                                                          q_seqlens=None,
                                                          maxlen=tf.reduce_max(valid_response_lengths),
                                                          batch_size=batch_size,
                                                          keep_prob=1.0)

            # Construct XEntropy training costs
            trn_cost, total_loss = self._construct_xent_cost(targets=targets,
                                                             logits=trn_logits,
                                                             pos_weight=float(n_samples),
                                                             is_training=True)
            evl_cost = self._construct_xent_cost(targets=valid_targets,
                                                 logits=valid_logits,
                                                 pos_weight=float(n_samples),
                                                 is_training=False)

            train_op = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            if load_path != None:
                self._load_variables(load_scope='model/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (
                    learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = (
                'Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print("Starting Training!\n-----------------------------")
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                # Run Training Loop
                loss = 0.0
                batch_time = time.time()
                for batch in xrange(n_batches):
                    _, loss_value = self.sess.run([train_op, trn_cost], feed_dict={self.dropout: dropout})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss += loss_value

                duration = time.time() - batch_time
                loss /= n_batches
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)
                print(loss_value)

                # Run Validation Loop
                eval_loss = 0.0
                valid_probs = None
                vld_targets = None
                total_size = 0
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        batch_eval_loss, \
                        batch_valid_preds, \
                        batch_valid_probs, \
                        batch_attention, \
                        batch_valid_targets = self.sess.run([evl_cost,
                                                             valid_predictions,
                                                             valid_probabilities,
                                                             valid_attention,
                                                             valid_targets])
                        size = batch_valid_probs.shape[0]
                        eval_loss += float(size) * batch_eval_loss
                        if valid_probs is None:
                            valid_probs = batch_valid_probs
                            vld_targets = batch_valid_targets
                        else:
                            valid_probs = np.concatenate((valid_probs, batch_valid_probs), axis=0)
                            vld_targets = np.concatenate((vld_targets, batch_valid_targets), axis=0)
                        total_size += size
                    except:  # tf.errors.OutOfRangeError:
                        break

                eval_loss = eval_loss / float(total_size)
                roc_score = roc(np.squeeze(vld_targets), np.squeeze(valid_probs))

                # Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            # Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print(format_str % (duration))
            self.save()

    def predict(self, test_pattern, batch_size=20, cache_inputs=False, apply_bucketing=True, topics=None):
        """
        Run inference on a trained model on a dataset.
        :param test_pattern: filepath to dataset to run inference/evaluation on
        :param batch_size: int
        :param cache_inputs: Whether to save the response, prompts, response lengths, and prompt lengths in
        text form together with the predictions. Useful, since bucketing changes the order of the files and this allows
        to investigate which prediction corresponds to which prompt/response pair
        :param apply_bucketing: bool, whether to apply bucketing, i.e. group examples by their response length to
        minimise the overhead associated with zero-padding. If False, the examples will be evaluated in the original
        order as read from the file.

        :return: Depends on whether the inputs are being cached. If cache_inputs=False:
        returns test_loss, test_probabilities_array, test_true_labels_array
        If cache_inputs=True:
        returns test_loss, test_probabilities_array, test_true_labels_array, test_response_lengths,
                test_prompt_lengths, test_responses_list, test_prompts_list
        """
        with self._graph.as_default():
            test_files = tf.gfile.Glob(test_pattern)
            if apply_bucketing:
                batching_function = self._batch_func
            else:
                batching_function = self._batch_func_without_bucket
           # n_examples = batch_size * 2
            #n_batches = n_examples / (batch_size * 2)
            test_iterator = self._construct_dataset_from_tfrecord(test_files,
                                                                  self._parse_func,
                                                                  self._map_func,
                                                                  batching_function,
                                                                  batch_size=batch_size,
                                                                  train=False,
                                                                  capacity_mul=100,
                                                                  num_threads=1)
            test_targets, \
            test_q_ids, \
            test_responses, \
            test_response_lengths, _, _ = test_iterator.get_next(name='valid_data')

            topics = tf.convert_to_tensor(topics, dtype=tf.float32)
            """
            # Change number of topics for unseen evaluation dataset
            self.network_architecture['n_topics'] = 56
            unigram_path = '/home/alta/relevance/vr311/data_vatsal/BULATS/tfrecords_train/unigrams.txt'
            test_targets, test_q_ids = self._sampling_function(targets=test_targets,
                                                     q_ids=test_q_ids,
                                                     unigram_path=unigram_path,
                                                     batch_size=batch_size,
                                                     n_samples=1,
                                                     name='test',
                                                     distortion=1.0)

            """
            test_prompts_bertified = tf.nn.embedding_lookup(topics, test_q_ids, name='test_prompt_loopkup')
             
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                test_predictions, \
                test_probabilities, \
                test_logits, \
                test_attention = self._construct_network(a_input=test_responses,
                                                         a_seqlens=test_response_lengths,
                                                         n_samples=0,
                                                         q_input=test_prompts_bertified,
                                                         q_seqlens=None,
                                                         maxlen=tf.reduce_max(test_response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=1.0)

            loss = self._construct_xent_cost(targets=test_targets, logits=tf.squeeze(test_logits), pos_weight=1.0,
                                             is_training=False)
            self.sess.run(test_iterator.initializer)
            if cache_inputs:
                return self._predict_loop_with_caching(loss, test_probabilities, test_targets,
                                                       test_responses, test_response_lengths, test_prompts,
                                                       test_prompt_lens)
            else:
                return self._predict_loop(loss, test_probabilities, test_targets, test_q_ids)

    def _predict_loop_with_caching(self, loss, test_probabilities, test_targets, test_responses, test_response_lengths,
                                   test_prompts, test_prompt_lens):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        test_responses_list = []
        test_prompts_list = []
        while True:
            try:
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_targets, \
                batch_responses, \
                batch_response_lengths, \
                batch_prompts, \
                batch_prompt_lens = self.sess.run([loss,
                                                   test_probabilities,
                                                   test_targets,
                                                   test_responses,
                                                   test_response_lengths,
                                                   test_prompts,
                                                   test_prompt_lens])

                size = batch_test_probs.shape[0]
                test_loss += float(size) * batch_eval_loss
                if count == 0:
                    test_probs_arr = batch_test_probs  # shape: (num_batches, 1)
                    test_labels_arr = batch_test_targets[:, np.newaxis]  # becomes shape: (num_batches, 1)
                    test_response_lens_arr = batch_response_lengths[:, np.newaxis]  # becomes shape: (num_batches, 1)
                    test_prompt_lens_arr = batch_prompt_lens[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)
                    test_response_lens_arr = np.concatenate(
                        (test_response_lens_arr, batch_response_lengths[:, np.newaxis]), axis=0)
                    test_prompt_lens_arr = np.concatenate((test_prompt_lens_arr, batch_prompt_lens[:, np.newaxis]),
                                                          axis=0)
                test_responses_list.extend(list(batch_responses))  # List of numpy arrays!
                test_prompts_list.extend(list(batch_prompts))  # List of numpy arrays!

                total_size += size
                count += 1
            except:  # todo: tf.errors.OutOfRangeError:
                break

        test_loss = test_loss / float(total_size)

        return (test_loss,
                test_probs_arr,
                test_labels_arr.astype(np.int32),
                test_response_lens_arr.astype(np.int32),
                test_prompt_lens_arr.astype(np.int32),
                test_responses_list,
                test_prompts_list)

    def _predict_loop(self, loss, test_probabilities, test_targets, test_q_ids):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        while True:
            try:
                print("Got to a")
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_targets, batch_test_q_ids = self.sess.run([loss,
                                                    test_probabilities,
                                                    test_targets, test_q_ids])

                print("Got to b")
                size = batch_test_probs.shape[0]
                test_loss += float(size) * batch_eval_loss
                #print(batch_test_targets)
                #print(batch_test_probs)
                print(batch_test_q_ids)
                if count == 0:
                    test_probs_arr = batch_test_probs  # shape: (num_batches, 1)
                    test_labels_arr = batch_test_targets[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)

                total_size += size
                count += 1
            except:  # todo: tf.errors.OutOfRangeError:
                break

        test_loss = test_loss / float(total_size)

        return (test_loss,
                test_probs_arr,
                test_labels_arr.astype(np.int32))

        # def rank(self, X, topics, name=None):
        #     with self._graph.as_default():
        #         test_probs = None
        #         batch_size = len(topics[1])
        #         test_probs = np.zeros(shape=(len(X[0]), batch_size), dtype=np.float32)
        #         for i in xrange(len(X[1])):
        #             if i % 10 == 0: print i
        #             batch_test_probs = self.sess.run(self._probabilities,
        #                                              feed_dict={self.x_a: np.asarray([X[0][i]] * batch_size),
        #                                                         self.alens: np.asarray([X[1][i]] * batch_size),
        #                                                         self.q_ids: np.arange(batch_size),
        #                                                         self.x_q: topics[0],
        #                                                         self.qlens: topics[1],
        #                                                         self.maxlen: np.max(X[1]),
        #                                                         self.batch_size: batch_size})
        #             test_probs[i, :] = np.squeeze(batch_test_probs)
        #         np.savetxt(name + '_probabilities_topics.txt', test_probs)
        #         test_probs = np.reshape(test_probs, newshape=(batch_size * len(X[0])))
        #         hist = np.histogram(test_probs, bins=100, range=[0.0, 1.0], density=True)
        #
        #         plt.plot(hist[0])
        #         plt.xticks(np.arange(0, 101, 20), [str(i / 100.0) for i in xrange(0, 101, 20)])
        #         plt.ylim(0, 50)
        #         plt.ylabel('Density')
        #         plt.xlabel('Relevance Probability')
        #         plt.title('Empirical PDF of Relevance Probabilities')
        #         # plt.show()
        #         plt.savefig('histogram_LINSKneg02.png')
        #         plt.close()

    def get_prompt_embeddings(self, prompts, prompt_lens, save_path):
        with self._graph.as_default():
            prompts = tf.convert_to_tensor(prompts, dtype=tf.int32)
            prompt_lens = tf.convert_to_tensor(prompt_lens, dtype=tf.int32)

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                prompt_embeddings = self._construct_prompt_encoder(p_input=prompts, p_seqlens=prompt_lens)

            embeddings = self.sess.run(prompt_embeddings)

            path = os.path.join(save_path, 'prompt_embeddings.txt')
            np.savetxt(path, embeddings)




class AttentionTopicModelStudent(AttentionTopicModel):
    def __init__(self, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0, seed=100,
                 epoch=None, num_teachers=None):

        AttentionTopicModel.__init__(self, network_architecture=network_architecture, name=name, save_path=save_path,
                                     load_path=load_path, debug_mode=debug_mode, seed=seed, epoch=epoch)

        self.num_teachers = num_teachers

    def _parse_func(self, example_proto):
        contexts, features = tf.parse_single_sequence_example(
            serialized=example_proto,
            context_features={"targets": tf.FixedLenFeature([], tf.float32),
                              "grade": tf.FixedLenFeature([], tf.float32),
                              "spkr": tf.FixedLenFeature([], tf.string),
                              "q_id": tf.FixedLenFeature([], tf.int64),
                              "teacher_pred": tf.FixedLenFeature([self.num_teachers], tf.float32)},
            sequence_features={'response': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                               'prompt': tf.FixedLenSequenceFeature([], dtype=tf.int64)})

        return contexts['targets'], contexts['teacher_pred'], contexts['q_id'], features['response'], features['prompt']

    def _map_func(self, dataset, num_threads, capacity, augment=None):
        dataset = dataset.map(lambda targets, teacher_preds, q_id, resp, prompt: (targets,
                                                                                  teacher_preds,
                                                                                  tf.cast(q_id, dtype=tf.int32),
                                                                                  tf.cast(resp, dtype=tf.int32),
                                                                                  tf.cast(prompt, dtype=tf.int32)),
                              num_parallel_calls=num_threads).prefetch(capacity)

        return dataset.map(lambda targets, teacher_preds, q_id, resp, prompt: (
            targets, teacher_preds, q_id, resp, tf.size(resp), prompt, tf.size(prompt)),
                           num_parallel_calls=num_threads).prefetch(capacity)

    def _batch_func(self, dataset, batch_size, num_buckets=10, bucket_width=10):
        # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
        def batching_func(x):
            return x.padded_batch(
                batch_size,
                # The first three entries are the source and target line rows;
                # these have unknown-length vectors.  The last two entries are
                # the source and target row sizes; these are scalars.
                padded_shapes=(
                    tf.TensorShape([]),  # targets -- unused
                    tf.TensorShape([self.num_teachers]),  # predictions -- unused
                    tf.TensorShape([]),  # q_id -- unused
                    tf.TensorShape([None]),  # resp
                    tf.TensorShape([]),  # resp len -- unused
                    tf.TensorShape([None]),  # prompt
                    tf.TensorShape([])),  # prompt len -- unused
                padding_values=(
                    0.0,  # targets -- unused
                    0.0,  # ensemble predictions - unused
                    np.int32(0),  # q_id -- unused
                    np.int32(0),  # resp
                    np.int32(0),  # resp len -- unused
                    np.int32(0),  # resp len -- unused
                    np.int32(
                        0)))

        def key_func(unused_1, unused_2, unused_3, unused_4, resp_len, unused_5, unused_6):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.

            # Bucket sentence pairs by the length of their response
            bucket_id = resp_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func,
                                                                        reduce_func=reduce_func,
                                                                        window_size=batch_size))

        return batched_dataset

    def _construct_sample_kl_div_student_cost(self, teacher_predictions, logits, is_training=False, num_teachers=10):
        print('Constructing XENT cost')
        targets = tf.reshape(teacher_predictions, [-1, 1])
        logits_all = tf.reshape(tf.tile(logits, [1, num_teachers]), [-1, 1])
        cost = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=logits_all, targets=targets, pos_weight=1.,
                                                     name='total_xentropy_per_batch'))

        if self._debug_mode > 1:
            tf.scalar_summary('XENT', cost)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def fit_student(self,
                    train_data,
                    valid_data,
                    load_path,
                    topics,
                    topic_lens,
                    unigram_path,
                    train_size=100,
                    valid_size=100,
                    learning_rate=1e-2,
                    lr_decay=0.8,
                    dropout=1.0,
                    batch_size=50,
                    distortion=1.0,
                    optimizer=tf.train.AdamOptimizer,
                    optimizer_params={},
                    n_epochs=30,
                    epoch=1,
                    use_teacher_stat=False):
        """
        Custom training procedure for knowledge distillation from teacher

        :param train_data:
        :param valid_data:
        :param load_path:
        :param topics:
        :param topic_lens:
        :param unigram_path:
        :param train_size:
        :param valid_size:
        :param learning_rate:
        :param lr_decay:
        :param dropout:
        :param batch_size:
        :param distortion:
        :param optimizer:
        :param optimizer_params:
        :param n_epochs:
        :param epoch:
        :param use_teacher_stat: bool - whether the cost function is between the average of teacher predictions and output
        or between each individual teacher's model prediction and output
        :return:
        """
        # todo: match sample does not work yet
        with self._graph.as_default():
            # Compute number of training examples and batch size
            n_examples = train_size
            n_batches = n_examples / batch_size

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                # Construct training data queues
                targets, \
                teacher_predictions, \
                q_ids, \
                responses, \
                response_lengths, _, _ = self._construct_dataset_from_tfrecord([train_data],
                                                                               self._parse_func,
                                                                               self._map_func,
                                                                               self._batch_func,
                                                                               batch_size,
                                                                               train=True,
                                                                               capacity_mul=1000,
                                                                               num_threads=8)

                valid_iterator = self._construct_dataset_from_tfrecord([valid_data],
                                                                       self._parse_func,
                                                                       self._map_func,
                                                                       self._batch_func,
                                                                       batch_size,
                                                                       train=False,
                                                                       capacity_mul=100,
                                                                       num_threads=1)
                valid_targets, \
                valid_teacher_predictions, \
                valid_q_ids, \
                valid_responses, \
                valid_response_lengths, _, _ = valid_iterator.get_next(name='valid_data')

                # Expand the dims of targets (normally done in the _sampling_function
                valid_targets = tf.expand_dims(valid_targets, axis=1)
                targets = tf.expand_dims(targets, axis=1)

                # targets, q_ids = self._sampling_function(targets=targets,
                #                                          q_ids=q_ids,
                #                                          unigram_path=unigram_path,
                #                                          batch_size=batch_size,
                #                                          n_samples=n_samples,
                #                                          name='train',
                #                                          distortion=distortion)
                # valid_targets, valid_q_ids = self._sampling_function(targets=valid_targets,
                #                                                      q_ids=valid_q_ids,
                #                                                      unigram_path=unigram_path,
                #                                                      batch_size=batch_size,
                #                                                      n_samples=n_samples,
                #                                                      name='valid',
                #                                                      distortion=1.0)

            # Preprocess the teacher predictions to create targets

            # Calculate student targets as the mean teacher ensemble prediction
            print(teacher_predictions.shape)
            trn_teacher_targets = tf.reduce_mean(teacher_predictions, axis=1, keep_dims=True)
            valid_teacher_targets = tf.reduce_mean(valid_teacher_predictions, axis=1, keep_dims=True)

            topics = tf.convert_to_tensor(topics, dtype=tf.int32)
            topic_lens = tf.convert_to_tensor(topic_lens, dtype=tf.int32)

            prompts = tf.nn.embedding_lookup(topics, q_ids, name='train_prompt_loopkup')
            prompt_lens = tf.gather(topic_lens, q_ids)

            valid_prompts = tf.nn.embedding_lookup(topics, valid_q_ids, name='valid_prompt_loopkup')
            valid_prompt_lens = tf.gather(topic_lens, valid_q_ids)

            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits, _, = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lengths,
                                                         n_samples=0,
                                                         q_input=prompts,
                                                         q_seqlens=prompt_lens,
                                                         maxlen=tf.reduce_max(response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=self.dropout)

                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lengths,
                                                          n_samples=0,
                                                          q_input=valid_prompts,
                                                          q_seqlens=valid_prompt_lens,
                                                          maxlen=tf.reduce_max(valid_response_lengths),
                                                          batch_size=batch_size,
                                                          keep_prob=1.0)

            # Construct XEntropy training costs
            if not use_teacher_stat:
                # Match the individual teacher predictions
                trn_cost, total_loss = self._construct_sample_kl_div_student_cost(
                    teacher_predictions=teacher_predictions,
                    logits=trn_logits,
                    is_training=True)
            else:
                # Match the teacher statistic (average in this case)
                trn_cost, total_loss = self._construct_xent_cost(targets=trn_teacher_targets,
                                                                 logits=trn_logits,
                                                                 pos_weight=1.,
                                                                 is_training=True)
            evl_cost = self._construct_xent_cost(targets=valid_targets,
                                                 logits=valid_logits,
                                                 pos_weight=1.,
                                                 is_training=False)

            train_op = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            if load_path != None:
                self._load_variables(load_scope='model/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (
                    learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = (
                'Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print("Starting Training!\n-----------------------------")
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                # Run Training Loop
                loss = 0.0
                batch_time = time.time()
                for batch in xrange(n_batches):
                    _, loss_value = self.sess.run([train_op, trn_cost], feed_dict={self.dropout: dropout})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss += loss_value

                duration = time.time() - batch_time
                loss /= n_batches
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

                # Run Validation Loop
                eval_loss = 0.0
                valid_probs = None
                vld_targets = None
                total_size = 0
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        batch_eval_loss, \
                        batch_valid_preds, \
                        batch_valid_probs, \
                        batch_attention, \
                        batch_valid_targets = self.sess.run([evl_cost,
                                                             valid_predictions,
                                                             valid_probabilities,
                                                             valid_attention,
                                                             valid_targets])
                        size = batch_valid_probs.shape[0]
                        eval_loss += float(size) * batch_eval_loss
                        if valid_probs is None:
                            valid_probs = batch_valid_probs
                            vld_targets = batch_valid_targets
                        else:
                            valid_probs = np.concatenate((valid_probs, batch_valid_probs), axis=0)
                            vld_targets = np.concatenate((vld_targets, batch_valid_targets), axis=0)
                        total_size += size
                    except:  # tf.errors.OutOfRangeError:
                        break

                eval_loss = eval_loss / float(total_size)
                roc_score = roc(np.squeeze(vld_targets), np.squeeze(valid_probs))

                # Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            # Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print(format_str % (duration))
            self.save()


class ATMPriorNetworkStudent(AttentionTopicModelStudent):
    def __init__(self, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0, seed=100,
                 epoch=None, num_teachers=None):
        AttentionTopicModelStudent.__init__(self, network_architecture=network_architecture, name=name,
                                            save_path=save_path, load_path=load_path, debug_mode=debug_mode, seed=seed,
                                            epoch=epoch, num_teachers=num_teachers)

    def _construct_nll_loss_under_dirichlet(self, teacher_predictions, logits, is_training=False):
        """Negative Log Likelihood (NLL) cost for a binary classification under Dirichlet prior"""
        print('Constructing NLL cost under Dirichlet prior')
        # Clip the predictions for numerical stability
        # teacher_predictions = tf.clip_by_value(teacher_predictions, clip_value_min=(0.0 + self.epsilon),
        #                                        clip_value_max=(1.0 - self.epsilon))

        concentration_params = tf.exp(logits)
        alpha1, alpha2 = tf.split(concentration_params, num_or_size_splits=2,
                                  axis=1)  # todo: write in the alternative way if doesn't work
        # The first column (alpha1) corresponds to P_relevant, and the 2nd column (alpha2) corresponds to P_off_topic.

        log_likelihood_const_part = tf.lgamma(alpha1 + alpha2) - tf.lgamma(alpha1) - tf.lgamma(alpha2)
        log_likelihood_var_part = tf.log(teacher_predictions) * (alpha1 - 1.0) + tf.log(1.0 - teacher_predictions) * (
            alpha2 - 1.0)
        log_likelihood = log_likelihood_const_part + log_likelihood_var_part

        nll_loss = -1.0 * tf.reduce_mean(log_likelihood, axis=1)  # Take the mean over individual ensemble predictions
        nll_cost = tf.reduce_mean(nll_loss)  # Take mean batch-wise (over the number of examples)

        if self._debug_mode > 1:
            tf.scalar_summary('nll_cost', nll_cost)

        if is_training:
            tf.add_to_collection('losses', nll_cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return nll_cost, total_cost
        else:
            return nll_cost

    def _construct_kl_loss_between_dirichlets(self, teacher_alphas, logits, is_training=False):
        """KL loss for binary classification under Dirichlet prior, calculated between Dirichlet
        parametrised by the model and the max-likelihood Dirichlet given observations (teacher predictions)"""
        print('Constructing KL loss between max-likelihood and model Dirichlet')

        model_alphas = tf.exp(logits)

        # The first column (alpha1) corresponds to P_relevant, and the 2nd column (alpha2) corresponds to P_off_topic.
        kl_divergence = tf.lgamma(tf.reduce_sum(teacher_alphas, axis=1)) - tf.lgamma(
            tf.reduce_sum(model_alphas, axis=1)) - tf.reduce_sum(tf.lgamma(teacher_alphas), axis=1) + tf.reduce_sum(
            tf.lgamma(model_alphas), axis=1) + tf.reduce_sum(
            (teacher_alphas - model_alphas) * (
            tf.digamma(teacher_alphas) - tf.digamma(tf.reduce_sum(teacher_alphas, axis=1, keep_dims=True))),
            axis=1)

        kl_cost = tf.reduce_mean(kl_divergence)  # Take mean batch-wise (over the number of examples)

        if self._debug_mode > 1:
            tf.scalar_summary('kl_cost', kl_cost)

        if is_training:
            tf.add_to_collection('losses', kl_cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return kl_cost, total_cost
        else:
            return kl_cost

    def _map_func_with_fit(self, dataset, num_threads, capacity, augment=None):
        """Map function that also fits the Dirichlet. Hopefully should improve training speed"""
        dataset = dataset.map(lambda targets, teacher_preds, q_id, resp, prompt: (targets,
                                                                                  teacher_preds,
                                                                                  tf.cast(q_id, dtype=tf.int32),
                                                                                  tf.cast(resp, dtype=tf.int32),
                                                                                  tf.cast(prompt, dtype=tf.int32)),
                              num_parallel_calls=num_threads).prefetch(capacity)

        return dataset.map(lambda targets, teacher_preds, q_id, resp, prompt: (
            targets, teacher_preds, q_id, resp, tf.size(resp), prompt, tf.size(prompt), tf.py_func(self._fit_dirichlet,
                                                                                             [teacher_preds],
                                                                                             tf.float32,
                                                                                             name='fit_maxl_dirich')),
                           num_parallel_calls=num_threads).prefetch(capacity)

    def _batch_func_with_fit(self, dataset, batch_size, num_buckets=10, bucket_width=10):
        # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
        def batching_func(x):
            return x.padded_batch(
                batch_size,
                # The first three entries are the source and target line rows;
                # these have unknown-length vectors.  The last two entries are
                # the source and target row sizes; these are scalars.
                padded_shapes=(
                    tf.TensorShape([]),  # targets -- unused
                    tf.TensorShape([self.num_teachers]),  # predictions -- unused
                    tf.TensorShape([]),  # q_id -- unused
                    tf.TensorShape([None]),  # resp
                    tf.TensorShape([]),  # resp len -- unused
                    tf.TensorShape([None]),  # prompt
                    tf.TensorShape([]),  # prompt len -- unused
                    tf.TensorShape([2])),  # Dirichlet parameters (alphas) -- unused
                padding_values=(
                    0.0,  # targets -- unused
                    0.0,  # ensemble predictions - unused
                    np.int32(0),  # q_id -- unused
                    np.int32(0),  # resp
                    np.int32(0),  # resp len -- unused
                    np.int32(0),  # prompt
                    np.int32(0),  # prompt len -- unused
                    0.0)  # Dirichlet parameters (alphas) -- unused
            )

        def key_func(unused_1, unused_2, unused_3, unused_4, resp_len, unused_5, unused_6, unused_7):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.

            # Bucket sentence pairs by the length of their response
            bucket_id = resp_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func,
                                                                        reduce_func=reduce_func,
                                                                        window_size=batch_size))

        return batched_dataset

    def _construct_softmax_xent_cost(self, labels, logits, is_training=False):
        print('Constructing XENT cost')
        xent_cost = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='total_xentropy')
        cost = tf.reduce_mean(xent_cost, name='total_xentropy_per_batch')

        if self._debug_mode > 1:
            tf.scalar_summary('XENT', cost)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def fit_student(self,
                    train_data,
                    valid_data,
                    load_path,
                    topics,
                    topic_lens,
                    unigram_path,
                    train_size=100,
                    valid_size=100,
                    learning_rate=1e-2,
                    lr_decay=0.8,
                    dropout=1.0,
                    batch_size=50,
                    distortion=1.0,
                    optimizer=tf.train.AdamOptimizer,
                    optimizer_params={},
                    n_epochs=30,
                    epoch=1,
                    use_teacher_stat=False):
        """Custom fit student. Minimises the negative log likelihood of the teacher model predictions under the
        parameterisation of the Dirichlet given by the outputs of the prior network."""
        with self._graph.as_default():
            # Compute number of training examples and batch size
            n_examples = train_size  # todo:
            n_batches = n_examples / batch_size

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                # Construct training data queues
                if use_teacher_stat:
                    targets, \
                    teacher_predictions, \
                    alphas, \
                    q_ids, \
                    responses, \
                    response_lengths, _, _ = self._construct_dataset_from_tfrecord([train_data],
                                                                   self._parse_func_with_dirich,
                                                                   self._map_func_with_dirich,
                                                                   self._batch_func_with_dirich,
                                                                   batch_size,
                                                                   train=True,
                                                                   capacity_mul=1000,
                                                                   num_threads=8)
                else:
                    targets, \
                    teacher_predictions, \
                    q_ids, \
                    responses, \
                    response_lengths, _, _ = self._construct_dataset_from_tfrecord([train_data],
                                                                                   self._parse_func,
                                                                                   self._map_func,
                                                                                   self._batch_func,
                                                                                   batch_size,
                                                                                   train=True,
                                                                                   capacity_mul=1000,
                                                                                   num_threads=8)


                valid_iterator = self._construct_dataset_from_tfrecord([valid_data],
                                                                       self._parse_func,
                                                                       self._map_func,
                                                                       self._batch_func,
                                                                       batch_size,
                                                                       train=False,
                                                                       capacity_mul=100,
                                                                       num_threads=1)
                valid_targets, \
                valid_teacher_predictions, \
                valid_q_ids, \
                valid_responses, \
                valid_response_lengths, _, _ = valid_iterator.get_next(name='valid_data')

                # Expand the dims of targets (normally done in the _sampling_function
                valid_targets = tf.expand_dims(valid_targets, axis=1)
                targets = tf.expand_dims(targets, axis=1)
                # Turn into softmax compatible format (class distribution instead of P_relevant)
                valid_targets = tf.concat([valid_targets, 1.0 - valid_targets], axis=1)
                targets = tf.concat([targets, 1.0 - targets], axis=1)

            topics = tf.convert_to_tensor(topics, dtype=tf.int32)
            topic_lens = tf.convert_to_tensor(topic_lens, dtype=tf.int32)

            prompts = tf.nn.embedding_lookup(topics, q_ids, name='train_prompt_loopkup')
            prompt_lens = tf.gather(topic_lens, q_ids)

            valid_prompts = tf.nn.embedding_lookup(topics, valid_q_ids, name='valid_prompt_loopkup')
            valid_prompt_lens = tf.gather(topic_lens, valid_q_ids)

            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits, _, = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lengths,
                                                         n_samples=0,
                                                         q_input=prompts,
                                                         q_seqlens=prompt_lens,
                                                         maxlen=tf.reduce_max(response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=self.dropout)

                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lengths,
                                                          n_samples=0,
                                                          q_input=valid_prompts,
                                                          q_seqlens=valid_prompt_lens,
                                                          maxlen=tf.reduce_max(valid_response_lengths),
                                                          batch_size=batch_size,
                                                          keep_prob=1.0)

            if use_teacher_stat:
                # Calculate the parameters of the max-likelihood Dirichlet
                # alphas = tf.py_func(self._fit_dirichlet, [teacher_predictions], tf.float32,
                #                     name='fit_max_likelihood_dirichlet')
                # Construct KL training cost
                trn_cost, total_loss = self._construct_kl_loss_between_dirichlets(teacher_alphas=alphas,
                                                                                  logits=trn_logits,
                                                                                  is_training=True)
            else:
                # Construct XEntropy training costs
                trn_cost, total_loss = self._construct_nll_loss_under_dirichlet(teacher_predictions=teacher_predictions,
                                                                                logits=trn_logits,
                                                                                is_training=True)
            evl_cost = self._construct_softmax_xent_cost(labels=valid_targets,
                                                         logits=valid_logits,
                                                         is_training=False)

            train_op = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            if load_path != None:
                self._load_variables(load_scope='model/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (
                    learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = (
                'Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print("Starting Training!\n-----------------------------")
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                # Run Training Loop
                loss = 0.0
                batch_time = time.time()
                for batch in xrange(n_batches):
                    _, loss_value = self.sess.run([train_op, trn_cost], feed_dict={self.dropout: dropout})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss += loss_value

                duration = time.time() - batch_time
                loss /= n_batches
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

                # Run Validation Loop
                eval_loss = 0.0
                valid_probs = None
                vld_targets = None
                total_size = 0
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        batch_eval_loss, \
                        batch_valid_preds, \
                        batch_valid_probs, \
                        batch_attention, \
                        batch_valid_targets = self.sess.run([evl_cost,
                                                             valid_predictions,
                                                             valid_probabilities,
                                                             valid_attention,
                                                             valid_targets])
                        size = batch_valid_probs.shape[0]
                        eval_loss += float(size) * batch_eval_loss
                        if valid_probs is None:
                            valid_probs = batch_valid_probs
                            vld_targets = batch_valid_targets
                        else:
                            valid_probs = np.concatenate((valid_probs, batch_valid_probs), axis=0)
                            vld_targets = np.concatenate((vld_targets, batch_valid_targets), axis=0)
                        total_size += size
                    except:  # tf.errors.OutOfRangeError:
                        break

                eval_loss = eval_loss / float(total_size)
                roc_score = roc(np.squeeze(vld_targets), np.squeeze(valid_probs))

                # Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            # Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print(format_str % (duration))
            self.save()

    def predict(self, test_pattern, batch_size=20, cache_inputs=False, apply_bucketing=True):
        """
        Custom predict for the prior network. A different predict needed because the prior network outputs two logits
        corresponding to classes off-topic and on-topic instead of a single probability of relevance.

        Run inference on a trained model on a dataset.
        :param test_pattern: filepath to dataset to run inference/evaluation on
        :param batch_size: int
        :param cache_inputs: Whether to save the response, prompts, response lengths, and prompt lengths in
        text form together with the predictions. Useful, since bucketing changes the order of the files and this allows
        to investigate which prediction corresponds to which prompt/response pair
        :param apply_bucketing: bool, whether to apply bucketing, i.e. group examples by their response length to
        minimise the overhead associated with zero-padding. If False, the examples will be evaluated in the original
        order as read from the file.

        :return: Depends on whether the inputs are being cached. If cache_inputs=False:
        returns test_loss, test_probabilities_array, test_true_labels_array
        If cache_inputs=True:
        returns test_loss, test_probabilities_array, test_true_labels_array, test_response_lengths,
                test_prompt_lengths, test_responses_list, test_prompts_list
        """
        with self._graph.as_default():
            test_files = tf.gfile.Glob(test_pattern)
            if apply_bucketing:
                batching_function = self._batch_func_eval
            else:
                batching_function = self._batch_func_without_bucket_eval
            test_iterator = AttentionTopicModel._construct_dataset_from_tfrecord(self,
                                                                                 test_files,
                                                                                 self._parse_func_eval,
                                                                                 self._map_func_eval,
                                                                                 batching_function,
                                                                                 batch_size=batch_size,
                                                                                 train=False,
                                                                                 capacity_mul=100,
                                                                                 num_threads=1)
            test_targets, \
            test_q_ids, \
            test_responses, \
            test_response_lengths, test_prompts, test_prompt_lens = test_iterator.get_next(name='valid_data')

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                test_predictions, \
                test_probabilities, \
                test_logits, \
                test_attention = self._construct_network(a_input=test_responses,
                                                         a_seqlens=test_response_lengths,
                                                         n_samples=0,
                                                         q_input=test_prompts,
                                                         q_seqlens=test_prompt_lens,
                                                         maxlen=tf.reduce_max(test_response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=1.0)

                # Convert the logits and probabilities into the old format (1 value instead of two)
                test_probabilities = test_probabilities[:, 0]
                test_probabilities = tf.expand_dims(test_probabilities, axis=1)

            # todo: Loss is fairly incorrect
            loss = self._construct_xent_cost(targets=test_targets, logits=tf.squeeze(test_logits[:, 0]), pos_weight=1.0,
                                             is_training=False)
            self.sess.run(test_iterator.initializer)

            if cache_inputs:  # todo: this won't work yet
                return self._predict_loop_with_caching(loss, test_probabilities, test_logits, test_targets,
                                                       test_responses, test_response_lengths, test_prompts,
                                                       test_prompt_lens)
            else:
                return self._predict_loop(loss, test_probabilities, test_logits, test_targets)

    def _predict_loop_with_caching(self, loss, test_probabilities, test_logits, test_targets, test_responses, test_response_lengths,
                                   test_prompts, test_prompt_lens):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        test_responses_list = []
        test_prompts_list = []
        while True:
            try:
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_logits, \
                batch_test_targets, \
                batch_responses, \
                batch_response_lengths, \
                batch_prompts, \
                batch_prompt_lens = self.sess.run([loss,
                                                   test_probabilities,
                                                   test_logits,
                                                   test_targets,
                                                   test_responses,
                                                   test_response_lengths,
                                                   test_prompts,
                                                   test_prompt_lens])

                size = batch_test_probs.shape[0]
                test_loss += float(size) * batch_eval_loss
                if count == 0:
                    test_probs_arr = batch_test_probs  # shape: (num_batches, 1)
                    test_logits_arr = batch_test_logits  # shape: (num_batches, 2)
                    test_labels_arr = batch_test_targets[:, np.newaxis]  # becomes shape: (num_batches, 1)
                    test_response_lens_arr = batch_response_lengths[:, np.newaxis]  # becomes shape: (num_batches, 1)
                    test_prompt_lens_arr = batch_prompt_lens[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_logits_arr = np.concatenate((test_logits_arr, batch_test_logits), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)
                    test_response_lens_arr = np.concatenate(
                        (test_response_lens_arr, batch_response_lengths[:, np.newaxis]), axis=0)
                    test_prompt_lens_arr = np.concatenate((test_prompt_lens_arr, batch_prompt_lens[:, np.newaxis]),
                                                          axis=0)
                test_responses_list.extend(list(batch_responses))  # List of numpy arrays!
                test_prompts_list.extend(list(batch_prompts))  # List of numpy arrays!

                total_size += size
                count += 1
            except:  # todo: tf.errors.OutOfRangeError:
                break

        test_loss = test_loss / float(total_size)

        return (test_loss,
                test_probs_arr,
                test_logits_arr,
                test_labels_arr.astype(np.int32),
                test_response_lens_arr.astype(np.int32),
                test_prompt_lens_arr.astype(np.int32),
                test_responses_list,
                test_prompts_list)

    def _predict_loop(self, loss, test_probabilities, test_logits, test_targets):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        while True:
            try:
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_logits, \
                batch_test_targets = self.sess.run([loss,
                                                    test_probabilities,
                                                    test_logits,
                                                    test_targets])

                size = batch_test_probs.shape[0]
                test_loss += float(size) * batch_eval_loss
                if count == 0:
                    test_probs_arr = batch_test_probs  # shape: (num_batches, 1)
                    test_logits_arr = batch_test_logits  # shape: (num_batches, 2)
                    test_labels_arr = batch_test_targets[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_logits_arr = np.concatenate((test_logits_arr, batch_test_logits), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)

                total_size += size
                count += 1
            except:  # todo: tf.errors.OutOfRangeError:
                break

        test_loss = test_loss / float(total_size)

        return (test_loss,
                test_probs_arr,
                test_logits_arr,
                test_labels_arr.astype(np.int32))

    def _parse_func_eval(self, example_proto):
        """Used for data that doesn't come with teacher predictions"""
        return AttentionTopicModel._parse_func(self, example_proto)

    def _map_func_eval(self, dataset, num_threads, capacity, augment=None):
        """Used for data that doesn't come with teacher predictions"""
        return AttentionTopicModel._map_func(self, dataset, num_threads, capacity, augment)

    def _batch_func_eval(self, dataset, batch_size, num_buckets=10, bucket_width=10):
        """Used for data that doesn't come with teacher predictions"""
        return AttentionTopicModel._batch_func(self, dataset, batch_size, num_buckets, bucket_width)

    def _batch_func_without_bucket_eval(self, dataset, batch_size):
        """Same as _batch_func, but doesn't apply bucketing and hence preserves order of the data.
        Used for data that doesn't come with teacher predictions"""
        return AttentionTopicModel._batch_func_without_bucket(self, dataset, batch_size)

    def _fit_dirichlet_batch(self, teacher_predictions):
        log_alphas = np.empty([teacher_predictions.shape[0], 2], dtype=np.float32)
        for i in xrange(teacher_predictions.shape[0]):
            row = teacher_predictions[i]
            log_alphas_row = scipy.optimize.fmin(lambda x: util.nll_exp(x, row), np.array([1., 1.]), disp=False)
            log_alphas[i] = log_alphas_row
        alphas = np.exp(log_alphas).astype(np.float32)
        return alphas

    def _fit_dirichlet(self, teacher_predictions):
        log_alphas = scipy.optimize.fmin(lambda x: util.nll_exp(x, teacher_predictions), np.array([1., 1.]), disp=False)
        alphas = np.exp(log_alphas).astype(np.float32)
        return alphas

    def _parse_func_with_dirich(self, example_proto):
        contexts, features = tf.parse_single_sequence_example(
            serialized=example_proto,
            context_features={"targets": tf.FixedLenFeature([], tf.float32),
                              "grade": tf.FixedLenFeature([], tf.float32),
                              "spkr": tf.FixedLenFeature([], tf.string),
                              "q_id": tf.FixedLenFeature([], tf.int64),
                              "teacher_pred": tf.FixedLenFeature([self.num_teachers], tf.float32),
                              "dirich_params": tf.FixedLenFeature([2], tf.float32)},
            sequence_features={'response': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                               'prompt': tf.FixedLenSequenceFeature([], dtype=tf.int64)})

        return contexts['targets'], contexts['teacher_pred'], contexts['dirich_params'], contexts['q_id'], features[
            'response'], features['prompt']

    def _map_func_with_dirich(self, dataset, num_threads, capacity, augment=None):
        dataset = dataset.map(lambda targets, teacher_preds, dirich, q_id, resp, prompt: (targets,
                                                                                          teacher_preds,
                                                                                          dirich,
                                                                                          tf.cast(q_id, dtype=tf.int32),
                                                                                          tf.cast(resp, dtype=tf.int32),
                                                                                          tf.cast(prompt,
                                                                                                  dtype=tf.int32)),
                              num_parallel_calls=num_threads).prefetch(capacity)

        return dataset.map(lambda targets, teacher_preds, dirich, q_id, resp, prompt: (
            targets, teacher_preds, dirich, q_id, resp, tf.size(resp), prompt, tf.size(prompt)),
                           num_parallel_calls=num_threads).prefetch(capacity)

    def _batch_func_with_dirich(self, dataset, batch_size, num_buckets=10, bucket_width=10):
        # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
        def batching_func(x):
            return x.padded_batch(
                batch_size,
                # The first three entries are the source and target line rows;
                # these have unknown-length vectors.  The last two entries are
                # the source and target row sizes; these are scalars.
                padded_shapes=(
                    tf.TensorShape([]),  # targets -- unused
                    tf.TensorShape([self.num_teachers]),  # predictions -- unused
                    tf.TensorShape([2]),  # Dirichlet params -- unused
                    tf.TensorShape([]),  # q_id -- unused
                    tf.TensorShape([None]),  # resp
                    tf.TensorShape([]),  # resp len -- unused
                    tf.TensorShape([None]),  # prompt
                    tf.TensorShape([])),  # prompt len -- unused
                padding_values=(
                    0.0,  # targets -- unused
                    0.0,  # ensemble predictions - unused
                    0.0,  # Dirichlet params - unused
                    np.int32(0),  # q_id -- unused
                    np.int32(0),  # resp
                    np.int32(0),  # resp len -- unused
                    np.int32(0),  # prompt
                    np.int32(0)))  # prompt len -- unused

        def key_func(unused_1, unused_2, unused_3, unused_4, unused_5, resp_len, unused_6, unused_7):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.

            # Bucket sentence pairs by the length of their response
            bucket_id = resp_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func,
                                                                        reduce_func=reduce_func,
                                                                        window_size=batch_size))
        return batched_dataset


class ATMPriorNetwork(AttentionTopicModel):
    def __init__(self, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0, seed=100,
                 epoch=None):
        AttentionTopicModel.__init__(self, network_architecture=network_architecture, name=name,
                                     save_path=save_path, load_path=load_path, debug_mode=debug_mode, seed=seed,
                                     epoch=epoch)

    def _construct_contrastive_loss(self, logits_in_domain, logits_out_domain, targets_in_domain,
                                    batch_size, in_domain_precision=10., smoothing_val=1e-2, is_training=False,
                                    out_of_domain_weight=1.):
        """
        Contrastive Loss as described in the Prior Net Paper
        :param logits_in_domain: logits for the in_domain batch
        :param logits_out_domain: logits for the out of domain batch
        :param targets_in_domain: targets for the in domain batch
        :param in_domain_precision: float
        :param smoothing_val: float
        :param is_training: bool
        :return: cost ( cost, total_cost tuple if is_training==True)
        """
        print('Constructing contrastive loss')

        model_alphas_in_domain = tf.exp(logits_in_domain)
        model_alphas_out_domain = tf.exp(logits_out_domain)

        # Specify the target alphas and smooth as required
        in_domain_target_alphas = tf.abs(targets_in_domain - smoothing_val) * in_domain_precision
        out_of_domain_target_alphas = tf.ones([batch_size, 2], dtype=tf.float32)

        # The first column (alpha1) corresponds to P_relevant, and the 2nd column (alpha2) corresponds to P_off_topic.

        # Construct the contrastive loss using is_in_domain to mask the appriopriate values
        contr_loss_in_domain = self.kl_divergence_between_dirchlets(in_domain_target_alphas, model_alphas_in_domain)
        contr_loss_out_domain = self.kl_divergence_between_dirchlets(out_of_domain_target_alphas, model_alphas_out_domain)

        cost = tf.reduce_mean(contr_loss_in_domain + out_of_domain_weight * contr_loss_out_domain) / (
        1. + out_of_domain_weight)  # Take mean batch-wise (over the number of examples)

        if self._debug_mode > 1:
            tf.scalar_summary('cost', cost)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def _construct_conflictive_cost(self, logits, targets, batch_size, regularisation_weight=0.1, is_training=False):
        print('Constructing conflictive (XENT + Dir(1, 1)) cost')
        model_alphas = tf.exp(logits)

        xent_cost = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits, name='total_xentropy')

        regularisation_target_alphas = tf.ones([batch_size, 2], dtype=tf.float32)
        regularisation_cost = self.kl_divergence_between_dirchlets(regularisation_target_alphas, model_alphas)

        cost = tf.reduce_mean(xent_cost + regularisation_weight * regularisation_cost, name='total_conflictive_per_batch')

        if self._debug_mode > 1:
            tf.scalar_summary('XENT', cost)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def _construct_contrastive_with_xe_loss(self, logits_in_domain, logits_out_domain, targets_in_domain,
                                            batch_size, in_domain_precision=20., is_training=False,
                                            out_of_domain_kl_weight=1., in_domain_kl_weight=0.5):
        """
        Contrastive Loss as described in the Prior Net Paper
        :param logits_in_domain: logits for the in_domain batch
        :param logits_out_domain: logits for the out of domain batch
        :param targets_in_domain: targets for the in domain batch
        :param in_domain_precision: float
        :param is_training: bool
        :return: cost ( cost, total_cost tuple if is_training==True)
        """
        print('Constructing contrastive loss with XE in domain')

        model_alphas_in_domain = tf.exp(logits_in_domain)
        model_alphas_out_domain = tf.exp(logits_out_domain)

        # Specify the target alphas and smooth as required
        xe_targets = targets_in_domain[:, 0]  # Shape should be [batch_size, 1]

        in_domain_target_alphas = tf.ones([batch_size, 2], dtype=tf.float32) * (in_domain_precision / 2.)
        out_of_domain_target_alphas = tf.ones([batch_size, 2], dtype=tf.float32)

        # The first column (alpha1) corresponds to P_relevant, and the 2nd column (alpha2) corresponds to P_off_topic.

        # Construct the contrastive loss using is_in_domain to mask the appriopriate values
        contr_loss_in_domain = self.kl_divergence_between_dirchlets(in_domain_target_alphas, model_alphas_in_domain)
        contr_loss_out_domain = self.kl_divergence_between_dirchlets(out_of_domain_target_alphas,
                                                                     model_alphas_out_domain)

        xe_in_domain = tf.nn.weighted_cross_entropy_with_logits(logits=logits_in_domain[:, 0], targets=xe_targets,
                                                                pos_weight=1., name='total_xentropy_per_batch')

        cost = tf.reduce_mean(
            xe_in_domain + in_domain_kl_weight * contr_loss_in_domain + out_of_domain_kl_weight * contr_loss_out_domain) / (
               1. + out_of_domain_kl_weight + in_domain_kl_weight)  # Take mean batch-wise (over the number of examples)

        if self._debug_mode > 1:
            tf.scalar_summary('cost', cost)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def _construct_contrastive_with_nll_loss(self, logits_in_domain, logits_out_domain, targets_in_domain,
                                             batch_size, in_domain_precision=10.,
                                             is_training=False,
                                             out_of_domain_weight=1.):
        """
        Contrastive Loss as described in the Prior Net Paper
        :param logits_in_domain: logits for the in_domain batch
        :param logits_out_domain: logits for the out of domain batch
        :param targets_in_domain: targets for the in domain batch
        :param in_domain_precision: float
        :param smoothing_val: float
        :param is_training: bool
        :return: cost ( cost, total_cost tuple if is_training==True)
        """
        print('Constructing contrastive loss')

        model_alphas_in_domain = tf.exp(logits_in_domain)
        model_alphas_out_domain = tf.exp(logits_out_domain)

        log_likelihood_const_part = tf.lgamma(tf.reduce_sum(model_alphas_in_domain, axis=1)) - tf.reduce_sum(
            tf.lgamma(model_alphas_in_domain), axis=1)
        log_likelihood_var_part = tf.reduce_sum(tf.log(targets_in_domain) * (model_alphas_in_domain - 1.), axis=1)
        log_likelihood = log_likelihood_const_part + log_likelihood_var_part

        nll_loss = -1.0 * tf.reduce_mean(log_likelihood, axis=1)  # Take the mean over individual ensemble predictions

        # Specify the target alphas and smooth as required
        out_of_domain_target_alphas = tf.ones([batch_size, 2], dtype=tf.float32)

        # The first column (alpha1) corresponds to P_relevant, and the 2nd column (alpha2) corresponds to P_off_topic.

        # Construct the contrastive loss using is_in_domain to mask the appriopriate values
        contr_loss_in_domain = nll_loss
        contr_loss_out_domain = self.kl_divergence_between_dirchlets(model_alphas_out_domain,
                                                                     out_of_domain_target_alphas)

        cost = tf.reduce_mean(contr_loss_in_domain + out_of_domain_weight * contr_loss_out_domain) / (
            1. + out_of_domain_weight)  # Take mean batch-wise (over the number of examples)

        if self._debug_mode > 1:
            tf.scalar_summary('cost', cost)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def kl_divergence_between_dirchlets(self, alpha_params1, alpha_params2):
        kl_divergence = tf.lgamma(tf.reduce_sum(alpha_params2, axis=1)) - tf.lgamma(
            tf.reduce_sum(alpha_params1, axis=1)) - tf.reduce_sum(tf.lgamma(alpha_params2), axis=1) + tf.reduce_sum(
            tf.lgamma(alpha_params1), axis=1) + tf.reduce_sum(
            (alpha_params2 - alpha_params1) * (
                tf.digamma(alpha_params2) - tf.digamma(tf.reduce_sum(alpha_params2, axis=1, keep_dims=True))),
            axis=1)
        return kl_divergence

    def _construct_softmax_xent_cost(self, labels, logits, is_training=False):
        print('Constructing XENT cost')
        xent_cost = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='total_xentropy')
        cost = tf.reduce_mean(xent_cost, name='total_xentropy_per_batch')

        if self._debug_mode > 1:
            tf.scalar_summary('XENT', cost)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost

    def _construct_inputs(self, data, batch_size, unigram_path=None, topics=None, topic_lens=None, train=True,
                          capacity_mul=800,
                          num_threads=1, distortion=1.0, name='input', sample=False):
        if train:
            targets, \
            q_ids, \
            responses, \
            response_lengths, prompts, prompt_lengths = self._construct_dataset_from_tfrecord([data],
                                                                           self._parse_func,
                                                                           self._map_func,
                                                                           self._batch_func,
                                                                           batch_size,
                                                                           train=True,
                                                                           capacity_mul=capacity_mul,
                                                                           num_threads=num_threads)
            iterator = None
        else:
            iterator = self._construct_dataset_from_tfrecord([data],
                                                             self._parse_func,
                                                             self._map_func,
                                                             self._batch_func,
                                                             batch_size,
                                                             train=False,
                                                             capacity_mul=capacity_mul,
                                                             num_threads=num_threads)
            targets, \
            q_ids, \
            responses, \
            response_lengths, prompts, prompt_lengths = iterator.get_next(name=name + '_data')

        if sample:
            assert topics is not None
            assert topic_lens is not None
            assert unigram_path is not None
            targets, q_ids = self._sampling_function(targets=targets,
                                                     q_ids=q_ids,
                                                     unigram_path=unigram_path,
                                                     batch_size=batch_size,
                                                     n_samples=1,
                                                     name=name + '_samples',
                                                     distortion=distortion)

            topics = tf.convert_to_tensor(topics, dtype=tf.int32)
            topic_lens = tf.convert_to_tensor(topic_lens, dtype=tf.int32)
            prompts = tf.nn.embedding_lookup(topics, q_ids, name=name + '_prompt_loopkup')
            prompt_lengths = tf.gather(topic_lens, q_ids)
        else:
            targets = tf.expand_dims(targets, axis=1)

        targets = tf.concat([targets, 1.0 - targets], axis=1)

        return targets, prompts, prompt_lengths, responses, response_lengths, iterator

    def fit_with_ood(self,
                      train_data_in_domain,
                      train_data_out_domain,
                      valid_data,
                      load_path,
                      topics_in_domain,
                      topic_lens_in_domain,
                      unigram_path_in_domain,
                      train_size=100,
                      valid_size=100,
                      learning_rate=1e-2,
                      lr_decay=0.8,
                      dropout=1.0,
                      presample_batch_size=50,
                      distortion=1.0,
                      optimizer=tf.train.AdamOptimizer,
                      optimizer_params={},
                      n_epochs=30,
                      n_samples=1,  # Number of negative samples to generate per positive sample
                      epoch=1,
                      which_trn_cost='contrastive',
                      out_of_domain_weight=1.):
        """Custom fit function for a prior network. """
        with self._graph.as_default():
            batch_size = ((n_samples + 1) * presample_batch_size)  # Batch size after sampling
            # Compute number of training examples and batch size
            n_examples = train_size
            n_batches = n_examples / batch_size

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:

                # Construct training data queues
                targets_in_domain, \
                prompts_in_domain, \
                prompt_lens_in_domain, \
                responses_in_domain, \
                response_lens_in_domain, _ = self._construct_inputs(train_data_in_domain,
                                                                    presample_batch_size,
                                                                    unigram_path=unigram_path_in_domain,
                                                                    topics=topics_in_domain,
                                                                    topic_lens=topic_lens_in_domain,
                                                                    train=True, capacity_mul=1000,
                                                                    num_threads=6, distortion=1.0,
                                                                    sample=True,
                                                                    name='train_in_domain')

                targets_out_domain, \
                prompts_out_domain, \
                prompt_lens_out_domain, \
                responses_out_domain, \
                response_lens_out_domain, _ = self._construct_inputs(train_data_out_domain,
                                                                     batch_size,  # 2x batch size because no sampling
                                                                     train=True, capacity_mul=1000,
                                                                     num_threads=6, sample=False,
                                                                     distortion=1.0, name='train_out_domain')
                valid_targets, \
                valid_prompts, \
                valid_prompt_lens, \
                valid_responses, \
                valid_response_lens, valid_iterator = self._construct_inputs(valid_data,
                                                                             presample_batch_size,
                                                                             unigram_path=unigram_path_in_domain,
                                                                             topics=topics_in_domain,
                                                                             topic_lens=topic_lens_in_domain,
                                                                             train=False,
                                                                             capacity_mul=100,
                                                                             sample=True,
                                                                             num_threads=1,
                                                                             name='valid')

            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                train_predictions, \
                train_probabilities, \
                train_logits_in_domain, _, = self._construct_network(a_input=responses_in_domain,
                                                                     a_seqlens=response_lens_in_domain,
                                                                     n_samples=n_samples,
                                                                     q_input=prompts_in_domain,
                                                                     q_seqlens=prompt_lens_in_domain,
                                                                     maxlen=tf.reduce_max(response_lens_in_domain),
                                                                     batch_size=presample_batch_size,
                                                                     keep_prob=self.dropout)
                train_predictions_out_domain, \
                train_probabilities_out_domain, \
                train_logits_out_domain, _, = self._construct_network(a_input=responses_out_domain,
                                                                      a_seqlens=response_lens_out_domain,
                                                                      n_samples=0,
                                                                      q_input=prompts_out_domain,
                                                                      q_seqlens=prompt_lens_out_domain,
                                                                      maxlen=tf.reduce_max(response_lens_out_domain),
                                                                      batch_size=batch_size,
                                                                      keep_prob=self.dropout)


                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lens,
                                                          n_samples=n_samples,
                                                          q_input=valid_prompts,
                                                          q_seqlens=valid_prompt_lens,
                                                          maxlen=tf.reduce_max(valid_response_lens),
                                                          batch_size=presample_batch_size,
                                                          keep_prob=1.0)

                # Use mean precision for training logging purposes
                mean_precision_id = tf.reduce_mean(tf.reduce_sum(tf.exp(train_logits_in_domain), axis=1))
                mean_precision_ood = tf.reduce_mean(tf.reduce_sum(tf.exp(train_logits_out_domain), axis=1))

            if which_trn_cost == 'contrastive':
                # Construct the conflictive training cost
                trn_cost, total_loss = self._construct_contrastive_loss(logits_in_domain=train_logits_in_domain,
                                                                        logits_out_domain=train_logits_out_domain,
                                                                        targets_in_domain=targets_in_domain,
                                                                        batch_size=batch_size,
                                                                        out_of_domain_weight=out_of_domain_weight,
                                                                        is_training=True)
            elif which_trn_cost == 'contrastive_with_xe' or which_trn_cost == 'contraceptive':
                # Construct the contraceptive training cost
                trn_cost, total_loss = self._construct_contrastive_with_xe_loss(logits_in_domain=train_logits_in_domain,
                                                                                logits_out_domain=train_logits_out_domain,
                                                                                targets_in_domain=targets_in_domain,
                                                                                batch_size=batch_size,
                                                                                out_of_domain_kl_weight=out_of_domain_weight,
                                                                                in_domain_kl_weight=0.1,
                                                                                is_training=True)
            else:
                raise AttributeError('{} is not a valid training cost for the ATM Prior Network'.format(which_trn_cost))

            evl_cost = self._construct_softmax_xent_cost(labels=valid_targets,
                                                         logits=valid_logits,
                                                         is_training=False)

            train_op = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            if load_path != None:
                self._load_variables(load_scope='model/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (
                    learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = (
                'Epoch %d, Train Loss = %.2f, ID Precision = %.2f, OOD Precision = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print("Starting Training!\n-----------------------------")
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                # Run Training Loop
                loss_total = 0.
                precision_id_total = 0.
                precision_ood_total = 0.
                batch_time = time.time()
                for batch in xrange(n_batches):
                    _, loss_value, precision_id_value, precision_ood_value = self.sess.run([train_op, trn_cost, mean_precision_id, mean_precision_ood],
                                                  feed_dict={self.dropout: dropout})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss_total += loss_value
                    precision_id_total += precision_id_value
                    precision_ood_total += precision_ood_value

                duration = time.time() - batch_time
                loss_total /= n_batches
                precision_id_total /= n_batches
                precision_ood_total /= n_batches
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

                # Run Validation Loop
                eval_loss = 0.0
                valid_probs = None
                vld_targets = None
                total_size = 0
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        batch_eval_loss, \
                        batch_valid_preds, \
                        batch_valid_probs, \
                        batch_attention, \
                        batch_valid_targets = self.sess.run([evl_cost,
                                                             valid_predictions,
                                                             valid_probabilities,
                                                             valid_attention,
                                                             valid_targets])
                        size = batch_valid_probs.shape[0]
                        eval_loss += float(size) * batch_eval_loss
                        if valid_probs is None:
                            valid_probs = batch_valid_probs
                            vld_targets = batch_valid_targets
                        else:
                            valid_probs = np.concatenate((valid_probs, batch_valid_probs), axis=0)
                            vld_targets = np.concatenate((vld_targets, batch_valid_targets), axis=0)
                        total_size += size
                    except Exception as e:  # tf.errors.OutOfRangeError:
                        # print("The end of validation loop exception:\n------------------\n", e, "\n--------------")
                        break

                eval_loss = eval_loss / float(total_size)
                roc_score = roc(np.squeeze(vld_targets), np.squeeze(valid_probs))

                # Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss_total, precision_id_total, precision_ood_total, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print(format_str % (epoch, loss_total, precision_id_total, precision_ood_total, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            # Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print(format_str % (duration))
            self.save()

    def fit(self,
            train_data,
            valid_data,
            load_path,
            topics,
            topic_lens,
            unigram_path,
            train_size=100,
            valid_size=100,
            learning_rate=1e-2,
            lr_decay=0.8,
            dropout=1.0,
            presample_batch_size=50,
            distortion=1.0,
            optimizer=tf.train.AdamOptimizer,
            optimizer_params={},
            n_epochs=30,
            n_samples=1,  # Number of negative samples to generate per positive sample
            which_trn_cost='conflictive',
            loss_regularisation_weight=0.1,
            epoch=1):
        """The prior net fit without out of distribution data."""
        with self._graph.as_default():
            batch_size = ((n_samples + 1) * presample_batch_size)  # Batch size after sampling
            print("Actual batch size used (post-sampling): {}".format(batch_size))
            # Compute number of training examples and batch size
            n_examples = train_size
            n_batches = n_examples / batch_size

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                # Construct training data queues
                # Construct training data queues
                targets, \
                prompts, \
                prompt_lens, \
                responses, \
                response_lens, _ = self._construct_inputs(train_data,
                                                          presample_batch_size,
                                                          unigram_path=unigram_path,
                                                          topics=topics,
                                                          topic_lens=topic_lens,
                                                          train=True, capacity_mul=1000,
                                                          num_threads=8, distortion=1.0,
                                                          sample=True,
                                                          name='train')

                valid_targets, \
                valid_prompts, \
                valid_prompt_lens, \
                valid_responses, \
                valid_response_lens, valid_iterator = self._construct_inputs(valid_data,
                                                                             presample_batch_size,
                                                                             unigram_path=unigram_path,
                                                                             topics=topics,
                                                                             topic_lens=topic_lens,
                                                                             train=False,
                                                                             capacity_mul=100,
                                                                             sample=True,
                                                                             num_threads=1,
                                                                             name='valid')

            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits, _, = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lens,
                                                         n_samples=n_samples,
                                                         q_input=prompts,
                                                         q_seqlens=prompt_lens,
                                                         maxlen=tf.reduce_max(response_lens),
                                                         batch_size=presample_batch_size,
                                                         keep_prob=self.dropout)

                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lens,
                                                          n_samples=n_samples,
                                                          q_input=valid_prompts,
                                                          q_seqlens=valid_prompt_lens,
                                                          maxlen=tf.reduce_max(valid_response_lens),
                                                          batch_size=presample_batch_size,
                                                          keep_prob=1.0)

            # Construct XEntropy training costs
            if which_trn_cost == 'conflictive':
                # Construct the conflictive training cost
                trn_cost, total_loss = self._construct_conflictive_cost(logits=trn_logits,
                                                                        targets=targets,
                                                                        batch_size=batch_size,
                                                                        regularisation_weight=loss_regularisation_weight,
                                                                        is_training=True)
            else:
                raise AttributeError('{} is not a valid training cost for the ATM Prior Network'.format(which_trn_cost))

            evl_cost = self._construct_softmax_xent_cost(labels=valid_targets,
                                                         logits=valid_logits,
                                                         is_training=False)

            train_op = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            if load_path != None:
                self._load_variables(load_scope='model/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (
                    learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = (
                'Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print("Starting Training!\n-----------------------------")
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                # Run Training Loop
                loss = 0.0
                batch_time = time.time()
                for batch in xrange(n_batches):
                    _, loss_value = self.sess.run([train_op, trn_cost], feed_dict={self.dropout: dropout})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss += loss_value

                duration = time.time() - batch_time
                loss /= n_batches
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

                # Run Validation Loop
                eval_loss = 0.0
                valid_probs = None
                vld_targets = None
                total_size = 0
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        batch_eval_loss, \
                        batch_valid_preds, \
                        batch_valid_probs, \
                        batch_attention, \
                        batch_valid_targets = self.sess.run([evl_cost,
                                                             valid_predictions,
                                                             valid_probabilities,
                                                             valid_attention,
                                                             valid_targets])
                        size = batch_valid_probs.shape[0]
                        eval_loss += float(size) * batch_eval_loss
                        if valid_probs is None:
                            valid_probs = batch_valid_probs
                            vld_targets = batch_valid_targets
                        else:
                            valid_probs = np.concatenate((valid_probs, batch_valid_probs), axis=0)
                            vld_targets = np.concatenate((vld_targets, batch_valid_targets), axis=0)
                        total_size += size
                    except:  # tf.errors.OutOfRangeError:
                        break

                eval_loss = eval_loss / float(total_size)
                roc_score = roc(np.squeeze(vld_targets), np.squeeze(valid_probs))

                # Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            # Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print(format_str % (duration))
            self.save()

    def predict(self, test_pattern, batch_size=20, apply_bucketing=True):
        """
        Custom predict for the prior network. A different predict needed because the prior network outputs two logits
        corresponding to classes off-topic and on-topic instead of a single probability of relevance.

        Run inference on a trained model on a dataset.
        :param test_pattern: filepath to dataset to run inference/evaluation on
        :param batch_size: int
        :param cache_inputs: Whether to save the response, prompts, response lengths, and prompt lengths in
        text form together with the predictions. Useful, since bucketing changes the order of the files and this allows
        to investigate which prediction corresponds to which prompt/response pair
        :param apply_bucketing: bool, whether to apply bucketing, i.e. group examples by their response length to
        minimise the overhead associated with zero-padding. If False, the examples will be evaluated in the original
        order as read from the file.

        :return: Depends on whether the inputs are being cached. If cache_inputs=False:
        returns test_loss, test_probabilities_array, test_true_labels_array
        If cache_inputs=True:
        returns test_loss, test_probabilities_array, test_true_labels_array, test_response_lengths,
                test_prompt_lengths, test_responses_list, test_prompts_list
        """
        with self._graph.as_default():
            test_files = tf.gfile.Glob(test_pattern)
            if apply_bucketing:
                batching_function = self._batch_func
            else:
                batching_function = self._batch_func_without_bucket
            test_iterator = AttentionTopicModel._construct_dataset_from_tfrecord(self,
                                                                                 test_files,
                                                                                 self._parse_func,
                                                                                 self._map_func,
                                                                                 batching_function,
                                                                                 batch_size=batch_size,
                                                                                 train=False,
                                                                                 capacity_mul=100,
                                                                                 num_threads=1)
            test_targets, \
            test_q_ids, \
            test_responses, \
            test_response_lengths, test_prompts, test_prompt_lens = test_iterator.get_next(name='valid_data')

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                test_predictions, \
                test_probabilities, \
                test_logits, \
                test_attention = self._construct_network(a_input=test_responses,
                                                         a_seqlens=test_response_lengths,
                                                         n_samples=0,
                                                         q_input=test_prompts,
                                                         q_seqlens=test_prompt_lens,
                                                         maxlen=tf.reduce_max(test_response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=1.0)

                # Convert the logits and probabilities into the old format (1 value instead of two)
                test_probabilities = test_probabilities[:, 0]
                test_probabilities = tf.expand_dims(test_probabilities, axis=1)

            # todo: Loss is fairly incorrect
            loss = self._construct_xent_cost(targets=test_targets, logits=tf.squeeze(test_logits[:, 0]), pos_weight=1.0,
                                             is_training=False)
            self.sess.run(test_iterator.initializer)

            # if cache_inputs:  # todo: this won't work yet
            #     return self._predict_loop_with_caching(loss, test_probabilities, test_logits, test_targets,
            #                                            test_responses, test_response_lengths, test_prompts,
            #                                            test_prompt_lens)
            return self._predict_loop(loss, test_probabilities, test_logits, test_targets)

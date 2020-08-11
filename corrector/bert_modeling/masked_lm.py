#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Title   : bert prediction
@File    : masked_lm.py
@Author  : Tian
@Time    : 2020/06/16 5:04 下午
@Version : 1.0
"""
import six
import tensorflow as tf
import numpy as np
import warnings
import os
import logging

from corrector.bert_modeling import modeling, tokenization

warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

class MaskedLMConfig(object):
    max_seq_length = 64
    vocab_file = "vocab.txt"
    bert_config_file = "bert_config.json"
    init_checkpoint = "model/pre-trained/bert_model.ckpt"
    char_meta_file = "data/char_meta.txt"
    topn = 3
    batch_size = 2

    @classmethod
    def from_dict(cls, json_object):
        config = MaskedLMConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config


class Model(object):
    def __init__(self, config):
        self.config = config
        self.config.bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)

        # placeholders
        self.input_ids = tf.placeholder(tf.int32, [None, self.config.max_seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, self.config.max_seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, self.config.max_seq_length], name='segment_ids')
        self.masked_lm_positions = tf.placeholder(tf.int32, [None, None],
                                                  name='masked_lm_positions')
        self.masked_lm_ids = tf.placeholder(tf.int32, [None, None],
                                            name='masked_lm_ids')
        self.masked_lm_weights = tf.placeholder(tf.float32, [None, None],
                                                name='masked_lm_weights')

        is_training = False

        # create model
        masked_lm_loss, masked_lm_example_loss, self.masked_lm_log_probs, self.probs = self.create_model(
            self.input_ids,
            self.input_mask,
            self.segment_ids,
            self.masked_lm_positions,
            self.masked_lm_ids,
            self.masked_lm_weights,
            is_training,
            self.config.bert_config)

        # prediction
        self.masked_lm_predictions = tf.argmax(self.masked_lm_log_probs, axis=-1, output_type=tf.int32)
        self.top_n_predictions = tf.nn.top_k(self.probs, k=config.topn, sorted=True, name="topn")

    def predict(self, batch, sess):
        """
        for predicting
        """

        input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, _ = batch

        feed_dict = {
            self.input_ids: input_ids,
            self.input_mask: input_mask,
            self.segment_ids: segment_ids,
            self.masked_lm_positions: masked_lm_positions,
            self.masked_lm_ids: masked_lm_ids,
            self.masked_lm_weights: masked_lm_weights
        }

        masked_lm_predictions, masked_lm_log_probs = sess.run(
            [self.masked_lm_predictions, self.masked_lm_log_probs], feed_dict)

        return masked_lm_predictions

    def topn_predict(self, batch, sess):
        """
        for predicting topn results
        """

        input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = batch.data

        feed_dict = {
            self.input_ids: input_ids,
            self.input_mask: input_mask,
            self.segment_ids: segment_ids,
            self.masked_lm_positions: masked_lm_positions,
            self.masked_lm_ids: masked_lm_ids,
            self.masked_lm_weights: masked_lm_weights
        }

        top_n_predictions = sess.run(self.top_n_predictions, feed_dict)
        topn_probs, topn_predictions = top_n_predictions

        return np.array(topn_probs, dtype=float), topn_predictions

    def create_model(self,
                     input_ids,
                     input_mask,
                     segment_ids,
                     masked_lm_positions,
                     masked_lm_ids,
                     masked_lm_weights,
                     is_training,
                     bert_config):
        """Create Masked Language Model"""


        bert_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        sequence_output = bert_model.get_sequence_output()
        embedding_table = bert_model.get_embedding_table()

        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs, probs = self.get_masked_lm_output(
            bert_config, sequence_output, embedding_table,
            masked_lm_positions, masked_lm_ids, masked_lm_weights)

        return masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs, probs

    @classmethod
    def get_masked_lm_output(cls, bert_config, input_tensor, output_weights, positions,
                             label_ids, label_weights):
        """Get loss and log probs for the masked LM."""
        input_tensor = cls.gather_indexes(input_tensor, positions)

        with tf.variable_scope("cls/predictions",reuse=tf.AUTO_REUSE):
            # Apply one more non-linear transformation to predict the masked token
            # This matrix is for pre-training, and can be used straightly for correction
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
                input_tensor = modeling.layer_norm(input_tensor)

            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)

            label_ids = tf.reshape(label_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])

            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

        return loss, per_example_loss, log_probs, probs

    @staticmethod
    def gather_indexes(sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor


class BatchData(object):
    """A batch of data for BERT"""

    def __init__(self, sentences, error_positions, tokenizer, seq_length=64):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = len(sentences)
        self.batch_error_num = None
        self.error_num_of_sents = None

        self.data = self.create_one_batch_data(sentences, error_positions)


    def create_one_batch_data(self, sentences, errors):
        input_ids_batch, \
        input_mask_batch, \
        segment_ids_batch, \
        masked_lm_positions_batch, \
        masked_lm_ids_batch, \
        masked_lm_weights_batch = [], [], [], [], [], []

        self.error_num_of_sents = [len(e) for e in errors]
        self.batch_error_num = max(self.error_num_of_sents)

        for i, sentence in enumerate(sentences):

            input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = \
                self.create_one_sent_data(sentence, errors[i])
            input_ids_batch.append(input_ids)
            input_mask_batch.append(input_mask)
            segment_ids_batch.append(segment_ids)
            masked_lm_positions_batch.append(masked_lm_positions)
            masked_lm_ids_batch.append(masked_lm_ids)
            masked_lm_weights_batch.append(masked_lm_weights)

        return np.array(input_ids_batch), np.array(input_mask_batch), np.array(segment_ids_batch), \
               np.array(masked_lm_positions_batch), np.array(masked_lm_ids_batch), np.array(masked_lm_weights_batch)


    def create_one_sent_data(self, sentence, errors):
        # tokenization
        tokens_raw = self.tokenizer.tokenize(tokenization.convert_to_unicode(sentence))

        # add [CLS] and [SEP]
        tokens = ["[CLS]", "。"]  + tokens_raw + ["。", "[SEP]"]
        errors = [e + 2 for e in errors]
        segment_ids = [0] * len(tokens)
        # todo: add length check

        # produce pseudo ground truth, since the truth is unknown when it comes to spelling checking.
        input_tokens, masked_lm_positions, masked_lm_labels = self.create_masks(tokens, errors)

        # convert to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(segment_ids)

        while len(input_ids) < self.seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        masked_lm_positions = list(masked_lm_positions)
        masked_lm_ids = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < self.batch_error_num:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights

    @staticmethod
    def create_masks(tokens, errors):
        input_tokens = list(tokens)
        masked_lm_positions = []
        masked_lm_labels = []

        # If only a few chars are marked as errors, use '[MASK]' to replace them will perform better;
        # While it is nonsense to replace a lot of chars.
        for index in errors:
            if len(errors) <= 3:
                masked_token = '[MASK]'
            else:
                masked_token = tokens[index]
            input_tokens[index] = masked_token
            masked_lm_positions.append(index)
            masked_lm_labels.append(tokens[index])

        return input_tokens, masked_lm_positions, masked_lm_labels

    def __repr__(self):
        return 'a batch of {} sentences, with sentence length of {}, ' \
               'error number per sentence of {}'.format(self.batch_size, self.seq_length, self.batch_error_num)


class DataProcessor(object):
    """chunk data into batches"""

    def __init__(self, max_seq_length, batch_size, tokenizer):
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data = None
        self.errors = None
        self.pos = None


    def add_data(self, sentences, error_positions):
        self.data = sentences
        self.errors = self._check_error_positions(error_positions)
        self.pos = 0

    def next(self):
        if self.data is None:
            logging.warning('待处理数据为空，请添加数据')
            return None
        if self.pos >= len(self.data):
            self.pos = 0
            self.data = None
            self.errors = None
            return None
        else:
            batch_data = self.data[self.pos: self.pos + self.batch_size]
            # 如果不指定error position，则认为所有字符都需要纠错
            if self.errors is not None:
                batch_errors = self.errors[self.pos: self.pos + self.batch_size]
            else:
                batch_errors = [range(len(d)) for d in batch_data]

            self.pos += self.batch_size

        return BatchData(batch_data, batch_errors, self.tokenizer, self.max_seq_length)

    def _check_error_positions(self, error_positions):
        if error_positions is None:
            return None
        if len(error_positions) != len(self.data):
            logging.error('提供的 error position 与 data 不匹配，error position 不生效')
            return None
        if not all(error_positions):
            logging.error('error position 有空值，error position 不生效')
            return None
        return error_positions


class MaskedLM(object):
    def __init__(self, config):
        config.vocab_file = os.path.join(os.path.dirname(__file__),
                                         config.vocab_file)
        config.bert_config_file = os.path.join(os.path.dirname(__file__),
                                               config.bert_config_file)
        config.init_checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                               config.init_checkpoint)
        self.config = config

        # create session
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4)
        session_conf.gpu_options.allow_growth = True
        self.session = tf.Session(config=session_conf)

        # load model

        self.model = self.load_model(config)
        self.session.run(tf.global_variables_initializer())

        self.tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file)
        self.data_processor = DataProcessor(config.max_seq_length, config.batch_size, self.tokenizer)

    @staticmethod
    def load_model(config):

        model = Model(config)

        tvars = tf.trainable_variables()

        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, config.init_checkpoint)

        tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)

        return model

    def find_topn_candidates(self, sentences, error_positions=None):
        """
        Args
        -----------------------------
        sentences: a list of sentences, e.g., ['the man went to the store.', 'he bought a gallon of milk.']
        error_positions: positions to be corrected, if None, every char will be corrected
        batch_size: default=1

        Return
        -----------------------------
        candidates for each nominated error in the sentences, e.g.,
        [[[('the', 0.88), ('a', 0.65)], ...], [...]]
        """

        self.data_processor.add_data(sentences, error_positions)
        res_stream = []
        probs_stream = []
        batch_error_num = []
        error_num = []
        while True:
            batch = self.data_processor.next()
            if batch is not None:
                logging.info('开始处理新的batch:%r', batch)
                topn_probs, topn_predictions = self.model.topn_predict(batch, self.session)

                batch_error_num.extend([batch.batch_error_num]*batch.batch_size)
                error_num.extend(batch.error_num_of_sents)
                res_stream.extend(topn_predictions)
                probs_stream.extend(topn_probs)

            else:
                break

        result = []
        start = 0

        for result_num, sent_error_num in zip(batch_error_num, error_num):
            one_sent_result = []

            for e in range(sent_error_num):
                one_char_result = []
                for ids, prob in zip(res_stream[start+e], probs_stream[start+e]):
                    one_char_result.append((self.tokenizer.ids_to_vocab[ids], prob))
                one_sent_result.append(one_char_result)
            result.append(one_sent_result)
            start += result_num

        return result


def test_masked_lm():
    config = MaskedLMConfig()
    lm = MaskedLM(config)
    res = lm.find_topn_candidates(['。国际电台苦名丰持人。', '。我爱北京大安门。'], [[5,7], [5]])

    for sen in res:
        print('长度为', len(sen))
        for char in sen:
            print(char)


if __name__ == '__main__':
    test_masked_lm()
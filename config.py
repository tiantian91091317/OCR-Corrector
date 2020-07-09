#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""全局配置类"""

class KwdsCorrecterConfig:
    """
    关键字纠错的配置
    """
    prob_threshold = 0.9
    similarity_threshold = 0.6
    char_meta_file = ""
    key_words_file = ""


class BertCorrecterConfig:
    prob_threshold = 0.9
    max_seq_length = 64
    topn = 3
    batch_size = 16
    char_meta_file = ""
    vocab_file = ""
    bert_config_file = ""
    init_checkpoint = ""


class CorrecterConfig:
    """
    纠错相关的配置
    """
    report_config = KwdsCorrecterConfig()
    your_config = KwdsCorrecterConfig()
    bert_config = BertCorrecterConfig()

class SystemConfig:
    """
    总配置
    """
    tfserving_port = ""
    tfserving_ip = ""
    correct_config = CorrecterConfig()
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : Tian
@File    : __init__.py
@Time    : 2020/8/3 10:21 PM
@Desc    :
@Version : 1.0
"""
import json
import os

from .bert_corrector import BertCorrector
from .keyword_corrector import KeywordCorrector, KwdCorrectorConfig
from .bert_modeling.masked_lm import MaskedLMConfig
from . import api_call



def initial():
    """
    初始化各纠错器
    :return:
    """

    config_json = os.path.join(os.path.dirname(__file__), 'config/config.json')
    with open(config_json, encoding='utf-8') as f:
        json_data = json.load(f)

    crct = {}
    for crct_cfg in json_data['correct_config']:
        if crct_cfg['corrector_type'] == 'keyword':
            config = KwdCorrectorConfig.from_dict(crct_cfg)
            crct[crct_cfg['biz_type']] = KeywordCorrector(config)
        else:
            config = MaskedLMConfig.from_dict(crct_cfg)
            crct[crct_cfg['biz_type']] = BertCorrector(config)
    return crct


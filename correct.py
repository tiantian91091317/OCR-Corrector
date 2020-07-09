#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
调用纠错服务的示例
"""
import argparse
import json
import logging
import os
from collections import namedtuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

from correcter.bert_correcter import BertCorrecter
from correcter.keyword_correcter import KeywordCorrecter
from utils import json_utils, tfserving_agent
import config

config_json = 'release_1.0.json'


def initial_correcters():
    """
    初始化各参数，及纠错器
    :return:
    """

    conf = config.SystemConfig()

    json_f = open(config_json, encoding='utf-8')
    json_data = json.load(json_f)
    json_utils.dic2class_ignore(json_data, conf)

    # mode：single（单机）模式/tfserving模式
    # 为了和docker容器启动的方式保持一致，从环境变量传入
    mode = os.getenv("MODE")
    if mode is None:
        logger.error("未在启动环境变量中设置启动模式MODE")
        raise ValueError("未在启动环境变量中设置启动模式MODE")

    # BERT方法兼容两种模式
    if mode == 'single':
        model = None
    else:
        Model = namedtuple('Model', ['topn_predict'])
        model = Model(topn_predict = tfserving_agent.correct_tf_serving_call)

    report_correcter = KeywordCorrecter(conf.correct_config.report_config)
    your_correcter = KeywordCorrecter(conf.correct_config.your_config)
    document_correcter = BertCorrecter(conf.correct_config.bert_config, model)

    correcters = {'report': report_correcter,
                  'your_business': your_correcter,
                  'document': document_correcter}

    return correcters

def my_ocr(image):
    """模拟 OCR 预测结果，此处直接读取示例图片的预测结果"""
    name, _ = os.path.splitext(image)
    ocr_result = 'data/' + name + '_ocr_result.json'
    with open(ocr_result, 'r', encoding='utf-8') as f:
        result = json.load(f)
    texts = result['texts']
    probs = result['probs']

    # 模拟图片分类
    business_type = 'report' if name == 'img2' else 'document'

    return texts, probs, business_type


def ocr_and_correct(img, ocr_correcters):
    logger.info('纠错图片【%s】的识别结果', img)
    text_arr, prob_arr, business_type  = my_ocr(img)
    logger.info('业务场景：【%s】', business_type)
    try:
        correcter = ocr_correcters[business_type]
    except KeyError:
        logger.error('错误的 business type: %s', business_type)
        return text_arr
    corrected = correcter.correct(text_arr, prob_arr)
    logger.info('纠错结果：')
    for original, corr in zip(text_arr, corrected):
        if original != corr:
            logger.info('【%s】纠正为【%s】', original,corr)

    return corrected

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img')
    args = parser.parse_args()
    img = args.img

    correcters = initial_correcters()
    ocr_and_correct(img, correcters)

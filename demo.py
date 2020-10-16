#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : Tian
@File    : demo.py
@Time    : 2020/8/3 10:21 PM
@Desc    : 在OCR中调用纠错的示例
@Version : 1.0
"""
import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')


import corrector as ocr_corrector


def my_ocr(image):
    """模拟 OCR 预测过程
    此处直接读取保存好的图片预测结果"""
    name, _ = os.path.splitext(image)
    ocr_result = name + '_ocr_result.json'
    with open(ocr_result, 'r', encoding='utf-8') as f:
        result = json.load(f)
    texts = result['texts']
    probs = result['probs']

    return texts, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img')  # 图片路径
    parser.add_argument('--biz')  # 业务类型
    parser.add_argument('--api')  # 接口名称
    args = parser.parse_args()
    img = args.img
    biz_type = args.biz
    api = args.api

    corrector = ocr_corrector.initial()
    if biz_type not in corrector:
        logger.error('错误的业务类型:%s', biz_type)
        return
    if not os.path.exists(img):
        logger.error('图片不存在，请检查图片路径:%s', img)
        return

    if api == 'own':
        ocr_results, recog_probs = my_ocr(img)
    else:
        third_ocr_api = ocr_corrector.api_call.get_call(api)
        ocr_results, recog_probs = third_ocr_api.ocr_from_path(img)

    if not ocr_results:
        logger.error('识别出现错误')
        return

    ocr_res_corrected = corrector.get(biz_type).correct(ocr_results, recog_probs)

    logger.info('纠错结果：')
    for original, corr in zip(ocr_results, ocr_res_corrected):
        if original != corr:
            logger.info('【%s】纠正为【%s】', original, corr)

if __name__ == '__main__':
    main()


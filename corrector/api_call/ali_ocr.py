#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
阿里云-高精版识别接口调用
接口文档：https://duguang.aliyun.com/document/%E9%AB%98%E7%B2%BE%E7%89%88.html
"""

import requests


def ocr(img):
    url = 'https://ocrapi-advanced.taobao.com/ocrservice/advanced'
    post_data = {"img":img,
                 "prob":True,
                 "charInfo":True
                 }
    app_code = 'your_code'
    headers = {'Authorization': 'APPCODE ' + app_code,
               'Content-Type': 'application/json; charset=UTF-8',
               'Connection': 'close'}

    try:
        res = requests.post(url, headers=headers, json=post_data, verify=False)
        res_data = res.json()
        texts, probs = parse_result(res_data)
        return texts, probs

    except Exception as e:
        import traceback
        traceback.print_stack()
        print(e)
        return None, None


def parse_result(res_data):
    """
    解析接口返回结果
    :param res_data:
    :return:
    """
    words = res_data['prism_wordsInfo']
    texts = []
    probs = []
    for word in words:
        texts.append(word['word'])
        chars = word['charInfo']
        prob = []
        for char in chars:
            prob.append(char['prob']/100)
        probs.append(prob)

    return texts, probs
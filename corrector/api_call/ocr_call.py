#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Title   : 接口调用
@File    : ocr_call.py
@Author  : Tian
@Time    : 2020/06/16 5:04 下午
@Version : 1.0
"""

import logging
import base64
import cv2
from abc import ABCMeta, abstractmethod

from . import ali_ocr


logger = logging.getLogger(__name__)

class BaseCall(metaclass=ABCMeta):
    """
        基于文件路径
    """
    def __init__(self):
      pass

    @abstractmethod
    def ocr(self, img_base64):
        pass

    def ocr_from_path(self, image_path):
        image = self.base64_from_path(image_path)
        return self.ocr(image)

    @staticmethod
    def base64_from_path(image_path):
        image = cv2.imread(image_path)
        base64_str = cv2.imencode('.jpg', image)[1].tostring()
        base64_str = base64.b64encode(base64_str)
        return str(base64_str, "utf-8")


class TencentCall(BaseCall):
    """
    Tencent OCR 识别
    """
    pass

class AliCall(BaseCall):
    """
     阿里巴巴OCR接口调用
    """

    def ocr(self, img):
        return ali_ocr.ocr(img)

class HuaweiCall(BaseCall):
    """
    华为OCR 各个接口
    """
    pass


class FaceCall(BaseCall):
    """
    旷视OCR 各个接口
    """
    pass


class BaiduCall(BaseCall):
    """
    百度OCR 各个接口
    """
    pass


def get_call(call_type):
    if call_type == 'ali':
        return AliCall()
    else:
        logger.error("指定的接口不存在：%s", call_type)
        return None


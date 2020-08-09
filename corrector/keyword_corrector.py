#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Title   : 关键字法纠错
@File    : keyword_corrector.py
@Author  : Tian
@Time    : 2020/06/16 5:04 下午
@Version : 1.0
"""
import logging
import re
import six

from corrector.base_corrector import BaseCorrector, CorrectorConfig
from corrector.utils.BKtree import BKTree

logger = logging.getLogger(__name__)

class KwdCorrectorConfig:
    prob_threshold = 0.9
    similarity_threshold = 0.3
    key_words_file = 'data/kwds_credit_report.txt'
    char_meta_file = 'data/char_meta.txt'

    @classmethod
    def from_dict(cls, json_object):
        config = KwdCorrectorConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

class KeywordCorrector(BaseCorrector):
    """
    关键词表纠错
    """
    def __init__(self, config: KwdCorrectorConfig):
        super().__init__(config=config)
        self.tree = self.build_tree()
        self.key_words = self.load_key_words_dict()


    def correct_all(self, texts, err_positions):
        """
        >>> kc = KeywordCorrector(CorrectorConfig)
        >>> kc.correct_all(['宋还本全', '数期大'],[[0,3],[0,2]])
        ['未还本金', '逾期天']
        """
        results = []
        for text, err in zip(texts, err_positions):
            try:
                if text in self.key_words:
                    results.append(text)
                    logger.debug('【%s】在关键词表中，跳过',text)
                    continue

                # 根据编辑距离找到近似的关键词
                distance = len(err)
                if distance == len(text):
                    distance = len(text) - 1
                logger.debug('纠错【%s】错误位置【%s】', text, err)
                k = self.tree.search(text, distance)   # ['未还本金','已还本金','还本金'，'宋某还本金']
                if not k:
                    results.append(text)
                    logger.debug('未找到编辑距离在【%d】以内的关键词', distance)
                    continue
                logger.debug('找到编辑距离在【%d】以内的关键词【%s】',distance, k)

                # 正则找到词中需要进一步匹配的字
                reg, origin = regulation(text, err)   # '^(.)还本(.)$'
                candidates = []  # ['未金','已金']
                for _k in k:
                    r = re.match(reg, _k)
                    if not r:   #  '还本金'，'宋某还本金'  被过滤
                        continue
                    cnd = ''.join([r.group(i+1) for i in range(len(err))])
                    candidates.append(cnd)     # ['未金','已金']
                if not candidates:
                    results.append(text)
                    logger.debug('未找到结构相同的关键词')
                    continue
                logger.debug('找到结构相同的的关键词，【%s】vs【%r】', origin, candidates)

                # 根据汉字笔画编码找到最终的正确字
                sims = []
                for cnd in candidates:
                    sims.append(self.char_sim.shape_similarity(origin, cnd))  #  [0.58, 0.30]
                if max(sims) < self.config.similarity_threshold:
                    results.append(text)
                    logger.debug('最大笔画相似度【%f】低于阈值【%f】不纠错', max(sims), self.config.similarity_threshold)
                    continue

                # 最终进行替换
                substitution = list(candidates[sims.index(max(sims))])  # ['未', '金']
                _t = list(text)
                for c in err:
                    _t[c] = substitution.pop(0)
                results.append(''.join(_t))
                logger.info('目标【%s】纠正为关键词【%s】，相似度【%f】', text, ''.join(_t), max(sims))

            except Exception:
                results.append(text)
                import traceback
                logger.error(traceback.format_exc())
                logger.error('纠错出现错误，跳过【%s】', text)

        return results

    def build_tree(self):
        tree = BKTree(self.config.key_words_file)
        tree.plant_tree()
        return tree

    def load_key_words_dict(self):
        with open(self.config.key_words_file) as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
        return set(lines)


def regulation(text, errors):
    """
    通过正则表达式，对search到的Node进行进一步过滤，为此生成正则表达式
    :param text:
    :param mask:
    :return:

    Test:
    >>> regulation('宋还本全',[1,0,0,0])
    '^(.)还本(.)$', '宋全'
    >>> regulation('数期大', [1,0,1])
    '^(.)期(.)$', '数大'
    """

    # reg = '^'
    reg = list(text)
    error_chars = ''
    for err in errors:
        reg[err] = '(.)'
        error_chars += text[err]
    reg = '^' + ''.join(reg) + '$'

    return reg, error_chars

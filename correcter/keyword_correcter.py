#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Title   : 关键字法纠错
@File    :   kwyword_correcter.py
@Author  : Tian
@Time    : 2020/06/16 5:04 下午
@Version : 1.0
"""
import logging
import re

from correcter.ocr_correcter import BaseCorrecter, CorrecterConfig
from correcter.utils.BKtree import BKTree

logger = logging.getLogger(__name__)



class KeywordCorrecter(BaseCorrecter):
    """
    关键词表纠错
    """
    def __init__(self, config: CorrecterConfig):
        super().__init__(config=config)
        self.tree = self.build_tree()
        self.key_words = self.load_key_words_dict()


    def correct_all(self, texts, mask):
        """
        >>> kc = KeywordCorrecter(CorrecterConfig)
        >>> kc.correct_all(['宋还本全', '数期大', '余额'],[[1,0,0,1],[1,0,1],[1,0]])
        ['未还本金', '逾期天', '余额']
        """
        # logger.info('需要纠正的文本:%s', texts)
        results = []
        for t, m in zip(texts, mask):
            try:
                if t in self.key_words:
                    results.append(t)
                    logger.debug('【%s】在关键词表中，跳过',t)
                    continue

                # 根据编辑距离找到近似的关键词
                wrong_place = [i for i in range(len(m)) if m[i]]
                distance = len(wrong_place)
                if distance == len(m):
                    distance = len(m) - 1
                logger.debug('纠错【%s】错误位置【%s】', t, wrong_place)
                k = self.tree.search(t, distance)   # ['未还本金','已还本金','还本金'，'宋某还本金']
                if not k:
                    results.append(t)
                    logger.debug('未找到编辑距离在【%d】以内的关键词', distance)
                    continue
                logger.debug('找到编辑距离在【%d】以内的关键词【%s】',distance, k)

                # 正则找到词中需要进一步匹配的字
                reg, origin = regulation(t, m)   # '^(.)还本(.)$'
                candidates = []  # ['未金','已金']
                for _k in k:
                    r = re.match(reg, _k)
                    if not r:   #  '还本金'，'宋某还本金'  被过滤
                        continue
                    cnd = ''.join([r.group(i+1) for i in range(len(wrong_place))])
                    candidates.append(cnd)     # ['未金','已金']
                if not candidates:
                    results.append(t)
                    logger.debug('未找到结构相同的关键词')
                    continue
                logger.debug('找到结构相同的的关键词，【%s】vs【%r】', origin, candidates)

                # 根据汉字笔画编码找到最终的正确字
                sims = []
                for cnd in candidates:
                    sims.append(self.char_sim.shape_similarity(origin, cnd))  #  [0.58, 0.30]
                if max(sims) < self.config.similarity_threshold:
                    results.append(t)
                    logger.debug('最大笔画相似度【%f】低于阈值【%f】不纠错', max(sims), self.config.similarity_threshold)
                    continue

                # 最终进行替换
                substitution = list(candidates[sims.index(max(sims))])  # ['未', '金']
                _t = list(t)
                for c in wrong_place:
                    _t[c] = substitution.pop(0)
                results.append(''.join(_t))
                logger.info('目标【%s】纠正为关键词【%s】，相似度【%f】', t, ''.join(_t), max(sims))

            except Exception:
                results.append(t)
                import traceback
                logger.error(traceback.format_exc())
                logger.error('纠错出现错误，跳过【%s】', t)

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


def regulation(text, mask):
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

    reg = '^'
    wrong_chars = ''
    for t, m in zip(text, mask):
        if m == 1:
            reg += '(.)'
            wrong_chars += t
        else:
            reg += t
    reg += '$'
    return reg, wrong_chars

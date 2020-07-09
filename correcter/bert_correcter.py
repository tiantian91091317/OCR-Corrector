#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Title   : 利用BERT模型进行纠错
@File    :   bert_correcter.py
@Author  : Tian
@Time    : 2020/06/16 5:04 下午
@Version : 1.0
"""
import re
import logging

from correcter.ocr_correcter import BaseCorrecter
from correcter.bert_modified.masked_lm import MaskedLM, MaskedLMConfig

logger = logging.getLogger(__name__)

class BertCorrecter(BaseCorrecter):
    """
    BERT纠错
    """
    def __init__(self, config: MaskedLMConfig, model=None):
        super().__init__(config=config)
        self.bert = MaskedLM(config, model)
        self.accept_correct = Curves.curve_02

    def correct_all(self, texts, mask):
        """
        >>> kc = BertCorrecter(MaskedLMConfig)
        >>> kc.correct_all(['本着平等、白愿、诚信、互利的原则，致同意本合同内容，并共同遵守。','无效、重大暇疵或不符合乙方其他规定的债权资产，'
        ... '乙方有权拒绝，不子初始登'],[[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ... [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        ['本着平等、自愿、诚信、互利的原则，致同意本合同内容，并共同遵守。', '无效、重大瑕疵或不符合乙方其他规定的债权资产，乙方有权拒绝，不予初始登']
        """
        # bert接收阿拉伯数字会出错（返回长度不定的阿拉伯数字）
        rep = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零'}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        number = re.compile("|".join(rep.keys()))
        texts_numfree = [number.sub(lambda m: rep[re.escape(m.group(0))], s) for s in texts]
        # 前后加'。'，同样是为了bert输出正确结果
        texts_processed = ['。' + t + '。' for t in texts_numfree]

        bert_out = self.bert.find_topn_candidates(texts_processed, self.config.batch_size)
        for i in range(len(texts)):
            try:
                error_pos = [k for k in range(len(mask[i])) if mask[i][k]]
                logger.debug('纠正【%s】错误位置【%s】', texts[i], error_pos)
                origin = list(texts[i])
                for e in error_pos:
                    # 不修正数字
                    if self.regulars['number'].match(origin[e]):
                        logger.debug('原字【%s】为数字，不纠错', origin[e])
                        continue

                    for j in range(self.config.topn):  # 0, 1, 2
                        confidence = bert_out[i][e+1][j][1]   # 第i个句子，第e个字，第j个预测结果，第1个元素（confidence）
                        pred = bert_out[i][e+1][j][0]  # 第i个句子，第e个字，第j个预测结果，第0个元素（pred）
                        char_similarity = self.char_sim.shape_similarity(pred, origin[e])
                        logger.debug('原字【%s】bert预测结果：【%s】,confidence：【%f】，char_similarity:【%f】',
                                     origin[e], pred, confidence, char_similarity)

                        # 检查预测结果
                        if origin[e] == pred:
                            continue
                        if not self.check_bert_out(origin[e], pred):   # 处理不能接受接错结果的情况
                            continue
                        if self.accept_correct(confidence, char_similarity):   # confidence 和 similarity 联合判断是否接受
                            logger.debug('※ 接受纠错 ※')
                            origin[e] = pred
                            break
                texts[i] = ''.join(origin)

            except Exception:
                import traceback
                logger.error(traceback.format_exc())
                logger.error('纠错出现错误，跳过【%s】',texts[i])
        return texts

    # bert需要重写是否纠错的过滤
    def do_correct_filter(self, text):
        # 包含字母的不纠错
        if re.search(self.regulars['alphabet'], text):
            return False
        # 包含小于3个汉字的不纠错
        if len(re.findall(self.regulars['chinese'], text)) < 3:
            return False
        # 超过最大长度的暂时跳过，因为这样的情况应该很少，仅防止出错
        if len(text) > self.config.max_seq_length - 2:
            logger.error('句子长度超过最大长度%d，跳过纠错', self.config.max_seq_length - 2)
            return False

        return True

    def check_bert_out(self, original, corrected_to):
        if corrected_to == '[UNK]':
            return False
        if '#' in corrected_to:
            return False
        if len(corrected_to) != len(original):
            return False
        if re.search(self.regulars['alphabet'], corrected_to):
            return False
        # 如果correct_to是繁体字，则不接受纠错
        if re.match(self.regulars['traditional'], corrected_to):
            return False
        return True


    @staticmethod
    def recover_number(sen1, sen2):
        """把句子1中的数字赋值到句子2中，用于恢复数字"""
        new_sen = sen2
        for i, s in enumerate(sen1):
            if u'\u0030' <= s <= u'\u0039':
                new_sen = new_sen[:i] + s + new_sen[i + 1:]

        return new_sen



class Curves(object):
    def __init__(self):
        pass

    @staticmethod
    def curve_null(confidence, similarity):
        """This curve is used when no filter is applied"""
        return True

    @staticmethod
    def curve_full(confidence, similarity):
        """This curve is used to filter out everything"""
        return False

    @staticmethod
    def curve_01(confidence, similarity):
        """
        we provide an example of how to write a curve. Empirically, curves are all convex upwards.
        Thus we can approximate the filtering effect of a curve using its tangent lines.
        """
        flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
        flag2 = 0.1 * confidence + similarity - 0.6 > 0

        if flag1 or flag2:
            return True

        return False

    @staticmethod
    def curve_02(confidence, similarity):  # this one is mine
        # similarity > 0.6 or confidence ~ 0.99
        # flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
        # flag2 = 0.1 * confidence + similarity - 0.67 > 0
        flag1 = confidence + similarity - 1 >= 0
        flag2 = confidence - 0.05 >= 0
        flag3 = similarity - 0.4 >= 0

        if flag1 and flag2 and flag3:
            return True

        return False

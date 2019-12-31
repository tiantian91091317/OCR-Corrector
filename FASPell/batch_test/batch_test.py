# PIL是一个常用的python图像库。在python 3.5中， 应该使用pip install pillow来安装
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import random
import codecs
import logging
import matplotlib
matplotlib.use('TkAgg')
import requests
import json
import base64
from io import BytesIO
import os
import re
import time

from faspell import SpellChecker, repeat_test

logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
corpus_root_path = 'batch_test/corpus'
special_sign = [u'\uff08', u'\uff09']  # 为括号、句号之类的，可以被bert纠错的符号准备
deep_url = ''
CONFIGS = json.loads(open('faspell_configs.json', 'r', encoding='utf-8').read())

class SmallCorpus(object):
    def __init__(self, corpus_root_path):
        self.corpus_root_path = corpus_root_path
        self.file_list = os.listdir(corpus_root_path)
        self.file_name = None
        self.corpus = self.load_corpus_randomly()

    def load_corpus_randomly(self):

        file_no = random.randint(0, len(self.file_list)-1)
        self.file_name = self.file_list[file_no]
        file_path = os.path.join(self.corpus_root_path, self.file_name)
        print('抽到文件：', file_path)
        corpus = []
        with codecs.open(file_path, 'r', 'utf-8') as f:
            for line in f:
                tl = line.strip()
                tl = re.split(r'[,|@@|，]\s*', tl)  # 由于爬取数据时做了拼接处理，所以需要在此拆分
                tl = self.filter_text(tl)
                corpus += tl
        logging.debug('corpus: %s', corpus[:100])

        return corpus

    @staticmethod
    def filter_text(text_list):
        '''过滤文本中的数字、空白
        :text_list: split之后的字符串列表
        '''
        new_text_list = []
        for t in text_list:
            t = filter_chinese_character(t)
            # t = re.sub(r'\d+', '', t)
            # t = re.sub(r'\s*', '', t)
            # t = re.sub(r'/', '-', t)
            if 0 < len(t) <= 19:
                new_text_list.append(t)

        return new_text_list

    def generate_text(self):
        '''
        从语料库中随机抽取一条文本；由于当前crnn的限制，只抽取19个字符以下的文本
        :return:text
        '''

        text = random.sample(self.corpus, 1)[0]
        # if len(text) > 19: return self.generate_text()

        return text


class ImgGenerator(object):
    '''
    生成图片
    corpus: Corpus object to generate text to draw
    font: font family(string), size (unit pound) and font color (in "#rrggbb" format)
    bg_color: in "#rrggbb" format
    '''
    def __init__(self, corpus,
                 font=("/Users/tt/correction/FASPell/batch_test/font/Songti SC Regular.ttf", 20, "#333333"),
                 bg_color=(210, 210, 210)):

        self.corpus = corpus
        self.font_family, self.font_size, self.font_color = font
        self.font = ImageFont.truetype(self.font_family, self.font_size)
        self.bg_color = bg_color

    def get_text(self):
        return self.corpus.generate_text()

    # by default, draw center align text
    def draw_text(self, img, text):
        dr = ImageDraw.Draw(img)
        dr.text((0, 0), text, fill = self.font_color, font = self.font)

    def draw_background(self):
        pass

    def transform(self):
        params = [1 - float(random.randint(1, 2)) / 100,
                  0,
                  0,
                  0,
                  1 - float(random.randint(1, 10)) / 100,
                  float(random.randint(1, 2)) / 500,
                  0.001,
                  float(random.randint(1, 2)) / 500
                  ]
        self.im = self.im.transform((self.width, self.height), Image.PERSPECTIVE, params)

    def filter(self, img):
        img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    def add_noise(self):
        pass

    def get_test_img_batch(self):
        pass

    def gen_img(self):
        '''
        一个生成器，生成图片
        :return:
        '''
        while True:
            text = self.get_text()
            width, height = self.font.getsize(text)
            img = Image.new("RGB", (width, height), (self.bg_color))
            # self.draw_background(img)
            self.draw_text(img, text)
            # self.add_noise(img)
            # self.transform(img)
            self.filter(img)

            yield img, text



def img_to_base64(img):
    '''
    transform PIL Image object to base64
    :param img:
    :return:
    '''

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return img_str

def img_to_string(img):

    img_str = img_to_base64(img).decode()
    content = []
    img = {"img": img_str}
    content.append(img)
    content = json.dumps(content)
    r = requests.post(url=deep_url, data=content)
    c = json.loads(r.content)
    prism_words_info = c["prism_wordsInfo"]

    return  prism_words_info[0]["word"]

def filter_chinese_character(char):
    '''
    仅保留文本中的汉字，及可以被bert接受的特殊符号
    :param char:
    :return:
    '''
    new_char = ''
    for c in char:
        if u'\u4e00' <= c <= u'\u9fa5':
            new_char += c
    return new_char

def compare_text(text1, text2):
    '''
    比较原始文本与识别后的文本是否相同。
    :param text1, text2:
    :return:
    '''

    if text1 == text2:
        return True
    else:
        return False

def correct_single_sentence(spell_checker, data):
    '''单句纠错'''


    all_results, _ = spell_checker.repeat_make_corrections([data], num=CONFIGS["general_configs"]["round"],
                                                                            is_train=False,
                                                                            train_on_difference=True)

    corrected_s = all_results[0][0]['corrected_sentence']
    history = all_results[0][0]['history']

    return corrected_s, history


def generate_test_cases_and_test(spell_checker, test_case_num = (5,20)):

    test_cases = []

    # could be seen as a batch; each batch reload a small file randomly
    for j in range(test_case_num[0]):
        # 随机从corpus中抽取一个文件
        corpus = SmallCorpus(corpus_root_path)
        logging.info(f'=====第{j}个文件：{corpus.file_name}')
        img_generator = ImgGenerator(corpus).gen_img()

        for i in range(test_case_num[1]):

            img, origin_text = next(img_generator)
            logging.info(f'-----第{j}-{i}个文本：{origin_text}')

            print('原始文本:', origin_text)

            crnn_text = img_to_string(img)

            print('识别结果：', crnn_text)

            is_same = compare_text(origin_text, crnn_text)

            print('是否识别正确：', is_same)

            if not is_same:
                img.save('batch_test/imgs/' + str(i) + origin_text + '.jpg')
                corrected_text, correction_history = correct_single_sentence(spell_checker, crnn_text)
                print('纠错结果：', corrected_text)
                is_correction_ok = compare_text(origin_text, corrected_text)
                test_cases.append({'No.':i,
                                   'origin_text':origin_text,
                                   'crnn_text':crnn_text,
                                   'corrected_text':corrected_text,
                                   'is_correction_ok':is_correction_ok,
                                   'correction_history':correction_history
                                    })

    # 保存全流程结果
    suffix = time.strftime("%m%d_%H%M%S", time.localtime())
    with codecs.open(f'batch_test/test_result_{suffix}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_cases, ensure_ascii=False, indent=4, sort_keys=False))

    # 将本次产生的测试用例保存到文件，方便下次测试FASPell
    with codecs.open(CONFIGS["exp_configs"]["testing_set"], 'a+', 'utf-8') as f:
        for i, c in enumerate(test_cases):
            f.write(f'{i}\t' + c['crnn_text'] + '\t' + c['origin_text'] + '\n')

    return test_cases


def correction_test(spell_checker):
    '''利用造的样本，测试纠错环节'''


    repeat_test(CONFIGS["exp_configs"]["testing_set"], spell_checker, CONFIGS["general_configs"]["round"],
                False, train_on_difference=True)


if __name__ == "__main__":
    spell_checker = SpellChecker()
    # generate_test_cases_and_test(spell_checker, (5,20))
    correction_test(spell_checker)

    print('done')





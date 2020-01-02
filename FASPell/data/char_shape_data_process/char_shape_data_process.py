'''将下载的原始数据处理为FALSpell需要的格式'''
import codecs
import logging
import pickle
import json
import os
import re

# 原始数据：
ids_path = 'ids.txt'
ids_path_supplementary = '../../../makemeahanzi-master/dictionary.txt'
# 原始数据读入后，暂存为二进制文件
char_ids_dict_path = 'char_ids_dict.txt'
char_unicode_dict_path = 'char_unicode_dict.txt'
# 经过递归处理后，得到的结果二进制文件
char_ids_result_dict_path = 'char_ids_result_dict.txt'
# 实际使用的txt文件
char_meta_path = '../char_meta.txt'
# 挑出的没有分解完全的叶子
leafs_path = 'leafs_to_decompose.txt'


logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def raw_data_to_dict():
    with codecs.open(ids_path, 'r', 'utf-8') as f:
        char_ids_dict = {}
        char_unicode_dict = {}
        for line in f:
            if line.find('#') == 0: continue
            print('======original line:', line.strip())
            line = line.replace('  ','\t')
            line_elements = line.strip().split('\t')  # line: U+5168	全	⿱人王[GJ]	⿱入王[TKV]
            print('line_elements:', line_elements)
            char_ids_dict[line_elements[1]] = line_elements[2].partition('[')[0]
            char_unicode_dict[line_elements[1]] = line_elements[0]

    return char_ids_dict, char_unicode_dict

def dump_char_ids_dict():
    char_ids_dict, char_unicode_dict = raw_data_to_dict()
    with open(char_ids_dict_path, 'wb') as f:
        pickle.dump(char_ids_dict, f)
    with open(char_unicode_dict_path, 'wb') as f:
        pickle.dump(char_unicode_dict, f)

def load_char_ids_dict():
    """加载char_ids_dict"""
    if not os.path.exists(char_ids_dict_path):
        dump_char_ids_dict()
    with open(char_ids_dict_path, 'rb') as f:
        char_ids_dict = pickle.load(f)

    return char_ids_dict

def load_char_ids_dict_2():  # result: dict {'char':'ids'}
    """加载补充的char_ids_dict_2"""
    char_ids_dict_2 = {}
    with codecs.open(ids_path_supplementary, 'r', 'utf-8') as f:
        for line in f:
            c = json.loads(line)
            char_ids_dict_2[c['character']] = c['decomposition']

    return char_ids_dict_2

def handle_stroke_level_problem():
    '''
    1、把 raw data 中叶子节点整出来
    2、用 补充数据decomposition一下
    3、再用 原始数据 decomposition 一下
    4、然后再看一下现在的叶子节点有哪些。可能需要手动处理
    '''
    char_ids_dict = load_char_ids_dict()
    leafs = {key: value for key, value in char_ids_dict.items() if len(value) == 1}
    char_ids_dict_supplementary = load_char_ids_dict_2()
    for k in leafs:
        new_value = char_ids_dict_supplementary.get(k, None)
        if new_value and len(new_value) > 1:
            leafs[k] = new_value

    char_unicode_dict = load_data(char_unicode_dict_path)

    with codecs.open(leafs_path, 'w', 'utf-8') as f:
        for k, v in leafs.items():
            line = '\t'.join([char_unicode_dict.get(k,'null'),k,v]) + '\n'
            f.write(line)

    leafs_new = {key: value for key, value in leafs.items() if len(value) == 1}

    return leafs_new



# ---------以上代码用于一次性生成文件---------

def dump_data(data, path):
    """暂存数据到二进制文件"""
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return None

def load_data(path,mode = 'pickle', key_no = 1, value_no = 2):
    """从json或二进制文件加载数据"""
    data = {}
    if mode == 'pickle':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif mode == 'json':
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                data[line[key_no]] = line[value_no]
    return data

def look_up_char():
    '''使用命令行查找汉字的分解，或者笔画
    0: 原始的字典
    1：补充字典
    2：手动修改的叶子字典
    3：最终的结果字典
    查汉字的分解： 3 字
    查笔画：3 丿 1
    '''
    char_ids_dict = load_data(char_ids_dict_path)
    char_ids_dict_supplementary = load_char_ids_dict_2()
    leafs = load_data(leafs_path,'json')
    char_ids_result_dict = load_data(char_meta_path, mode='json', key_no=1, value_no=3)
    char_unicode_dict = load_data(char_unicode_dict_path)

    data_sets = [char_ids_dict, char_ids_dict_supplementary, leafs, char_ids_result_dict]
    import sys
    if len(sys.argv) < 2:
        return
    data_no = int(sys.argv[1])
    char = sys.argv[2]
    if len(sys.argv) > 3:
        for k,v in data_sets[data_no].items():
            if v.find(char) == -1: continue
            print('\t'.join([char_unicode_dict[k],k,v]))
        return
    print(f'{char_unicode_dict[char]}\t{char}\t{data_sets[data_no].get(char,None)}')

def has_damn_num(char):
    for c in char:
        if '\u2460' <= c <= '\u2472': return True
    return False

def update_char_ids_dict():
    #1、读入，2、过滤，3、update
    new_leafs = load_data(leafs_path, mode='json')
    new_leafs = {k:v for k,v in new_leafs.items() if v.find('？')==-1}
    char_ids_dict = load_data(char_ids_dict_path)
    char_ids_dict.update(new_leafs)
    dump_data(char_ids_dict, char_ids_dict_path)

    return char_ids_dict
    # dump_data(char_ids_dict, 'new_char_ids_dict.txt')


def update_leafs():
    char_ids_dict = load_char_ids_dict()
    leafs = {key: value for key, value in char_ids_dict.items() if len(value) == 1}
    # 太坑了，原版用①②这种符号替换了一些汉字块，还得把这些汉字decomposition一下
    char_with_num = {key: value for key, value in char_ids_dict.items() if has_damn_num(value)}
    leafs.update(char_with_num)
    logging.info('和原数据融合后未处理完成的叶子数量：%d', len(leafs))
    char_unicode_dict = load_data(char_unicode_dict_path)
    with codecs.open(leafs_path, 'w', 'utf-8') as f:
        for k, v in leafs.items():
            line = '\t'.join([char_unicode_dict.get(k,'null'),k,v]) + '\n'
            f.write(line)


class Char(object):

    ids_dict = {}

    @classmethod
    def set_ids_dict(cls, ids_dict):
        cls.ids_dict = ids_dict

    def __init__(self, char):
        self.char = char
        self.ids_raw = self.ids_dict.get(char, char)
        self.ids = ''
        self.is_leaf = (len(self.ids_raw) == 1)
        self.is_structure = self.is_leaf and ('\u2FF0' <= self.ids_raw <= '\u2FFB')


    def generate_ids(self, char):
        '''
        :param char: Char object
        :return:
        '''

        logging.debug('now processing: %s', char.char)
        logging.debug('is structure: %s', char.is_structure)
        logging.debug('is leaf: %s', char.is_leaf)

        if char.is_leaf or char.is_structure:
            self.ids += char.char
            logging.debug('add ids: %s', char.char)
            return

        logging.debug('its ids: %s', char.ids_raw)

        for c in char.ids_raw:  # ⿱一⿻冂从
            self.generate_ids(Char(c))

        return


# 对于char_ids_dict中的每一个汉字，递归地求其IDS（二维汉字的一维表示，能表征其笔画、及笔画组织的结构） e.g.贫 : ⿱⿱⿰丿乁⿹𠃌丿⿵⿰丨𠃌⿰丿乁
def process_all_chars_in_dict(char_ids_dict):
    '''递归生成 分解到笔画 的ids，以二进制形式保存在 char_ids_result_dict_path 里。'''
    char_ids_result_dict = {}
    for i, c in enumerate(char_ids_dict):
        logging.debug('=====处理汉字：%s', c)
        char = Char(c)
        char.generate_ids(char)
        char_ids_result_dict[c] = char.ids
        logging.debug('=====处理结果：%s', char.ids)
        if i % 10000 ==0:
            logging.info('处理完成%d个汉字', i)

    dump_data(char_ids_result_dict, char_ids_result_dict_path)

def process_one_char_ids(c, updated = False):
    '''测试汉字'''

    if updated:
        char_ids_dict = update_char_ids_dict()
        logging.info('已更新char_ids_dict')
    else:
        char_ids_dict = load_data(char_ids_dict_path)
    Char.set_ids_dict(char_ids_dict)
    logging.debug('=====处理汉字：%s', c)
    char = Char(c)
    char.generate_ids(char)
    logging.debug('=====处理结果：%s', char.ids)



# 生成项目要求的char_meta.txt
def generate_char_meta():  # line = [unicode, char, placeholder, ids]
    '''U+5348	午	wu3;ng5;O;GO;ngọ	⿱⿰丿一⿻一丨
    '''
    placeholder = 'null;null;null;null;null'
    lines = []

    if not os.path.exists(char_ids_result_dict_path):
        process_all_chars_in_dict()

    with open(char_ids_result_dict_path, 'rb') as f:  # ids
        char_ids_result_dict = pickle.load(f)
    with open(char_unicode_dict_path, 'rb') as f:  # unicode
        char_unicode_dict = pickle.load(f)

    for char in char_ids_result_dict:
        line = [char_unicode_dict.get(char, 'null'),
                char,
                placeholder,
                char_ids_result_dict[char],
                '\n']
        line_string = '\t'.join(line)
        logging.debug('generated a new line: %s', line_string)
        lines.append(line_string)

    with codecs.open(char_meta_path, 'w', 'utf-8') as f:
        for line in lines:
            f.write(line)


def main(updated = True):
    """在手动修改 leafs_to_decompose 文件之后，运行main()，更新char_ids_dict"""
    if updated:
        char_ids_dict = update_char_ids_dict()
        logging.info('已更新char_ids_dict')
    else:
        char_ids_dict = load_data(char_ids_dict_path)
    Char.set_ids_dict(char_ids_dict)
    process_all_chars_in_dict(char_ids_dict)  # 递归至叶子节点
    logging.info('完成所有汉字的重新分解')
    generate_char_meta()  # 生成规定格式的文档
    logging.info('目标文件已保存')
    update_leafs()
    logging.info('已更新叶子文件')



if __name__ == '__main__':
    # process_one_char_ids('两')
    main(updated = True)
    # look_up_char()
    # update_leafs()







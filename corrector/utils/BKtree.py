#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : Tian
@File    : BKtree.py
@Time    : 2020/8/3 10:21 PM
@Desc    : BKTree class for keywords search
@Version : 1.0
"""

"""
reference: 刘树春等, 深度实践OCR：基于深度学习的文字识别, 9.1.1节
"""

from Levenshtein import distance
import os

class Node(object):  # 🌲的结点
    def __init__(self, word):
        self.word = word
        self.children = {}

    def __repr__(self):
        return '<Node: %r>' % self.word

class BKTree(object):
    def __init__(self, diction, dist_func=distance):
        self.root = None
        self.dist_func = dist_func
        self.diction = self.load_diction(diction)

    def add(self, word):
        if self.root is None:  # 根节点空着，先放到根节点
            self.root = Node(word)
            return

        node = Node(word)
        curr = self.root  # 初始比较对象为根节点
        dist = self.dist_func(word, curr.word)

        while dist in curr.children:  # 已经有对应的孩子了
            curr = curr.children[dist]
            dist = self.dist_func(word, curr.word)

        curr.children[dist] = node
        node.parent = curr


    def search(self, word, max_dist):
        """
        >>> tree = BKTree('../data/kwds_credit_report.txt')
        >>> tree.plant_tree()
        >>> tree.search('宋还本金',1)
        [<Node: '未还本金'>, <Node: '已还本金'>]
        >>> tree.search('数期大',2)
        [<Node: '逾期天'>]
        """
        candidates = [self.root]
        found = []
        while len(candidates) > 0:
            node = candidates.pop(0)  # 从头开始
            dist = self.dist_func(node.word, word)

            if dist <= max_dist:
                found.append(node)

            for child_dist, child in node.children.items():
                if dist - max_dist <= child_dist <= dist + max_dist:
                    candidates.append(child)
        if found:
            found = [f.word for f in found]
        return found

    @staticmethod
    def load_diction(diction):
        diction = os.path.join(os.path.dirname(os.path.dirname(__file__)), diction)
        with open(diction, encoding="utf-8") as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
        return set(lines)

    def plant_tree(self):  # 种树
        for w in self.diction:
            self.add(w)











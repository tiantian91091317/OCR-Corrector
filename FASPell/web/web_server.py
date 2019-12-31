# -*- coding:utf-8 -*-
from flask import Flask, jsonify, request, abort, render_template, Response
from faspell import SpellChecker
from threading import current_thread
import logging
import time
import os
import json

app = Flask(__name__)
app.debug = True
print('当前路径：', os.getcwd())
CONFIGS = json.loads(open('faspell_configs.json', 'r', encoding='utf-8').read())

@app.route("/")
def index():
    return render_template('index.html', version="version")

# 图片的识别
@app.route('/correction.post', methods=['POST'])
def correction():
    spell_checker = SpellChecker()
    print('request.form:', request.form)

    data = [request.form['sentence']]
    if data is None:
        abort(500)
        abort(Response('请输入待纠错的文本'))
    print('data:', data)
    all_results, correction_history = spell_checker.repeat_make_corrections(data, num=CONFIGS["general_configs"]["round"],
                                                                            is_train=False,
                                                                            train_on_difference=True)
    print('all_results:',all_results)

    result = all_results[0][0]



    return render_template('result.html', result=result)


if __name__ == "__main__":
    app.run()


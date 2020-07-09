# coding: utf-8
"""
模型转换 ckpt > pb
"""
import os
from correcter.bert_modified.masked_lm import Model  # 自己的模型网络
from correcter.bert_modified.masked_lm import MaskedLMConfig
import tensorflow as tf
from tensorflow.saved_model.signature_def_utils import build_signature_def
from tensorflow.saved_model.builder import SavedModelBuilder
from tensorflow.saved_model.utils import build_tensor_info
tf.app.flags.DEFINE_boolean('debug', True, '')
tf.app.flags.DEFINE_string('ckpt_mod_path', "model/pre-trained/bert_model.ckpt", '')
tf.app.flags.DEFINE_string('save_mod_dir', "model/multi_pb", '')

FLAGS = tf.app.flags.FLAGS

def convert():
    # 保存转换好的模型目录
    savedModelDir = FLAGS.save_mod_dir
    # 每次转换都生成一个版本目录
    for i in range(100000, 9999999):
        cur = os.path.join(savedModelDir, str(i))
        if not tf.gfile.Exists(cur):
            savedModelDir = cur
            break

    # 原ckpt模型
    ckptModPath = FLAGS.ckpt_mod_path

    model = Model(MaskedLMConfig())
    topn_probs, topn_predictions = model.top_n_predictions

    saver = tf.train.Saver()
    session = tf.Session()
    saver.restore(sess=session, save_path=ckptModPath)


    # 保存转换训练好的模型
    builder = SavedModelBuilder(savedModelDir)
    inputs = {
        "input_ids": build_tensor_info(model.input_ids),
        "input_mask": build_tensor_info(model.input_mask),
        "segment_ids": build_tensor_info(model.segment_ids),
        "masked_lm_positions": build_tensor_info(model.masked_lm_positions),
        "masked_lm_ids": build_tensor_info(model.masked_lm_ids),
        "masked_lm_weights": build_tensor_info(model.masked_lm_weights)
    }

    outputs = {
        "topn_probs":build_tensor_info(topn_probs),
        "topn_predictions":build_tensor_info(topn_predictions)
    }

    prediction_signature = build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME   # 标记输入输出
    )

    builder.add_meta_graph_and_variables(
        sess=session,
        tags=[tf.saved_model.tag_constants.SERVING],   # 打个默认的标签
        signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
    )

    builder.save()


if __name__ == '__main__':
    convert()

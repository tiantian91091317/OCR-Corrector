#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
bert 模型的 tfserving 调用
"""

import logging
import numpy as np
import tensorflow as tf
# import grpc
# from tensorflow.contrib.util import make_tensor_proto
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2_grpc

logger = logging.getLogger(__name__)


def create_channel(name, IP, PORT):
    logger.info("TF Serving 通道连接 - name:%s IP:%s PORT:%s", name, IP, PORT)
    # 报文大小限制
    options = [('grpc.max_send_message_length', 512 * 1024 * 1024),
               ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    channel = grpc.insecure_channel("{}:{}".format(IP, PORT), options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # 预测请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = name
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    logger.info("链接模型[%s]的通道创建,IP:%s,端口:%d,", name, IP, PORT)

    return stub, request

def correct_tf_serving_call(batch, conf):
    stub, request = create_channel(conf.correct_config.name, conf.tfserving_ip, conf.tfserving_port)

    input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, _ = batch

    request.inputs["input_ids"].CopyFrom(make_tensor_proto(np.array(input_ids), dtype=tf.int32))
    request.inputs["input_mask"].CopyFrom(make_tensor_proto(np.array(input_mask), dtype=tf.int32))
    request.inputs["segment_ids"].CopyFrom(make_tensor_proto(np.array(segment_ids), dtype=tf.int32))
    request.inputs["masked_lm_positions"].CopyFrom(make_tensor_proto(np.array(masked_lm_positions), dtype=tf.int32))
    request.inputs["masked_lm_ids"].CopyFrom(make_tensor_proto(np.array(masked_lm_ids), dtype=tf.int32))
    request.inputs["masked_lm_weights"].CopyFrom(make_tensor_proto(np.array(masked_lm_weights), dtype=tf.float32))

    logger.debug("调用纠错识别模型预测，开始")
    response = stub.Predict(request, 60.0)
    logger.debug("调用纠错识别模型预测，结束")

    results = {}
    for key in response.outputs:
        tensor_proto = response.outputs[key]
        results[key] = tf.contrib.util.make_ndarray(tensor_proto)
        logger.debug("解析tensor完成")
    topn_probs = results["topn_probs"]
    topn_predictions = results["topn_predictions"]

    return np.array(topn_probs, dtype=float), topn_predictions
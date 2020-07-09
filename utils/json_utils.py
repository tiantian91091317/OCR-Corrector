#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import json
import logging

logger = logging.getLogger(__name__)


def dic2class(py_data, obj):
    """
    已经转成dict的数据转成自定义独享
    :param py_data: dict格式
    :param obj: 自定义对象
    :return:
    """
    for name in [name for name in dir(obj) if not name.startswith('_')]:
        if name not in py_data:
            setattr(obj, name, None)
        else:
            value = getattr(obj, name)
            setattr(obj, name, set_value(value, py_data[name]))


def dic2class_ignore(py_data, obj):
    """
    已经转成dict的数据转成自定义对象
    :param py_data: dict格式
    :param obj: 自定义对象
    :return:
    """
    for name in [name for name in dir(obj) if not name.startswith('_')]:
        if name not in py_data:
            logger.debug("%r json中不存在，忽略，取默认值",name)
            # setattr(obj, name, obj.name)
        else:
            value = getattr(obj, name)
            setattr(obj, name, set_value(value, py_data[name],ignore_null=True))


def set_value(value, py_data,ignore_null=False):
    if str(type(value)).__contains__('.'):
        # value 为自定义类
        if ignore_null:
            dic2class_ignore(py_data, value)
        else:
            dic2class(py_data, value)
    elif str(type(value)) == "<class 'list'>":
        # value为列表
        if value.__len__() == 0:
            # value列表中没有元素，无法确认类型
            value = py_data
        else:
            # value列表中有元素，以第一个元素类型为准
            child_value_type = type(value[0])
            value.clear()
            for child_py_data in py_data:
                child_value = child_value_type()
                child_value = set_value(child_value, child_py_data)
                value.append(child_value)
    else:
        value = py_data
    return value


def obj2json(obj):
    """
    复杂结构转json
    :param obj:
    :return: dict结构
    """
    result = json.dumps(obj, ensure_ascii=False, default=lambda o: o.__dict__, sort_keys=False,
                        indent=4)
    return json.loads(result)
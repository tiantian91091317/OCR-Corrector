form .. import pycorrector

sentence = '胯行转账'

de = pycorrector.detect(sentence)
print(de)
co = pycorrector.correct(sentence)
print(co)


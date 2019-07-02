form .. import pycorrector

sentence = '兔密支付'

de = pycorrector.detect(sentence)
print(de)
co = pycorrector.correct(sentence)
print(co)


# OCR-Corrector

专为OCR设计的纠错器。

未来计划涵盖OCR需要的各种NLP工具，包括：
1. 粘连文本分词
2. 命名实体识别
3. 键值对匹配

# 文本纠错功能（2020.07）

输入OCR识别结果（文本+单字符置信度），输出修正后的文本。

## 示例1

输入：
```
text = '我爱北京大安门'
probs = [0.99, 0.99, 0.99, 0.99, 0.56, 0.99, 0.99]
```

输出：
```
text_corrected = '我爱北京天安门'
```

## 示例2

输入：
```
text = '本着平等、白愿、诚信、互利的原则'
probs = [0.99, 0.99, 0.99, 0.99, 0.99, 0.78, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
```
输出：
```
text_corrected = '本着平等、自愿、诚信、互利的原则'
```


# 细分场景

目前按照业务场景，分别开发了两种纠错器：文档识别纠错器，单据识别纠错器

## 文档识别
文档是指书籍内页拍摄的图片、扫描的合同等有大段文字的图片。

### 示例


原图：
<img src="./data/img1.jpeg" width="400" />

纠错结果：
<img src="./data/示意图.png" width="400" />


## 单据识别
单据是指字段、格式相对固定，有统一模板或者近似统一的图片，比如身份证、银行卡、驾驶证、发票等等，主要特点是单据上出现的文本段相对固定。

### 示例

**todo:**
原图：
<img src="./data/示意图.png" width="400" />

错误文本：
<img src="./data/示意图.png" width="400" />


# 使用方法

1. clone 项目
```bash
git clone https://github.com/tiantian91091317/OCR-Corrector.git
```

2. 下载模型和数据
1）下载[预训练好的BERT模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) 到 corrector/model/pre-trained 目录下
2）下载 char_meta.txt 字形编码（IDS）文件下载后放到 corrector/config 目录下
下载地址:https://pan.baidu.com/s/1iqA-GbzzHBBWfWaxe1g_fg  密码:3f11

3. 安装
```bash
python setup.py install
```

## 使用

### 方法一
可以嵌入到OCR识别的代码里面，将识别模型输出的结果输入纠错器。

```python
import ocr_corrector

corrector = ocr_corrector.initial()
ocr_results, recog_probs = my_ocr(img)
ocr_res_corrected = corrector.correct(ocr_results, recog_probs, biz_type)

```

可以通过运行以下命令进行测试：
```bash
# 测试文档识别纠错
python demo.py --img=data/img1.jpeg --biz=doc --api=own
# 测试单据识别纠错
python demo.py --img=data/img2.jpeg --biz=report --api=own
```

### 方法二
可以调用识别API后进行后处理。目前支持[阿里高精版识别接口](https://market.aliyun.com/products/57124001/cmapi028554.html?spm=5176.200117.0.0.4f57261aiZhbVd&innerSource=search#sku=yuncode2255400000)的调用。
需要先申请 app code（可以开通免费试用）；然后在 ```corrector/api_call/ali_ocr.py``` 中更新app code：
```python
	url = 'https://ocrapi-advanced.taobao.com/ocrservice/advanced'
    post_data = {"img":img,
                 "prob":True,
                 "charInfo":True
                 }
    app_code = your_app_code
```
然后可以任意传图片测试纠错结果。

```bash
python demo.py --img=data/test.jpg --biz=[doc|report] --api=ali 
```

### 新增单据类型
文档识别的纠错主要利用局部语义信息进行纠错，**无需**特殊配置；
对于单据识别，由于主要基于其关键词表进行纠错，所以需要进行配置（以新增保单识别为例）：
1. 在 ```corrector/config/config.json``` 中增加新单据类型的配置：
```json
{
      "biz_type": "insurance",
      "corrector_type":"keyword",
      "prob_threshold": 0.9,
      "similarity_threshold": 0.6,
      "char_meta_file": "config/char_meta.txt",
      "key_words_file": "config/kwds_insurance.txt"
    }
```
2. 在 ```corrector/config/``` 目录下增加关键词表 ```kwds_insurance.txt```：
```
投保人
被保险人
受益人
险种名称
……
```

# 原理

## 文档识别的纠错

## 单据识别的纠错

# 参考项目

# 创新点

## 置信度（2020.07）

## 关键词表法纠错（2020.07）

# 未来计划

1. 将纠错拓展到非汉字的其他字符，比如 日期、证件号码、标点符号等；
2. 形成OCR所需的NLP工具包，包括粘连文本分词、命名实体识别、键值对匹配等等

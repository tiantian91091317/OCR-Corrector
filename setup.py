from setuptools import setup
setup(
	name="ocr_corrector",
	version="1.0",
	description="ocr correct service",
	author="Tian",
	author_email="tiantian91091317@gmail.com",
	url="https://github.com/tiantian91091317/OCR-Corrector",
	license="LGPL",
	packages=['ocr_corrector', 'ocr_corrector.bert_modeling',
			  'ocr_corrector.utils', 'ocr_corrector.api_call'],
	package_dir={'ocr_corrector':'corrector'},
	package_data={'ocr_corrector.bert_modeling':['vocab.txt','bert_config.json'],
				  'ocr_corrector':['data/*', 'config/config.json',
								   'config/kwds*', '../demo.py']}
	)
#@@ json 数据导入

import json
path = 'E:/Job/MLanguage/Python/Workplace_DataAnalysis/pydata-book/datasets/bitly_usagov/example.txt'
records = [json.loads(line) for line in open(path)]

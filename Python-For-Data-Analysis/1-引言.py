#@@ json 数据导入

import json
path = 'E:/Job/MLanguage/Python/Workplace_DataAnalysis/pydata-book/datasets/bitly_usagov/example.txt'
records = [json.loads(line) for line in open(path)]
time_zones = [rec['tz'] for rec in records if 'tz' in rec]

#@@ defaultdict （对dict的功能扩展，当key不存在时，放回一个默认值，而不是KeyError）
from collections import defaultdict
# 统计次数
def get_count(sequence):
  counts = defaultdict(int)
  for x in sequence:
    count[x] += 1
  return count


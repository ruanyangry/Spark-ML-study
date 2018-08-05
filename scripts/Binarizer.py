# _*_ coding:utf-8 _*_

'''
Binarizer
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import Binarizer

spark = SparkSession.builder.appName("binarizer").getOrCreate()

continuousDataFrame = spark.createDataFrame([
    (0, 0.1),
    (1, 0.8),
    (2, 0.2)
], ["label", "feature"])

print("#-------------------------------------")
print(continuousDataFrame)
print("#-------------------------------------")
print(" ")

binarizer=Binarizer(threshold=0.5,inputCol="feature",outputCol="binarizer_feature")
binarizedDataFrame=binarizer.transform(continuousDataFrame)

print("#-------------------------------------")
print(binarizedDataFrame)
print("#-------------------------------------")
print(" ")

binarizedFeatures=binarizedDataFrame.select("binarizer_feature")

for binarized_feature, in binarizedFeatures.collect():
	print(binarized_feature)

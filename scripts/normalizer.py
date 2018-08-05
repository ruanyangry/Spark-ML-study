# _*_ coding:utf-8 _*_

'''
Normalizer
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import Normalizer

spark = SparkSession.builder.appName("normalizer").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

dataframe=spark.read.format("libsvm").load(paths+"sample_isotonic_regression_libsvm_data.txt")

normalizer=Normalizer(inputCol="features",outputCol="normFeatures",p=1.0)

l1NormData = normalizer.transform(dataframe)
l1NormData.show()

lInfNormData=normalizer.transform(dataframe,{normalizer.p:float("inf")})
lInfNormData.show()

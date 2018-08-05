# _*_ coding:utf-8 _*_

'''
Vectorindexer
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorIndexer

spark = SparkSession.builder.appName("vectorindexer").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_isotonic_regression_libsvm_data.txt")

indexer=VectorIndexer(inputCol="features",outputCol="indexed",maxCategories=10)

indexerModel=indexer.fit(data)

indexedData=indexerModel.transform(data)

indexedData.show()


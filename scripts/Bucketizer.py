# _*_ coding:utf-8 _*_

'''
Bucketizer
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer

spark = SparkSession.builder.appName("Bucketizer").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

#dataframe=spark.read.format("libsvm").load(paths+"sample_isotonic_regression_libsvm_data.txt")

splits=[-float("inf"),-0.5,0.0,0.5,float("inf")]

data=[(-0.5,), (-0.3,), (0.0,), (0.2,)]

dataFrame=spark.createDataFrame(data,["features"])

bucketizer=Bucketizer(splits=splits,inputCol="features",outputCol="bucketizerfeatures")

bucketedData=bucketizer.transform(dataFrame)

bucketedData.show()


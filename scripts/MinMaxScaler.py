# _*_ coding:utf-8 _*_

'''
MinMaxScaler
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler

spark = SparkSession.builder.appName("MinMaxScaler").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

dataframe=spark.read.format("libsvm").load(paths+"sample_isotonic_regression_libsvm_data.txt")

scaler=MinMaxScaler(inputCol="features",outputCol="scaledFeatures")

scalerModel=scaler.fit(dataframe)

scaledData=scalerModel.transform(dataframe)

scaledData.show()

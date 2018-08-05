# _*_ coding:utf-8 _*_

'''
StandardScaler
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler

spark = SparkSession.builder.appName("normalizer").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

dataframe=spark.read.format("libsvm").load(paths+"sample_isotonic_regression_libsvm_data.txt")

scaler=StandardScaler(inputCol="features",outputCol="scaledfeatures",withStd=True,\
withMean=False)

scalerModel=scaler.fit(dataframe)

scaledData=scalerModel.transform(dataframe)

scaledData.show()


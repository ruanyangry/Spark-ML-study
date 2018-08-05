# _*_ coding:utf-8 _*_

'''
VectorAssembler
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("VectorAssembler").getOrCreate()

dataset=spark.createDataFrame([(0, 18, 1.0, Vectors.dense([0.0, 10.0, 0.5]), 1.0)],\
["id", "hour", "mobile", "userFeatures", "clicked"])

assembler=VectorAssembler(inputCols=["hour","mobile","userFeatures"],outputCol="features")

output=assembler.transform(dataset)

print(output.select("features","clicked").first())
    

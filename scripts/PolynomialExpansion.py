# _*_ coding:utf-8 _*_

'''
PolynomialExpansion
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("polynomialexpansion").getOrCreate()

df = spark.createDataFrame([(Vectors.dense([-2.0, 2.3]),),
                      (Vectors.dense([0.0, 0.0]),),
                      (Vectors.dense([0.6, -1.1]),)],
                     ["features"])
                     
px=PolynomialExpansion(degree=3,inputCol="features",outputCol="polyFeatures")

polyDF=px.transform(df)

for expanded in polyDF.select("polyFeatures").take(3):
	print(expanded)

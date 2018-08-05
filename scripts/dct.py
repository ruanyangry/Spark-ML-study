# _*_ coding:utf-8 _*_

'''
Discrete Cosine Transform(DCT)
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("dct").getOrCreate()

df = spark.createDataFrame([
    (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
    (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
    (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

dct=DCT(inverse=False,inputCol="features",outputCol="featuresDCT")

dctDf=dct.transform(df)

for dcts in dctDf.select("featuresDCT").take(3):
	print(dcts)

dctDf.show()

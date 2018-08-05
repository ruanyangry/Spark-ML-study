# _*_ coding:utf-8 _*_

'''
SQLTransformer
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import SQLTransformer

spark = SparkSession.builder.appName("SQLTransformer").getOrCreate()

df=spark.createDataFrame([(0,1.0,3.0),(2,2.0,5.0)],["id","v1","v2"])

sqlTrans=SQLTransformer(statement="SELECT *,(v1+v2) AS v3,(v1*v2) AS v4 FROM __THIS__")

sqlTrans.transform(df).show()

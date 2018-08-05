# _*_ coding:utf-8 _*_

'''
StringIndexer
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder.appName("stringindexer").getOrCreate()

df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])
    
indexer=StringIndexer(inputCol="category",outputCol="categoryindex")

indexed=indexer.fit(df).transform(df)

indexed.show()



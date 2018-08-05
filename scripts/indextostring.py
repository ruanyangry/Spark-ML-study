# _*_ coding:utf-8 _*_

'''
IndexToString
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import IndexToString, StringIndexer

spark = SparkSession.builder.appName("indextostring").getOrCreate()

df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])

stringIndexer=StringIndexer(inputCol="category",outputCol="categoryindex")

model=stringIndexer.fit(df)

indexed=model.transform(df)

indexed.show()

converter=IndexToString(inputCol="categoryindex",outputCol="originalcategory")

converted=converter.transform(indexed)

converted.select("id","originalcategory").show()

df.show()

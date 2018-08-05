# _*_ coding:utf-8 _*_

'''
OneHotEncoder
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder,StringIndexer

spark = SparkSession.builder.appName("onehotencoder").getOrCreate()

df = spark.createDataFrame([
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (5, "c")
], ["id", "category"])

stringIndexer=StringIndexer(inputCol="category",outputCol="categoryIndex")

model=stringIndexer.fit(df)

indexed=model.transform(df)

encoder=OneHotEncoder(dropLast=False,inputCol="categoryIndex",outputCol="categoryvecs")

encoder=encoder.transform(indexed)

encoder.select("id","categoryvecs")

encoder.show()

# _*_ coding:utf-8 _*_

'''
CountVectorizer methods
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer

spark = SparkSession.builder.appName("countvectorizer").getOrCreate()

df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])

cv=CountVectorizer(inputCol="words",outputCol="reatures",vocabSize=3,minDF=2.0)

model=cv.fit(df)
result=model.transform(df)
result.show()

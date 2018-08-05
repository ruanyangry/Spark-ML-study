# _*_ coding:utf-8 _*_

'''
Word2Vec methods
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec

spark = SparkSession.builder.appName("word2vec").getOrCreate()

documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

word2vec=Word2Vec(vectorSize=3,minCount=0,inputCol="text",outputCol="result")
model=word2vec.fit(documentDF)
result=model.transform(documentDF)

for feature in result.select("result").take(3):
	print(feature)

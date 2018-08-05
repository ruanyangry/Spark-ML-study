# _*_ coding:utf-8 _*_

'''
TF-IDF methods
'''

from pyspark.ml.feature import HashingTF,IDF,Tokenizer
from pyspark import SparkContext as sc
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TFIDF").getOrCreate()

sentenceData = spark.createDataFrame([  
    (0, "Hi I heard about Spark"),  
    (0, "I wish Java could use case classes"),  
    (1, "Logistic regression models are neat")  
], ["label", "sentence"]) 

# First, Tokenizer method
tokenizer=Tokenizer(inputCol='sentence',outputCol="words")
wordsdata=tokenizer.transform(sentenceData)

# Second, HashingTF method

hashingTF=HashingTF(inputCol="words",outputCol="rawFeatures",numFeatures=20)
featurizedData=hashingTF.transform(wordsdata)

# Last, idf method

idf=IDF(inputCol="rawFeatures",outputCol="features")
idfModel=idf.fit(featurizedData)
rescaledData=idfModel.transform(featurizedData)

for features_label in rescaledData.select("features","label").take(3):
	print(features_label)

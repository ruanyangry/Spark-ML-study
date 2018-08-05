# _*_ coding:utf-8 _*_

'''
Tokenizer and RegexTokenizer
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer,RegexTokenizer

spark = SparkSession.builder.appName("tokenizerr").getOrCreate()

# First, build dataframe

sentenceDataFrame = spark.createDataFrame([
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
], ["label", "sentence"])

tokenizer=Tokenizer(inputCol="sentence",outputCol="words")
wordsDataFrame=tokenizer.transform(sentenceDataFrame)
print(wordsDataFrame)

for words_label in wordsDataFrame.select("words","label").take(3):
	print(words_label)
	
regexTokenizer=RegexTokenizer(inputCol="sentence",outputCol="words",pattern="\\w")

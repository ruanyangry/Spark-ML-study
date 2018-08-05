# _*_ coding:utf-8 _*_

'''
n-gram method
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram

spark = SparkSession.builder.appName("n-gram").getOrCreate()

# Create Dataframe


wordDataFrame = spark.createDataFrame([
    (0, ["Hi", "I", "heard", "about", "Spark"]),
    (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
    (2, ["Logistic", "regression", "models", "are", "neat"])
], ["label", "words"])

ngram=NGram(inputCol="words",outputCol="ngrams")

# Transform

ngramDataFrame=ngram.transform(wordDataFrame)

for ngrams_label in ngramDataFrame.select("ngrams","label").take(3):
	print(ngrams_label)

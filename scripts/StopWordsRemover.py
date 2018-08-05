# _*_ coding:utf-8 _*_

'''
StopWordsRemover methods
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover

spark = SparkSession.builder.appName("stopwordsremover").getOrCreate()

sentenceData = spark.createDataFrame([
    (0, ["I", "saw", "the", "red", "baloon"]),
    (1, ["Mary", "had", "a", "little", "lamb"])
], ["label", "raw"])


remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
remover.transform(sentenceData).show(truncate=False)

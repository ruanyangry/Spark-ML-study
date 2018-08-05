# _*_ coding:utf-8 _*_

'''
NaiveBayes
'''

from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_libsvm_data.txt")

train,test=data.randomSplit([0.6,0.4],1234)

nb=NaiveBayes(smoothing=1.0,modelType="multinomial")

model=nb.fit(train)
print("#----------------------------------------")
print(model)
print("#----------------------------------------")
print(" ")

result=model.transform(test)
predictionAndLabels=result.select("prediction","label")

print("#----------------------------------------")
print(predictionAndLabels)
print("#----------------------------------------")
print(" ")

evaluator=MulticlassClassificationEvaluator(metricName="accuracy")
print("Accuracy: " + str(evaluator.evaluate(predictionAndLabels)))

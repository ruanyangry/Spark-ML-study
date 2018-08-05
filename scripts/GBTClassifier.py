# _*_ coding:utf-8 _*_

'''
GBTClassifier
'''

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer,VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_libsvm_data.txt")

labelIndexer=StringIndexer(inputCol="label",outputCol="indexedLabel").fit(data)

featureIndexer=VectorIndexer(inputCol="features",outputCol="indexedFeatures",maxCategories=4).fit(data)

trainingData,testData=data.randomSplit([0.7,0.3])

# Train a GBT model

gbt=GBTClassifier(labelCol="indexedLabel",featuresCol="indexedFeatures",maxIter=10)

# Chain indexers and GBT in a Pipeline

pipeline=Pipeline(stages=[labelIndexer,featureIndexer,gbt])

# Train model. This also runs the indexers

model=pipeline.fit(trainingData)

# Make predictions

predictions=model.transform(testData)

# Select example rows to display

predictions.select("prediction", "indexedLabel", "features").show(10)

# Select (prediction,true label) and compute test error

evaluator=MulticlassClassificationEvaluator(labelCol="indexedLabel",\
predictionCol="prediction",metricName="accuracy")

accuracy=evaluator.evaluate(predictions)

print("Test Error = %g "%(1.0-accuracy))

gbtModel=model.stages[2]

print(gbtModel)

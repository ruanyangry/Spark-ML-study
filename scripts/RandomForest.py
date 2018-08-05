# _*_ coding:utf-8 _*_

'''
RandomForest
'''

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer,VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_libsvm_data.txt")

labelIndexer=StringIndexer(inputCol="label",outputCol="indexedLabel").fit(data)

featureIndexer=VectorIndexer(inputCol="features",outputCol="indexedFeatures",maxCategories=4).fit(data)

trainingData,testData=data.randomSplit([0.7,0.3])

rf=RandomForestClassifier(labelCol="indexedLabel",featuresCol="indexedFeatures",numTrees=10)

pipeline=Pipeline(stages=[labelIndexer,featureIndexer,rf])

model=pipeline.fit(trainingData)

predictions=model.transform(testData)

predictions.select("prediction", "indexedLabel", "features").show(10)

evaluator=MulticlassClassificationEvaluator(labelCol="indexedLabel",predictionCol="prediction",\
metricName="accuracy")

accuracy=evaluator.evaluate(predictions)

print("Test Error = %g" % (1.0 - accuracy))
 
rfModel = model.stages[2]
print(rfModel)  # summary only

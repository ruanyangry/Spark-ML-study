# _*_ coding:utf-8 _*_

'''
DecisionTree
'''

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer,VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

# Load the data stored in LIBSVM format as a DataFrame

data=spark.read.format("libsvm").load(paths+"sample_libsvm_data.txt")

# Index labels,adding metadata to the label column
# Fit on whole dataset to include all labels in index.

labelIndexer=StringIndexer(inputCol="label",outputCol="indexedLabel").fit(data)

# Automatically indentify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continous

featureIndexer=VectorIndexer(inputCol="features",outputCol="indexedFeatures",\
maxCategories=4).fit(data)

# Split the data into training and test sets

trainingData,testData=data.randomSplit([0.7,0.3])

# Train a DecisionTree model

dt=DecisionTreeClassifier(labelCol="indexedLabel",featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline

pipeline=Pipeline(stages=[labelIndexer,featureIndexer,dt])

# Train model. This also runs the indexers

model=pipeline.fit(trainingData)

# Make predictions

predictions=model.transform(testData)

# select example row to display

predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction,true label) and compute test error

evaluator=MulticlassClassificationEvaluator(labelCol="indexedLabel",predictionCol="prediction",\
metricName="accuracy")

accuracy=evaluator.evaluate(predictions)

print("Test error = %g "%(1.0-accuracy))

treeModel=model.stages[2]

# summary only

print(treeModel)

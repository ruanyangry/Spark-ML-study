# _*_ coding:utf-8 _*_

'''
DecisionTreeRegressor
'''

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("DecisionTreeRegressor").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_libsvm_data.txt")

# Automatically identify categorical features, and indexthem
# We specify maxCategories so features with > 4 distinct values are treated
# as continues

featureIndexer=VectorIndexer(inputCol="features",outputCol="indexedFeatures",\
maxCategories=4).fit(data)

trainingData,testData=data.randomSplit([0.7,0.3])

# Train a DecisionTree model
dt=DecisionTreeRegressor(featuresCol="indexedFeatures")

# chain indexer and tree in a Pipeline

pipeline=Pipeline(stages=[featureIndexer,dt])

# Train model. This also runs the indexer

model=pipeline.fit(trainingData)

# Make predictions

predictions=model.transform(testData)

# Select example rows to display

predictions.select("prediction","label","features").show(5)

# Select (prediction,true label) and compute test error

evaluator=RegressionEvaluator(labelCol="label",predictionCol="prediction",metricName="rmse")

rmse=evaluator.evaluate(predictions)

print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

treeModel = model.stages[1]

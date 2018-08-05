# _*_ coding:utf-8 _*_

'''
GBTRegressor
'''

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("GBTRegressor").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_libsvm_data.txt")

featureIndexer=VectorIndexer(inputCol="features",outputCol="indexedFeatures",maxCategories=4).fit(data)

trainingData,testData=data.randomSplit([0.7,0.3])

gbt=GBTRegressor(featuresCol="indexedFeatures",maxIter=10)

pipeline=Pipeline(stages=[featureIndexer,gbt])

model=pipeline.fit(trainingData)

predictions=model.transform(testData)

predictions.select("prediction","label","features").show(5)

evaluator=RegressionEvaluator(labelCol="label",predictionCol="prediction",metricName="rmse")

rmse=evaluator.evaluate(predictions)

print("Root Mean Squared Error (rmse) on test data %g"%(rmse))

gbtModel=model.stages[1]

print(gbtModel)

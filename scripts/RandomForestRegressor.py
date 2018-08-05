# _*_ coding:utf-8 _*_

'''
RandomForestRegressor
'''

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("RandomForestRegressor").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_libsvm_data.txt")

featureIndexer=VectorIndexer(inputCol="features",outputCol="indexedFeatures",maxCategories=4).fit(data)

trainingData,testData=data.randomSplit([0.7,0.3])

rf=RandomForestRegressor(featuresCol="indexedFeatures")

pipeline=Pipeline(stages=[featureIndexer,rf])

model=pipeline.fit(trainingData)

predictions=model.transform(testData)

predictions.select("prediction","label","features").show(5)

evaluator=RegressionEvaluator(labelCol="label",predictionCol="prediction",metricName="rmse")

rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
 
rfModel = model.stages[1]
print(rfModel)  # summary only

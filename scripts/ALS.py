# _*_ coding:utf-8 _*_

'''
Recommendation: ALS
'''

from pyspark.sql import SparkSession,Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName("ALS").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

lines=spark.read.text(paths+"als/sample_movielens_ratings.txt").rdd
parts=lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=long(p[3])))

ratings=spark.createDataFrame(ratingsRDD)
training,test=ratings.randomSplit([0.8,0.2])

# Build the recommendation model using ALS on the training data

als=ALS(maxIter=5,regParam=0.01,userCol="userId",itemCol="moiveId",ratingCol="rating")
model=als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions=model.transform(test)
evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")

rmse=evaluator.evaluate(predictions)
print("Root Mean Square Error = "+str(rmse))

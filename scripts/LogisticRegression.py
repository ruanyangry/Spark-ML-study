# _*_ coding:utf-8 _*_

'''
LogisticRegression
'''

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.appName("LogisticRegression").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

# Load training data

training=spark.read.format("libsvm").load(paths+"sample_libsvm_data.txt")

lr=LogisticRegression(maxIter=10,regParam=0.3,elasticNetParam=0.8)

# Fit the model

lrModel=lr.fit(training)

# print the coefficients and intercept for logistic regression

print("Coefficients: "+str(lrModel.coefficients))
print("Intercept: "+str(lrModel.intercept))

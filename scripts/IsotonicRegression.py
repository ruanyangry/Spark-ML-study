# _*_ coding:utf-8 _*_

'''
IsotonicRegression
'''

from pyspark.sql import SparkSession
from pyspark.ml.regression import IsotonicRegression,IsotonicRegressionModel

spark = SparkSession.builder.appName("AFTSurvivalRegression").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_isotonic_regression_libsvm_data.txt")

# Trains an isotonic regression model.

model=IsotonicRegression().fit(data)

print("Boundaries in increasing order: "+str(model.boundaries))
print("Predictions associated with the boundaries: "+str(model.predictions))

# Makes predictions

model.transform(data).show()


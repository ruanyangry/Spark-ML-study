# _*_ coding:utf-8 _*_

'''
GeneralizedLinearRegression
'''

from pyspark.sql import SparkSession
from pyspark.ml.regression import GeneralizedLinearRegression

spark = SparkSession.builder.appName("GeneralizedLinearRegression").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

dataset=spark.read.format("libsvm").load(paths+"sample_linear_regression_data.txt")

glr=GeneralizedLinearRegression(family="gaussian",link="identity",maxIter=10,regParam=0.3)

model=glr.fit(dataset)

print("Coefficient: "+str(model.coefficients))
print("Intercept: "+str(model.intercept))

# Summarize the model over the training set and print out som metrics

summary=model.summary

print("#--------------------------------------------#")
print(summary)
print("#--------------------------------------------#")
print(" ")

print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))
print("Dispersion: " + str(summary.dispersion))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()

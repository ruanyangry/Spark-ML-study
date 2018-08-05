# _*_ coding:utf-8 _*_

'''
TrainValidationSplit	
'''

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder,TrainValidationSplit

spark = SparkSession.builder.appName("TrainValidationSplit").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

# Prepare training and test data

data=spark.read.format("libsvm").load(paths+"sample_linear_regression_data.txt")

train,test=data.randomSplit([0.7,0.3])
lr=LinearRegression(maxIter=10,regParam=0.1)


# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.

paramGrid=ParamGridBuilder().addGrid(lr.regParam,[0.1,0.01]).addGrid(lr.elasticNetParam,[0.0,0.5,1.0]).build()

# In this case the estimator is simply the linear regression.
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.

tvs=TrainValidationSplit(estimator=lr,estimatorParamMaps=paramGrid,\
evaluator=RegressionEvaluator(),trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.

model=tvs.fit(train)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.

prediction=model.transform(test)

for row in prediction.take(5):
	print(row)

# _*_ coding:utf-8 _*_

'''
OneVsRest
'''

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression,OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("OneVsRest").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_multiclass_classification_data.txt")

# generate the train/test split.

train,test=data.randomSplit([0.7,0.3])

# instantiate the base classifier

lr=LogisticRegression(maxIter=10,tol=1E-6,fitIntercept=True)

# instantiate the one vs Rest Classifier

ovr=OneVsRest(classifier=lr)

# Train the multiclass model

ovrModel=ovr.fit(train)

# score the model on test data

predictions=ovrModel.transform(test)

predictions.show()

# obtain evaluator

evaluator=MulticlassClassificationEvaluator(metricName="accuracy")

# compute the classification error on test data

accuracy=evaluator.evaluate(predictions)
print("Test Error: "+str(1-accuracy))

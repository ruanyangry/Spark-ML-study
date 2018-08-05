# _*_ coding:utf-8 _*_

'''
MultilayerPerceptronClassifier
'''

from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_multiclass_classification_data.txt")

splits=data.randomSplit([0.6,0.4],1234)

train=splits[0]
test=splits[1]

# Specify layers for the neural network
# input layer of size 4 (features),two intermediate of size 5 and 4
# and output of size 3 (classes)

layers=[4,5,4,3]

# create the trainer and set its parameters

trainer=MultilayerPerceptronClassifier(maxIter=100,layers=layers,blockSize=128,seed=1234)

# train the model

model=trainer.fit(train)

# compute accuracy on the test set

result=model.transform(test)

predictionAndLabels=result.select("prediction","label")

evaluator=MulticlassClassificationEvaluator(metricName="accuracy")

print("Accuracy: " + str(evaluator.evaluate(predictionAndLabels)))

result.show()

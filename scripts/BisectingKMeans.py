# _*_ coding:utf-8 _*_

'''
BisectingKMeans
'''

from pyspark.sql import SparkSession
from pyspark.ml.clustering import BisectingKMeans

spark = SparkSession.builder.appName("BisectingKMeans").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_kmeans_data.txt")

# Trains a bisecting k-means model.

bkm=BisectingKMeans().setK(2).setSeed(1)
model=bkm.fit(data)

# Evaluate clustering

cost=model.computeCost(data)
print("Within Set Sum of Squared Errors = "+str(cost))

# Shows the result
print("Cluster Centers:")
centers=model.clusterCenters()
for center in centers:
	print(center)

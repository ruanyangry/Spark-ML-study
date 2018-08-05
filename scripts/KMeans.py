# _*_ coding:utf-8 _*_

'''
KMeans
'''

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.appName("KMeans").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

dataset=spark.read.format("libsvm").load(paths+"sample_kmeans_data.txt")

# Trains a k-means model.

kmeans=KMeans().setK(2).setSeed(1)
model=kmeans.fit(dataset)

# Evaluate clustering by computing within set sum of squared errors.

wssse=model.computeCost(dataset)
print("Within Set Sum of Squared Error = "+str(wssse))

# Shows the result

centers=model.clusterCenters()
print("Cluster Centers:")
for center in centers:
	print(center)

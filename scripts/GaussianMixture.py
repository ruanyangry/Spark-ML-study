# _*_ coding:utf-8 _*_

'''
GaussianMixture
'''

from pyspark.sql import SparkSession
from pyspark.ml.clustering import GaussianMixture

spark = SparkSession.builder.appName("GaussianMixture").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

data=spark.read.format("libsvm").load(paths+"sample_kmeans_data.txt")

gmm=GaussianMixture().setK(2)
model=gmm.fit(data)

print("Gaussian: ")
model.gaussiansDF.show()

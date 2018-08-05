# _*_ coding:utf-8 _*_

'''
PCA
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("pca").getOrCreate()

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]

df=spark.createDataFrame(data,["features"])

# n_component = 3

pca=PCA(k=3,inputCol="features",outputCol="pcafeatures")

model=pca.fit(df)

result=model.transform(df).select("pcafeatures")

result.show(truncate=False)

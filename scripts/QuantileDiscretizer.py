# _*_ coding:utf-8 _*_

'''
QuantileDiscretizer
'''

from pyspark.sql import SparkSession
from pyspark.ml.feature import QuantileDiscretizer

spark = SparkSession.builder.appName("QuantileDiscretizer").getOrCreate()

data= [(0, 18.0,), (1, 19.0,), (2, 8.0,), (3, 5.0,), (4, 2.2,)]

df=spark.createDataFrame(data,["id","hour"])

discretizer=QuantileDiscretizer(numBuckets=3,inputCol="hour",outputCol="result")

result=discretizer.fit(df).transform(df)

result.show()

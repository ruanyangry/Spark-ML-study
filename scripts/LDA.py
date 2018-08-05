# _*_ coding:utf-8 _*_

'''
Latent Dirichlet Allocation
'''

from pyspark.sql import SparkSession
from pyspark.ml.clustering import LDA

spark = SparkSession.builder.appName("LDA").getOrCreate()

paths="/export/home/ry/spark-2.2.1-bin-hadoop2.7/data/mllib/"

dataset=spark.read.format("libsvm").load(paths+"sample_lda_libsvm_data.txt")

# Trains a LDA model.

lda=LDA(k=10,maxIter=10)
model=lda.fit(dataset)

ll=model.logLikelihood(dataset)
lp=model.logPerplexity(dataset)

print("The lower bound on the log likelihood of the entire corpus: "+str(ll))
print("The upper bound bound on perplexity: "+str(lp))

# Describe topics
topics=model.describeTopics(3)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)

# shows the result
transformed=model.transform(dataset)
transformed.show(truncate=False)


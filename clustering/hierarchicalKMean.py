from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans


cluster_df = spark.read.csv("./../data/clustering_dataset.csv",header=True, inferSchema=True)
cluster_df.show()



vectoAssembler = VectorAssembler(inputCols=["col1","col2","col3"], outputCol="features")
vcluster_df = vectoAssembler.transform(cluster_df)
vcluster_df.show()


kmeans = KMeans().setK(3)
kmeans = kmeans.setSeed(1)
kmodel = kmeans.fit(vcluster_df)


centers = kmodel.clusterCenters()
centers


from pyspark.ml.clustering import BisectingKMeans
bkMeans = BisectingKMeans().setK(3)
bkMeans = bkMeans.setSeed(1)


bkmodel = bkMeans.fit(vcluster_df)
bkcenteres = bkmodel.clusterCenters()
bkcenteres
centers

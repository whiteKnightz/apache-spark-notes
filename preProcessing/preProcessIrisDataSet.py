from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer


iris_df = spark.read.csv("/../data/iris.data", inferSchema=True)
iris_df.show()

iris_df = iris_df.select(col("_c0").alias("sepal_length"),
                         col("_c1").alias("sepal_width"),
                         col("_c2").alias("petal_length"),
                         col("_c3").alias("petal_width"),
                         col("_c4").alias("species")
                         )

vectorAssembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")
viris_df = vectorAssembler.transform(iris_df)


indexer = StringIndexer(inputCol="species", outputCol="label")
iciris_df = indexer.fit(viris_df).transform(viris_df)
iciris_df.show()

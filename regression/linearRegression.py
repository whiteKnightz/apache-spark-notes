from pyspark.ml.regression import LinearRegression

pp_df = spark.read.csv("data/power_plant.csv", header = True, inferSchema=True)
pp_df.show()

from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols=["AT","V","AP","RH"], outputCol="features")
vpp_df = vectorAssembler.transform(pp_df)
vpp_df.take(1)

lr = LinearRegression(featuresCol="features", labelCol="PE")
lr_model = lr.fit(vpp_df)
lr_model.coefficients
lr_model.intercept
lr_model.summary.rootMeanSquaredError

lr_model.save("lr1.model")


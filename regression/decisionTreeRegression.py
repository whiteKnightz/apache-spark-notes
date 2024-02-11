from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler


pp_df = spark.read.csv("data/power_plant.csv", header = True, inferSchema=True)
pp_df.show()


vectorAssembler = VectorAssembler(inputCols=["AT","V","AP","RH"], outputCol="features")
vpp_df = vectorAssembler.transform(pp_df)
vpp_df.take(1)

splits = vpp_df.randomSplit([0.7, 0.3])

train_db = splits[0]
test_df = splits[1]


train_db.count()
test_df.count()
vpp_df.count()


dt = DecisionTreeRegressor(featuresCol="features", labelCol="PE")
dt_model = dt.fit(train_db)
dt_predictions = dt_model.transform(test_df)


dt_evaluator=RegressionEvaluator(labelCol="PE", predictionCol="prediction", metricName="rmse")


rmse = dt_evaluator.evaluate(dt_predictions)
rmse

